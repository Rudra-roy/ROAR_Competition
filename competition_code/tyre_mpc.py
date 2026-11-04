"""Slip-aware LTV Frenet MPC on OSQP: dynamic bicycle model with saturating tyres (vehicle_params["tyre"],
fitted by tools/fit_tyres.py). Selected with vehicle_params["mpc"]["model"] == "tyre"; otherwise mpc.FrenetMPC runs.

State x = [e_y, e_psi, v, a_act, delta, beta, r], input u = [delta_cmd, a_cmd]  (delta: front steer angle, left +;
beta: sideslip = course - yaw; r: yaw rate). Euler step dt, linearised about a nominal trajectory: the nonlinear
model rolled out from the measured state under the previous solution's inputs shifted one tick (after a reset: the
slip-clipped steady-state steer for kappa_ref and a damped P-law on v_ref), with an SQP trust region on delta_cmd:
    e_y+   = e_y + dt v_k (e_psi + beta)              the car moves along its COURSE, not its nose
    e_psi+ = e_psi + dt (r - v kappa_ref(s_k))          v the state: braking lowers the yaw rate the line needs
    v+     = v + dt a_act;  a_act+ = a_act + c (a_cmd - a_act);  delta+ = delta + c_d (delta_cmd - delta)
    beta+  = beta + dt ((F_f + F_r)/v_k - r);  r+ = r + dt (a F_f - b F_r)/I
    alpha_f = delta - beta - a r/v_k, alpha_r = -beta + b r/v_k, F = mu tanh(C alpha/mu) ~ F0 + G (alpha - alpha0)
Near the tyres' peak G -> 0: the QP sees that more steer buys no more force, which the kinematic model never did.
Constraints: |delta_cmd| <= g(v_k) (full lock), accel box (+ jerk), soft |alpha_f|, |alpha_r| <= alpha_max (L1
slack: the grippy side of the peak), soft |a_y| <= a_lat_margin a_lat(s) with a_y = F_f + F_r (the per-section grip
the planner used), soft |e_y| <= ey_max. Cost on e_y, e_psi, v - v_ref, beta, and input moves.
Decision vector z = [x_0..x_N, u_0..u_{N-1}, sig_alpha_1..N, sig_ey_1..N, sig_ay_1..N].
"""
import math

import numpy as np
import osqp
import scipy.sparse as sp

NX, NU = 7, 2
SLACK_L1, SLACK_L2 = 1e4, 10.0
MAG_X, MAG_U = (1.0, 0.05, 10.0, 5.0, 0.05, 0.05, 0.5), (0.05, 5.0)
K_V_FALLBACK = 2.0
MAX_ITER = 8000
F_SAT = 0.95  # inverse tyre map clip (|F| < F_SAT mu) for steady-state nominals
STEER_HOLD_V = 2.0  # m/s: below this the steer command is 0 (no wind-up while the car sits at the spawn)
STEER_CAP_LOW = (0.2, 8.0, 12.0)  # |delta| <= 0.2 rad below 8 m/s, ramping to full lock by 12 m/s: a 2 m spawn
# offset is closed in a gentle arc instead of full lock at walking pace, a 14 deg slide and a counter-steer
V_FLOOR = 10.0  # m/s: the lateral model's 1/v terms use max(v, V_FLOOR) (Euler at dt 0.05 is stable above ~8 m/s;
# below it the fit had no data and the car is pulling away in a straight line anyway)


class Tyres:
    def __init__(self, t):
        self.Cf, self.Cr, self.mu_f, self.mu_r = float(t["Cf"]), float(t["Cr"]), float(t["mu_f"]), float(t["mu_r"])
        self.a, self.b, self.I, self.tau_d = float(t["a"]), float(t["b"]), float(t["I"]), float(t["tau_d"])
        self.g_v, self.g_k = np.asarray(t["g_v"], float), np.asarray(t["g_k"], float)
        self.k_lt = float(t.get("k_lt", 0.0))  # braking load transfer: front grip x (1 - k ax/g), rear x (1 + k ax/g)

    def g(self, v):
        return float(np.interp(v, self.g_v, self.g_k))

    def forces(self, alpha_f, alpha_r, ax=0.0):
        """F_f, F_r and their slopes G_f, G_r at the given slip angles (per unit mass), at longitudinal accel ax."""
        uf, ur = self.Cf * alpha_f / self.mu_f, self.Cr * alpha_r / self.mu_r
        tf, tr = np.tanh(uf), np.tanh(ur)
        wf, wr = 1.0 - self.k_lt * ax / 9.81, 1.0 + self.k_lt * ax / 9.81
        return wf * self.mu_f * tf, wr * self.mu_r * tr, wf * self.Cf * (1 - tf * tf), wr * self.Cr * (1 - tr * tr)

    def alpha_for(self, F, C, mu):
        return mu / C * math.atanh(max(min(F / mu, F_SAT), -F_SAT))

    def steady(self, kappa, v, alpha_max=None):
        """Steady-state (delta, beta, r) on a circle of curvature kappa at speed v. With alpha_max the slip angles
        are clipped to it: a linearisation point on the grippy side of the peak even when the corner is over speed
        (at the saturated point the tyre slope is ~0 and the QP would think slip is free to shed)."""
        v = max(v, V_FLOOR)
        L = self.a + self.b
        ay = v * v * kappa
        af = self.alpha_for(self.b * ay / L, self.Cf, self.mu_f)
        ar = self.alpha_for(self.a * ay / L, self.Cr, self.mu_r)
        if alpha_max is not None:
            af, ar = max(min(af, alpha_max), -alpha_max), max(min(ar, alpha_max), -alpha_max)
        r = v * kappa
        beta = self.b * kappa - ar
        return af + beta + self.a * kappa, beta, r

    def step(self, x, u, kappa_ref, dt, c, cd):
        """Nonlinear Euler step of the full state (v taken from the state)."""
        e_y, e_psi, v, a_act, d, beta, r = x
        vv = max(v, V_FLOOR)
        Ff, Fr, _, _ = self.forces(d - beta - self.a * r / vv, -beta + self.b * r / vv, a_act)
        return np.array([e_y + dt * v * (e_psi + beta), e_psi + dt * (r - v * kappa_ref), v + dt * a_act,
                         a_act + c * (u[1] - a_act), d + cd * (u[0] - d),
                         beta + dt * ((Ff + Fr) / vv - r), r + dt * (self.a * Ff - self.b * Fr) / self.I])


class TyreMPC:
    def __init__(self, veh, cfg):
        self.veh = veh
        self.ty = ty = Tyres(veh.p["tyre"])
        self.N, self.dt = N, dt = int(cfg["N"]), float(cfg["dt"])
        self.c = c = dt / cfg.get("a_tau_s", veh.p["brake_tau_s"])
        self.cd = cd = min(dt / ty.tau_d, 1.0)
        self.a_min = -veh.a_brake * veh.brake_scale
        self.a_lat_margin = float(cfg.get("a_lat_margin", 1.0))
        self.alpha_max = math.radians(float(cfg.get("alpha_max_deg", 7.0)))
        self.jerk = cfg.get("jerk_max")
        # brake_ellipse: braking shares grip with cornering, (a/A)^2 + (a_y/(c a_lat))^2 <= 1 with a_y the nominal
        # trajectory's lateral force (c = brake_ellipse_c): full braking in a straight line, less as the car turns
        self.brake_ellipse = bool(cfg.get("brake_ellipse", 0))
        self.k_lin = 0.5 * c / dt
        self.nx, nu = nx, nu = NX * (N + 1), NU * N
        self.iu = iu = nx
        self.isa, self.ise, self.iay = isa, ise, iay = nx + nu, nx + nu + N, nx + nu + 2 * N
        self.nz = nz = nx + nu + 3 * N

        # stage cost q_ey e_y^2 + q_epsi (e_psi + beta)^2 + q_v (v - v_ref)^2 + q_beta beta^2: the COURSE error is
        # what moves the car off the line; the nose sits at e_psi = -beta on a steady corner
        W = np.zeros((NX, NX))
        W[0, 0], W[2, 2] = cfg["q_ey"], cfg["q_v"]
        W[1, 1] = W[1, 5] = W[5, 1] = cfg["q_epsi"]
        W[5, 5] = cfg["q_epsi"] + cfg.get("q_beta", 0.0)
        W[6, 6] = self.q_r = float(cfg.get("q_r", 0.0))  # (r - v kappa_ref)^2: damps the yaw mode the plan can excite
        D = sp.eye(N) - sp.eye(N, k=-1)
        self.r = np.array([cfg.get("r_ddelta", 2e3), cfg["r_da"]])
        self.trust = float(cfg.get("trust_delta", 0.04))  # rad, |delta_cmd - nominal| per solve (SQP trust region)
        P = sp.block_diag([sp.csc_matrix((NX, NX)), sp.kron(sp.eye(N), 2 * W), sp.kron(2 * D.T @ D, sp.diags(self.r)),
                           sp.diags(np.full(3 * N, 2 * SLACK_L2))])
        self.q_v = cfg["q_v"]

        rows, cols, vals = [], [], []

        def add(r, col, v):
            rows.append(r), cols.append(col), vals.append(v)
            return len(vals) - 1

        for i in range(NX):
            add(i, i, 1.0)
        eyidx, latidx, psidx = [], [], []
        for k in range(N):  # x_{k+1} - A_k x_k - B_k u_k = c_k
            r0, xk, uk = NX * (k + 1), NX * k, iu + NU * k
            for i in range(NX):
                add(r0 + i, xk + NX + i, 1.0)
            add(r0, xk, -1.0)
            eyidx += [add(r0, xk + 1, 0.0), add(r0, xk + 5, 0.0)]  # -dt v (e_psi, beta)
            add(r0 + 1, xk + 1, -1.0), add(r0 + 1, xk + 6, -dt)
            psidx.append(add(r0 + 1, xk + 2, 0.0))  # +dt kappa_ref_k on v
            add(r0 + 2, xk + 2, -1.0), add(r0 + 2, xk + 3, -dt)
            add(r0 + 3, xk + 3, -(1.0 - c)), add(r0 + 3, uk + 1, -c)
            add(r0 + 4, xk + 4, -(1.0 - cd)), add(r0 + 4, uk, -cd)
            for i in (5, 6):  # beta, r rows: -(A) on delta, beta, r (diagonal folded in)
                latidx += [add(r0 + i, xk + j, 0.0) for j in (4, 5, 6)]
        for j in range(nu):
            add(nx + j, iu + j, 1.0)
        self.rs = rs = nx + nu
        sidx = []
        for k in range(N):  # slip rows on x_{k+1}: alpha_f -+ sig <= +-alpha_max, alpha_r likewise
            xk1 = NX * (k + 1)
            for j, (cd_, cb) in enumerate(((1.0, -1.0), (0.0, -1.0))):  # (delta, beta) coefficients of alpha_f/alpha_r
                for sgn in (1.0, -1.0):
                    rr = rs + 4 * k + 2 * j + (sgn < 0)
                    if cd_:
                        add(rr, xk1 + 4, cd_)
                    add(rr, xk1 + 5, cb)
                    sidx.append(add(rr, xk1 + 6, 0.0))
                    add(rr, isa + k, -sgn)
        self.ro = ro = rs + 4 * N
        for k in range(N):  # soft |e_y_{k+1}| <= ey_max
            add(ro + 2 * k, NX * (k + 1), 1.0), add(ro + 2 * k, ise + k, -1.0)
            add(ro + 2 * k + 1, NX * (k + 1), 1.0), add(ro + 2 * k + 1, ise + k, 1.0)
        self.ray = ray = ro + 2 * N
        ayidx = []
        for k in range(N):  # soft |a_y_{k+1}| <= a_lat: coefficients on delta, beta, r of x_{k+1}
            xk1 = NX * (k + 1)
            for sgn in (1.0, -1.0):
                rr = ray + 2 * k + (sgn < 0)
                ayidx += [add(rr, xk1 + j, 0.0) for j in (4, 5, 6)]
                add(rr, iay + k, -sgn)
        self.rsl = rsl = ray + 2 * N
        for k in range(3 * N):  # slacks >= 0
            add(rsl + k, isa + k, 1.0)
        self.rj = rj = rsl + 3 * N
        nj = N - 1 if self.jerk else 0
        for k in range(nj):
            add(rj + k, iu + NU * (k + 1) + 1, 1.0), add(rj + k, iu + NU * k + 1, -1.0)
        self.m = m = rj + nj
        self.sz = sz = np.r_[np.tile(MAG_X, N + 1), np.tile(MAG_U, N), np.ones(3 * N)]
        self.sr = sr = np.r_[np.tile(MAG_X, N + 1), np.tile(MAG_U, N), np.full(4 * N, MAG_X[5]), np.ones(2 * N),
                             np.full(2 * N, 10.0), np.ones(3 * N), np.full(nj, MAG_U[1])]
        self.sc = sz[cols] / sr[rows]
        self.vals = np.array(vals)
        self.eyidx, self.latidx = np.array(eyidx).reshape(N, 2), np.array(latidx).reshape(N, 2, 3)
        self.psidx = np.array(psidx)
        self.sidx, self.ayidx = np.array(sidx).reshape(N, 2, 2), np.array(ayidx).reshape(N, 2, 3)
        A = sp.csc_matrix((np.arange(1.0, len(vals) + 1), (rows, cols)), shape=(m, nz))
        A.sort_indices()
        self.perm = A.data.astype(int) - 1
        X0 = np.zeros((N + 1, NX))
        X0[:, 2] = 30.0
        self._write_dynamic(X0, np.zeros(N))
        A.data = (self.vals * self.sc)[self.perm]
        P = sp.diags(sz) @ P @ sp.diags(sz)
        self.l, self.u = np.zeros(m), np.zeros(m)
        ey_max = cfg["ey_max"]
        self.l[ro:ro + 2 * N:2], self.u[ro:ro + 2 * N:2] = -np.inf, ey_max
        self.l[ro + 1:ro + 2 * N:2], self.u[ro + 1:ro + 2 * N:2] = -ey_max, np.inf
        self.u[rsl:rj] = np.inf
        self.l[rj:m], self.u[rj:m] = -(self.jerk or 0) * dt, (self.jerk or 0) * dt
        self.q = np.zeros(nz)
        self.q[isa:] = SLACK_L1
        self.solver = osqp.OSQP()
        self.solver.setup(sp.triu(P, format="csc"), self.q * sz, A, self.l / sr, self.u / sr, verbose=False,
                          warm_starting=True, polishing=False, eps_abs=1e-4, eps_rel=5e-4, max_iter=MAX_ITER,
                          adaptive_rho_interval=25, check_dualgap=False)
        self.reset()

    def reset(self):
        self._a_act = self._d_act = None
        self._z = self._U = None
        self.solver.warm_start(x=np.zeros(self.nz), y=np.zeros(self.m))

    # -- linearisation ------------------------------------------------------------------------------------------------
    def _rollout(self, x0, s0, kappa_ref_fn, v_ref_fn, U=None):
        """Nominal trajectory: the nonlinear model from x0 under U (N, 2); U None = first call: steady-state steer for
        kappa_ref (slip clipped to alpha_max) and a damped P-law on v_ref. -> X (N+1, NX), U (N, 2), s (N+1)."""
        N, dt, veh, ty = self.N, self.dt, self.veh, self.ty
        X, s = np.empty((N + 1, NX)), np.empty(N + 1)
        X[0], s[0] = x0, s0
        Uo = np.empty((N, NU))
        for k in range(N):
            v = max(X[k, 2], 0.0)
            kr = kappa_ref_fn(s[k])
            if U is None:
                a = min(max(self.k_lin * (v_ref_fn(s[k] + dt * v) - v), self.a_min), veh.a_drive(v))
                d = ty.steady(kr, v, self.alpha_max)[0]
            else:
                d, a = U[k]
            Uo[k] = (d, a)
            X[k + 1] = ty.step(X[k], (d, a), kr, dt, self.c, self.cd)
            X[k + 1, 2] = max(X[k + 1, 2], 0.0)
            s[k + 1] = s[k] + dt * v
        return X, Uo, s

    def _write_dynamic(self, X, kr):
        """Rewrite the A entries that depend on the nominal X (N+1, NX) and kappa_ref; returns the affine terms of
        the beta and r rows and the constant of the a_y linearisation (on x_{k+1})."""
        ty, dt, N = self.ty, self.dt, self.N
        v = np.maximum(X[:N, 2], V_FLOOR)
        d0, b0, r0 = X[:N, 4], X[:N, 5], X[:N, 6]
        af0, ar0 = d0 - b0 - ty.a * r0 / v, -b0 + ty.b * r0 / v
        Ff, Fr, Gf, Gr = ty.forces(af0, ar0, X[:N, 3])
        self.vals[self.psidx] = dt * kr
        self.vals[self.eyidx] = (-dt * np.maximum(X[:N, 2], 0.0))[:, None]
        self.vals[self.latidx[:, 0, :]] = -np.c_[dt * Gf / v, 1 - dt * (Gf + Gr) / v,
                                               dt * (-Gf * ty.a + Gr * ty.b) / v ** 2 - dt]
        self.vals[self.latidx[:, 1, :]] = -np.c_[dt * ty.a * Gf / ty.I, dt * (-ty.a * Gf + ty.b * Gr) / ty.I,
                                               1 - dt * (ty.a ** 2 * Gf + ty.b ** 2 * Gr) / (ty.I * v)]
        c_beta = dt * (Ff + Fr - Gf * af0 - Gr * ar0) / v
        c_r = dt * (ty.a * (Ff - Gf * af0) - ty.b * (Fr - Gr * ar0)) / ty.I
        v1 = np.maximum(X[1:, 2], V_FLOOR)
        self.vals[self.sidx[:, 0, :]] = (-ty.a / v1)[:, None]
        self.vals[self.sidx[:, 1, :]] = (ty.b / v1)[:, None]
        d1, b1, r1 = X[1:, 4], X[1:, 5], X[1:, 6]
        af1, ar1 = d1 - b1 - ty.a * r1 / v1, -b1 + ty.b * r1 / v1
        Ff1, Fr1, Gf1, Gr1 = ty.forces(af1, ar1, X[1:, 3])
        self.vals[self.ayidx] = np.c_[Gf1, -(Gf1 + Gr1), (-Gf1 * ty.a + Gr1 * ty.b) / v1][:, None, :]
        return c_beta, c_r, Ff1 + Fr1 - Gf1 * af1 - Gr1 * ar1

    def _delta_cap(self, v, g):
        """Per-step |delta_cmd| bound: 0 when stationary, a low-speed cap, then full lock g(v)."""
        c0, v0, v1 = STEER_CAP_LOW
        w = np.clip((v - v0) / (v1 - v0), 0.0, 1.0)
        return np.where(v < STEER_HOLD_V, 0.0, np.minimum(g, c0 + w * np.maximum(g - c0, 0.0)))

    # -- solve ----------------------------------------------------------------------------------------------------------
    def solve(self, e_y, e_psi, v, beta, r, s_total, kappa_ref_fn, v_ref_fn, u_prev, a_lat_fn=None, a_brk_fn=None):
        """-> (delta_cmd [rad, left +], a_cmd, ok). u_prev = (delta, a) issued last tick. a_brk_fn(s_total): braking
        scale of the section (per-section pedal braking); None = 1."""
        N, dt, veh, ty, nx, iu = self.N, self.dt, self.veh, self.ty, self.nx, self.iu
        a_lat_fn = a_lat_fn or (lambda s: veh.a_lat_max)
        d_prev, a_prev = float(u_prev[0]), float(u_prev[1])
        a_act = a_prev if self._a_act is None else self._a_act
        d_act = d_prev if self._d_act is None else self._d_act
        kr0 = kappa_ref_fn(s_total)
        x0 = ty.step(np.array([e_y, e_psi, v, a_act, d_act, beta, r]), (d_prev, a_prev), kr0, dt, self.c, self.cd)
        if not np.isfinite(x0).all():
            self._a_act = self._d_act = None
            return self._fallback(np.nan_to_num(x0), kr0, v_ref_fn(s_total), a_lat_fn(s_total))
        self._a_act, self._d_act = x0[3], x0[4]
        s0 = s_total + dt * v
        U_prev = None if self._U is None else np.r_[self._U[1:], self._U[-1:]]
        X, Ub, s = self._rollout(x0, s0, kappa_ref_fn, v_ref_fn, U_prev)
        kr = np.array([kappa_ref_fn(x) for x in s[:N]], float)
        vr = np.array([v_ref_fn(x) for x in s[1:]], float)
        al_raw = np.array([a_lat_fn(x) for x in s[:N]], float)
        al = self.a_lat_margin * al_raw
        if not all(np.isfinite(a).all() for a in (X, kr, vr, al)):
            return self._fallback(x0, kr0, vr[0], al_raw[0])
        c_beta, c_r, ay_c = self._write_dynamic(X, kr)
        vl = X[:, 2]
        g = np.array([ty.g(x) for x in vl[:N]])
        bs = np.array([a_brk_fn(x) for x in s[:N]], float) if a_brk_fn else np.ones(N)
        a_lo = -np.array([veh.a_brake_max(x) for x in vl[:N]]) * bs
        if self.brake_ellipse:
            c_el = veh.p.get("brake_ellipse_c", 1.0)
            ay_nom = np.abs(ay_c + (self.vals[self.ayidx[:, 0, :]] * X[1:, 4:7]).sum(1))  # a_y at the nominal x_{k+1}
            share = np.sqrt(np.clip(1.0 - (ay_nom / (c_el * np.maximum(al, 1.0))) ** 2, 0.15, 1.0))
            a_lo = a_lo * share
        ad = np.maximum([veh.a_drive(x) for x in vl[:N]], a_lo)
        q = self.q
        q[NX + 2:nx:NX] = -2 * self.q_v * vr
        q[NX + 6:nx:NX] = -2 * self.q_r * vl[1:] * np.r_[kr[1:], kr[-1]]
        q[iu:iu + NU] = -2 * self.r * (d_prev, a_prev)
        l, u = self.l, self.u
        l[:NX] = u[:NX] = x0
        l[NX + 5:nx:NX] = u[NX + 5:nx:NX] = c_beta
        l[NX + 6:nx:NX] = u[NX + 6:nx:NX] = c_r
        rs = self.rs
        gc = self._delta_cap(vl[:N], g)
        l[nx:rs:2] = np.maximum(Ub[:, 0] - self.trust, -gc)  # trust region round the nominal steer, within the cap
        u[nx:rs:2] = np.maximum(np.minimum(Ub[:, 0] + self.trust, gc), l[nx:rs:2])
        l[nx + 1:rs:2], u[nx + 1:rs:2] = a_lo, ad
        if self.jerk:
            u[nx + 1] = min(ad[0], a_prev + self.jerk * dt)
            l[nx + 1] = min(max(a_lo[0], a_prev - self.jerk * dt), u[nx + 1])
        am = self.alpha_max
        l[rs:self.ro:2], u[rs:self.ro:2] = -np.inf, am
        l[rs + 1:self.ro:2], u[rs + 1:self.ro:2] = -am, np.inf
        ray = self.ray
        l[ray:ray + 2 * N:2], u[ray:ray + 2 * N:2] = -np.inf, al - ay_c
        l[ray + 1:ray + 2 * N:2], u[ray + 1:ray + 2 * N:2] = -al - ay_c, np.inf
        self.solver.update(Ax=(self.vals * self.sc)[self.perm], q=q * self.sz, l=l / self.sr, u=u / self.sr)
        res = self.solver.solve(raise_error=False)
        if not np.all(np.isfinite(res.x)) or res.info.status_val != osqp.SolverStatus.OSQP_SOLVED:
            self.solver.warm_start(x=np.zeros(self.nz), y=np.zeros(self.m))
            return self._fallback(x0, kr0, vr[0], al_raw[0])
        z = res.x * self.sz
        self._z = z
        self._U = z[iu:iu + NU * N].reshape(N, NU).copy()
        delta = float(np.clip(z[iu], l[nx], u[nx]))
        a = float(np.clip(z[iu + 1], l[nx + 1], u[nx + 1]))
        return delta, a, True

    def _fallback(self, x0, kref, vref, a_lat):
        """Pure pursuit on the line + P speed control, mapped to a steady-state steer angle."""
        self._z = self._U = None
        e_y, e_psi, v, kref, vref, a_lat = np.nan_to_num([*x0[:3], kref, vref, a_lat], nan=0.0, posinf=0.0, neginf=0.0)
        v = max(v, 0.0)
        a_lat = a_lat if a_lat > 0 else self.veh.a_lat_max
        Ld = max(10.0, 0.6 * v)
        alpha = math.atan2(0.5 * kref * Ld * Ld - e_y, Ld) - e_psi
        km = a_lat / max(v, 1.0) ** 2
        kappa = float(np.clip(2 * math.sin(alpha) / Ld, -km, km))
        g = float(self._delta_cap(np.array([v]), np.array([self.ty.g(v)]))[0])
        delta = float(np.clip(self.ty.steady(kappa, v)[0], -g, g))
        a = float(np.clip(K_V_FALLBACK * (vref - v), self.a_min, self.veh.a_drive(v)))
        return delta, a, False
