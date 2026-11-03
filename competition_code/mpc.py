"""LTV Frenet MPC on OSQP, with a pure-pursuit fallback.

State x = [e_y, e_psi, v, a_act, k_act], input u = [kappa, a_cmd], linearised about e = 0, v = v_k:
    e_y+   = e_y + dt*v_k*e_psi
    e_psi+ = e_psi + dt*v_k*(k_act - kappa_ref(s_k))
    v+     = v + dt*a_act
    a_act+ = a_act + dt/tau*(a_cmd - a_act)        first-order actuator lag, tau = brake_tau_s
    k_act+ = k_act + dt/tau_k*(kappa - k_act)      steering -> yaw-curvature lag, tau_k = steer_tau_s
v_k = previous solution's predicted speeds shifted one tick (first call / after a
fallback or reset: the model driven by a damped P-law on v_ref, so the trajectory is reachable).
s_k is integrated from v_k.
One-tick delay: the measured state is propagated one step with u_prev and u_0 is the command for the next tick.
a_act and k_act are not measured. FrenetMPC keeps them as internal state: the lag filters run over the issued
commands, which equal the previous solve's predictions at this tick. On the first call they start at u_prev (settled).
Constraints: |kappa_k| <= K(v_k) (steering), a_min <= a_k <= a_drive(v_k), and the lateral-acceleration limit
|kappa| <= a_lat/v^2 as its tangent at v_t = max(v_k, V_TAN_MIN): +-kappa + (2a/v_t^3) v <= 3a/v_t^2. a/v^2 is
convex, so every tangent is an inner bound, and it couples v and kappa: slowing down buys curvature.
|e_y| <= ey_max is soft with an L1 exact penalty (plus a small quadratic term).
Options (cfg keys, absent = off): a_tau_s replaces brake_tau_s as the longitudinal lag; jerk_max bounds
|a_k - a_{k-1}| <= jerk_max*dt (a_{-1} = u_prev); ellipse adds the friction-ellipse facets of tools/build_profile.py,
(a/A_brk)^2 + (v^2 kappa/(c a_lat))^2 <= 1 while braking, as tangent lines at 30/60 deg with v^2 at v_k (c =
brake_ellipse_c): -cos(t) a/A_brk +- sin(t) v_k^2 kappa/(c a_lat) <= 1 (loose for a > 0).
Decision vector z = [x_0..x_N, u_0..u_{N-1}, sig_1..sig_N]. P is constant; A keeps a fixed sparsity and only
its v-dependent entries are rewritten each solve.
"""
import math

import numpy as np
import osqp
import scipy.sparse as sp

NX, NU = 5, 2
SLACK_L1, SLACK_L2 = 1e4, 10.0
ELLIPSE_DEG = (30.0, 60.0)  # 0 deg = the a_min box, 90 deg ~ the lateral tangent rows
# Typical magnitudes of [e_y, e_psi, v, a_act] and [kappa, a_cmd]: the QP is solved in z' = z / mag with rows
# divided by their own magnitude, so OSQP's inf-norm tolerances mean the same thing for kappa (~0.01) as for v.
MAG_X, MAG_U = (1.0, 0.05, 10.0, 5.0, 0.01), (0.01, 5.0)
V_TAN_MIN = 10.0  # m/s, lowest tangent point of the lateral-accel bound (below it K(v) binds anyway)
K_V_FALLBACK = 2.0  # 1/s, fallback speed P-gain


class FrenetMPC:
    def __init__(self, veh, cfg):
        self.veh = veh
        self.N, self.dt = N, dt = int(cfg["N"]), float(cfg["dt"])
        self.c = c = dt / cfg.get("a_tau_s", veh.p["brake_tau_s"])
        self.ck = ck = min(dt / veh.p["steer_tau_s"], 1.0) if veh.p["steer_tau_s"] > 0 else 1.0
        self.a_min = -veh.a_brake * veh.brake_scale
        self.k_lin = 0.5 * c / dt  # 1/(2 tau): damping ~0.7 with the lag, for the first-call linearisation
        self.nx, nu = nx, nu = NX * (N + 1), NU * N
        self.nz = nz = nx + nu + N
        self.iu = iu = nx

        # cost 1/2 z'Pz + q'z
        wx = np.zeros(NX * (N + 1))
        wx[NX:] = np.tile([cfg["q_ey"], cfg["q_epsi"], cfg["q_v"], 0.0, 0.0], N)
        D = sp.eye(N) - sp.eye(N, k=-1)
        self.r = np.array([cfg["r_dkappa"], cfg["r_da"]])
        P = sp.block_diag([sp.diags(2 * wx), sp.kron(2 * D.T @ D, sp.diags(self.r)),
                           sp.diags(np.full(N, 2 * SLACK_L2))])
        self.q_v = cfg["q_v"]

        # constraints l <= Az <= u, built as COO with every entry (incl. v-dependent ones) present
        rows, cols, vals = [], [], []

        def add(r, col, v):
            rows.append(r), cols.append(col), vals.append(v)
            return len(vals) - 1

        for i in range(NX):  # x_0 = propagated measurement
            add(i, i, 1.0)
        vidx = []
        for k in range(N):  # x_{k+1} - A_k x_k - B_k u_k = c_k
            r, xk, uk = NX * (k + 1), NX * k, iu + NU * k
            for i in range(NX):
                add(r + i, xk + NX + i, 1.0)
                add(r + i, xk + i, {3: -(1.0 - c), 4: -(1.0 - ck)}.get(i, -1.0))
            vidx += [add(r, xk + 1, 0.0), add(r + 1, xk + 4, 0.0)]
            add(r + 2, xk + 3, -dt)
            add(r + 3, uk + 1, -c)
            add(r + 4, uk, -ck)
        for j in range(nu):  # input boxes
            add(nx + j, iu + j, 1.0)
        self.rl = rl = nx + nu
        lidx = []
        for k in range(N):  # lateral-accel tangent: +-kappa_k + c_k v_k <= b_k
            for sgn in (1.0, -1.0):
                add(rl + 2 * k + (sgn < 0), iu + NU * k, sgn)
                lidx.append(add(rl + 2 * k + (sgn < 0), NX * k + 2, 0.0))
        ro, isg = rl + 2 * N, nx + nu
        for k in range(N):  # soft |e_y_{k+1}| <= ey_max, sig >= 0
            add(ro + 2 * k, NX * (k + 1), 1.0), add(ro + 2 * k, isg + k, -1.0)
            add(ro + 2 * k + 1, NX * (k + 1), 1.0), add(ro + 2 * k + 1, isg + k, 1.0)
            add(ro + 2 * N + k, isg + k, 1.0)
        self.rj = rj = ro + 3 * N
        self.jerk = cfg.get("jerk_max")
        nj = N - 1 if self.jerk else 0
        for k in range(nj):  # a_{k+1} - a_k (k = 0 vs u_prev is folded into the input box)
            add(rj + k, iu + NU * (k + 1) + 1, 1.0), add(rj + k, iu + NU * k + 1, -1.0)
        self.re = re = rj + nj
        th = np.radians(ELLIPSE_DEG) if cfg.get("ellipse") else np.zeros(0)
        self.e_sin = np.sin(th) / veh.p.get("brake_ellipse_c", 1.0)
        eidx = []
        for k in range(N if len(th) else 0):
            for j, t in enumerate(th):
                for sgn in (0, 1):
                    r = re + 2 * len(th) * k + 2 * j + sgn
                    add(r, iu + NU * k + 1, -math.cos(t) / -self.a_min)
                    eidx.append(add(r, iu + NU * k, 0.0))
        ne = 2 * len(th) * N
        self.m = m = re + ne
        self.sz = sz = np.r_[np.tile(MAG_X, N + 1), np.tile(MAG_U, N), np.ones(N)]
        self.sr = sr = np.r_[np.tile(MAG_X, N + 1), np.tile(MAG_U, N), np.full(2 * N, MAG_U[0]), np.ones(3 * N),
                             np.full(nj, MAG_U[1]), np.ones(ne)]
        self.sc = sz[cols] / sr[rows]  # per-entry scale of A' = diag(1/sr) A diag(sz)
        self.vals, self.vidx, self.lidx = np.array(vals), np.array(vidx).reshape(N, 2), np.array(lidx).reshape(N, 2)
        self.eidx = np.array(eidx, int).reshape(N, len(th), 2)
        self.e_sgn = np.array([1.0, -1.0])
        A = sp.csc_matrix((np.arange(1.0, len(vals) + 1), (rows, cols)), shape=(m, nz))
        A.sort_indices()
        self.perm = A.data.astype(int) - 1  # CSC slot -> COO entry
        self.vals[self.vidx] = -dt * 30.0  # real values at a nominal speed: OSQP's scaling is fixed at setup
        self.vals[self.lidx] = 2 * veh.a_lat_max / 30.0 ** 3
        self._set_ellipse(np.full(N, 30.0), np.full(N, veh.a_lat_max))
        A.data = (self.vals * self.sc)[self.perm]
        P = sp.diags(sz) @ P @ sp.diags(sz)
        self.l, self.u = np.zeros(m), np.zeros(m)
        ey_max = cfg["ey_max"]
        self.l[rl:ro] = -np.inf
        self.l[ro:ro + 2 * N:2], self.u[ro:ro + 2 * N:2] = -np.inf, ey_max
        self.l[ro + 1:ro + 2 * N:2], self.u[ro + 1:ro + 2 * N:2] = -ey_max, np.inf
        self.u[ro + 2 * N:rj] = np.inf
        self.l[rj:re], self.u[rj:re] = -(self.jerk or 0) * dt, (self.jerk or 0) * dt
        self.l[re:], self.u[re:] = -np.inf, 1.0
        self.q = np.zeros(nz)
        self.q[isg:] = SLACK_L1

        self.solver = osqp.OSQP()
        # Deterministic: fixed max_iter and no time_limit (the sim is synchronous, wall time doesn't score).
        # check_dualgap off: with the grip limit binding and the slack active, objectives reach ~1e5 and OSQP 1.x's
        # duality-gap test held converged iterates to max_iter (12 % of near-limit cold solves). Termination still
        # needs the primal/dual residuals within eps_abs/eps_rel, which the MAG scaling makes meaningful per row.
        self.solver.setup(sp.triu(P, format="csc"), self.q * sz, A, self.l / sr, self.u / sr, verbose=False,
                          warm_starting=True, polishing=False, eps_abs=1e-4, eps_rel=5e-4, max_iter=4000,
                          adaptive_rho_interval=25, check_dualgap=False)
        self.reset()

    def reset(self):
        """Forget the linearisation, the actuator estimate and the warm start (respawn / teleport)."""
        self._a_act = self._k_act = None
        self._v_pred = None
        self.solver.warm_start(x=np.zeros(self.nz), y=np.zeros(self.m))

    def solve(self, e_y, e_psi, v, s_total, kappa_ref_fn, v_ref_fn, u_prev, a_lat_fn=None):
        N, dt, veh, nx, iu = self.N, self.dt, self.veh, self.nx, self.iu
        a_lat_fn = a_lat_fn or (lambda s: veh.a_lat_max)
        k_prev, a_prev = float(u_prev[0]), float(u_prev[1])
        a_act = a_prev if self._a_act is None else self._a_act
        k_act = k_prev if self._k_act is None else self._k_act
        x0 = np.array([e_y + dt * v * e_psi, e_psi + dt * v * (k_act - kappa_ref_fn(s_total)),
                       v + dt * a_act, a_act + self.c * (a_prev - a_act), k_act + self.ck * (k_prev - k_act)])
        self._a_act = x0[3] if math.isfinite(x0[3]) else None
        self._k_act = x0[4] if math.isfinite(x0[4]) else None
        s0 = s_total + dt * v

        vl = np.empty(N + 1)
        vl[0] = max(x0[2], 0.0) if math.isfinite(x0[2]) else math.nan  # max(nan, 0) would hide a NaN
        if self._v_pred is not None:
            vl[1:] = np.maximum(np.r_[self._v_pred[2:], self._v_pred[-1]], 0.0)
            s = s0 + dt * np.r_[0.0, np.cumsum(vl[:-1])]
        else:  # simulate the model under a damped P-law on v_ref: a reachable trajectory, so the tangent
            s = np.empty(N + 1)  # bound (feasible only for v <= 1.5 v_t) stays feasible
            s[0], ak = s0, x0[3]
            for k in range(N):
                s[k + 1] = s[k] + dt * vl[k]
                ac = min(max(self.k_lin * (v_ref_fn(s[k + 1]) - vl[k]), self.a_min), veh.a_drive(vl[k]))
                vl[k + 1] = max(vl[k] + dt * ak, 0.0)
                ak += self.c * (ac - ak)
        kr = np.array([kappa_ref_fn(x) for x in s[:N]], float)
        vr = np.array([v_ref_fn(x) for x in s[1:]], float)
        al = np.array([a_lat_fn(x) for x in s[:N]], float)
        if not all(np.isfinite(a).all() for a in (x0, vl, kr, vr, al)):
            return self._fallback(x0, kr[0], vr[0], al[0])  # never let a NaN into the solver
        ks = np.array([veh.K(x) for x in vl[:N]])
        ad = np.maximum([veh.a_drive(x) for x in vl[:N]], self.a_min)
        vt = np.maximum(vl[:N], V_TAN_MIN)

        self.vals[self.vidx] = -dt * vl[:N, None]
        self.vals[self.lidx] = (2 * al / vt ** 3)[:, None]
        self._set_ellipse(vl[:N], al)
        q = self.q
        q[NX + 2:nx:NX] = -2 * self.q_v * vr
        q[iu:iu + NU] = -2 * self.r * (k_prev, a_prev)
        l, u, rl = self.l, self.u, self.rl
        l[:NX] = u[:NX] = x0
        l[NX + 1:nx:NX] = u[NX + 1:nx:NX] = -dt * vl[:N] * kr
        l[nx:rl:2], u[nx:rl:2] = -ks, ks
        l[nx + 1:rl:2], u[nx + 1:rl:2] = self.a_min, ad
        if self.jerk:  # a_0 within jerk_max*dt of the command being applied now (empty -> keep the box's upper end)
            u[nx + 1] = min(ad[0], a_prev + self.jerk * dt)
            l[nx + 1] = min(max(self.a_min, a_prev - self.jerk * dt), u[nx + 1])
        u[rl:rl + 2 * N] = np.repeat(3 * al / vt ** 2, 2)
        self.solver.update(Ax=(self.vals * self.sc)[self.perm], q=q * self.sz, l=l / self.sr, u=u / self.sr)
        z = self._run()
        if z is None:
            return self._fallback(x0, kr[0], vr[0], al[0])
        self._v_pred = z[2:nx:NX].copy()
        km0 = min(ks[0], al[0] / max(x0[2], 1.0) ** 2)
        kappa = float(np.clip(z[iu], -km0, km0))
        a = float(np.clip(z[iu + 1], l[nx + 1], u[nx + 1]))
        return kappa, a, True

    def _set_ellipse(self, v, a_lat):
        """kappa coefficients of the ellipse facets: +-sin(t)/c * v^2/a_lat."""
        if self.eidx.size:
            self.vals[self.eidx] = (v ** 2 / a_lat)[:, None, None] * self.e_sin[None, :, None] * self.e_sgn

    def _run(self):
        """Solve; the primal solution, or None unless OSQP_SOLVED (non-finite iterates also reset the warm start)."""
        res = self.solver.solve(raise_error=False)
        if not np.all(np.isfinite(res.x)):
            self.solver.warm_start(x=np.zeros(self.nz), y=np.zeros(self.m))
            return None
        return res.x * self.sz if res.info.status_val == osqp.SolverStatus.OSQP_SOLVED else None

    def _fallback(self, x0, kref, vref, a_lat):
        """Pure pursuit on a lookahead point of the line (circle-arc approximation) + P speed control."""
        self._v_pred = None
        e_y, e_psi, v, kref, vref, a_lat = np.nan_to_num([*x0[:3], kref, vref, a_lat], nan=0.0, posinf=0.0, neginf=0.0)
        v = max(v, 0.0)
        a_lat = a_lat if a_lat > 0 else self.veh.a_lat_max
        Ld = max(10.0, 0.6 * v)
        alpha = math.atan2(0.5 * kref * Ld * Ld - e_y, Ld) - e_psi
        km = min(self.veh.K(v), a_lat / max(v, 1.0) ** 2)
        kappa = float(np.clip(2 * math.sin(alpha) / Ld, -km, km))
        a = float(np.clip(K_V_FALLBACK * (vref - v), self.a_min, self.veh.a_drive(v)))
        return kappa, a, False
