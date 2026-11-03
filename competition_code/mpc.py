"""LTV Frenet MPC on OSQP, with a pure-pursuit fallback.

State x = [e_y, e_psi, v, a_act], input u = [kappa, a_cmd], linearised about e = 0, v = v_k:
    e_y+   = e_y + dt*v_k*e_psi
    e_psi+ = e_psi + dt*v_k*(kappa - kappa_ref(s_k))
    v+     = v + dt*a_act
    a_act+ = a_act + dt/tau*(a_cmd - a_act)        first-order actuator lag, tau = brake_tau_s
Curvature acts with no lag. v_k = previous solution's predicted speeds shifted one tick (first call / after a
fallback: v_ref along the predicted s); s_k is integrated from v_k.
One-tick delay: the measured state is propagated one step with u_prev and u_0 is the command for the next tick.
a_act is not measured. FrenetMPC keeps it as internal state: the lag filter run over the issued commands, which
equals the previous solve's predicted a_act at this tick. On the first call it starts at u_prev[1] (lag settled).
Decision vector z = [x_0..x_N, u_0..u_{N-1}, sig_1..sig_N], sig = slack of the soft |e_y| <= ey_max.
P is constant; A keeps a fixed sparsity and only its v_k entries are rewritten each solve.
"""
import math

import numpy as np
import osqp
import scipy.sparse as sp

NX, NU = 4, 2
SLACK_W = 1e6
K_V_FALLBACK = 2.0  # 1/s, fallback speed P-gain


class FrenetMPC:
    def __init__(self, veh, cfg):
        self.veh = veh
        self.N, self.dt = N, dt = int(cfg["N"]), float(cfg["dt"])
        self.c = c = dt / veh.p["brake_tau_s"]
        self.a_min = -veh.a_brake * veh.brake_scale
        self.nx, nu = nx, nu = NX * (N + 1), NU * N
        self.nz = nz = nx + nu + N
        self.iu = iu = nx

        # cost 1/2 z'Pz + q'z
        wx = np.zeros(NX * (N + 1))
        wx[NX:] = np.tile([cfg["q_ey"], cfg["q_epsi"], cfg["q_v"], 0.0], N)
        D = sp.eye(N) - sp.eye(N, k=-1)
        self.r = np.array([cfg["r_dkappa"], cfg["r_da"]])
        P = sp.block_diag([sp.diags(2 * wx), sp.kron(2 * D.T @ D, sp.diags(self.r)),
                           sp.diags(np.full(N, 2 * SLACK_W))])
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
                add(r + i, xk + i, -(1.0 - c) if i == 3 else -1.0)
            vidx += [add(r, xk + 1, 0.0), add(r + 1, uk, 0.0)]
            add(r + 2, xk + 3, -dt)
            add(r + 3, uk + 1, -c)
        for j in range(nu):  # input bounds
            add(nx + j, iu + j, 1.0)
        ro = nx + nu
        for k in range(N):  # soft |e_y_{k+1}| <= ey_max, sig >= 0
            add(ro + 2 * k, NX * (k + 1), 1.0), add(ro + 2 * k, nx + nu + k, -1.0)
            add(ro + 2 * k + 1, NX * (k + 1), 1.0), add(ro + 2 * k + 1, nx + nu + k, 1.0)
            add(ro + 2 * N + k, nx + nu + k, 1.0)
        m = ro + 3 * N
        self.vals, self.vidx = np.array(vals), np.array(vidx).reshape(N, 2)
        A = sp.csc_matrix((np.arange(1.0, len(vals) + 1), (rows, cols)), shape=(m, nz))
        A.sort_indices()
        self.perm = A.data.astype(int) - 1  # CSC slot -> COO entry
        self.vals[self.vidx] = -dt * 30.0  # real values at a nominal speed: OSQP's scaling is fixed at setup
        A.data = self.vals[self.perm]
        self.l, self.u = np.zeros(m), np.zeros(m)
        ey_max = cfg["ey_max"]
        self.l[ro:ro + 2 * N:2], self.u[ro:ro + 2 * N:2] = -np.inf, ey_max
        self.l[ro + 1:ro + 2 * N:2], self.u[ro + 1:ro + 2 * N:2] = -ey_max, np.inf
        self.u[ro + 2 * N:] = np.inf

        self.solver = osqp.OSQP()
        # Deterministic: fixed max_iter and no time_limit (the sim is synchronous, wall time doesn't score).
        self.solver.setup(sp.triu(P, format="csc"), np.zeros(nz), A, self.l, self.u, verbose=False,
                          warm_starting=True, polishing=False, eps_abs=1e-4, max_iter=4000,
                          adaptive_rho_interval=25)
        self._a_act = None
        self._v_pred = None

    def solve(self, e_y, e_psi, v, s_total, kappa_ref_fn, v_ref_fn, u_prev):
        N, dt, veh = self.N, self.dt, self.veh
        k_prev, a_prev = float(u_prev[0]), float(u_prev[1])
        a_act = a_prev if self._a_act is None else self._a_act
        x0 = np.array([e_y + dt * v * e_psi, e_psi + dt * v * (k_prev - kappa_ref_fn(s_total)),
                       v + dt * a_act, a_act + self.c * (a_prev - a_act)])
        self._a_act = x0[3]
        s0 = s_total + dt * v

        vl = np.empty(N + 1)
        vl[0] = max(x0[2], 0.0)
        if self._v_pred is not None:
            vl[1:] = np.maximum(np.r_[self._v_pred[2:], self._v_pred[-1]], 0.0)
            s = s0 + dt * np.r_[0.0, np.cumsum(vl[:-1])]
        else:
            s = np.empty(N + 1)
            s[0] = s0
            for k in range(N):
                s[k + 1] = s[k] + dt * vl[k]
                vl[k + 1] = max(v_ref_fn(s[k + 1]), 0.0)
        kr = np.array([kappa_ref_fn(x) for x in s[:N]])
        vr = np.array([v_ref_fn(x) for x in s[1:]])
        km = np.array([veh.kappa_max(x) for x in vl[:N]])
        ad = np.maximum([veh.a_drive(x) for x in vl[:N]], self.a_min)

        self.vals[self.vidx] = -dt * vl[:N, None]
        q = np.zeros(self.nz)
        q[NX + 2:self.nx:NX] = -2 * self.q_v * vr
        q[self.iu:self.iu + NU] = -2 * self.r * (k_prev, a_prev)
        l, u, nx = self.l, self.u, self.nx
        l[:NX] = u[:NX] = x0
        l[NX + 1:nx:NX] = u[NX + 1:nx:NX] = -dt * vl[:N] * kr
        l[nx:nx + 2 * N:2], u[nx:nx + 2 * N:2] = -km, km
        l[nx + 1:nx + 2 * N:2], u[nx + 1:nx + 2 * N:2] = self.a_min, ad
        self.solver.update(Ax=self.vals[self.perm], q=q, l=l, u=u)
        res = self.solver.solve()
        z = res.x
        if res.info.status_val != osqp.SolverStatus.OSQP_SOLVED or not np.all(np.isfinite(z)):
            return self._fallback(x0, s0, kappa_ref_fn, v_ref_fn)
        self._v_pred = z[2:nx:NX].copy()
        kappa = float(np.clip(z[self.iu], -km[0], km[0]))
        a = float(np.clip(z[self.iu + 1], self.a_min, ad[0]))
        return kappa, a, True

    def _fallback(self, x0, s0, kappa_ref_fn, v_ref_fn):
        """Pure pursuit on a lookahead point of the line (circle-arc approximation) + P speed control."""
        self._v_pred = None
        e_y, e_psi, v = x0[0], x0[1], max(x0[2], 0.0)
        Ld = max(10.0, 0.6 * v)
        alpha = math.atan2(0.5 * kappa_ref_fn(s0) * Ld * Ld - e_y, Ld) - e_psi
        km = self.veh.kappa_max(v)
        kappa = float(np.clip(2 * math.sin(alpha) / Ld, -km, km))
        a = float(np.clip(K_V_FALLBACK * (v_ref_fn(s0) - v), self.a_min, self.veh.a_drive(v)))
        return kappa, a, False
