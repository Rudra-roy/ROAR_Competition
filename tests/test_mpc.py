import os
import time
from types import SimpleNamespace

import numpy as np

from mpc import FrenetMPC
from profile_oracle import oracle_profile
from vehicle import Vehicle

V = Vehicle.from_json(os.path.join(os.path.dirname(__file__), "..", "competition_code", "data", "vehicle_params.json"))
CFG = V.p["mpc"]
DT = CFG["dt"]
TAU = V.p["brake_tau_s"]
A_MIN = -V.a_brake * V.brake_scale


def plant(x, u, kappa_ref):
    """Exact model the MPC predicts with: x = [e_y, e_psi, v, a_act], u = [kappa, a_cmd]."""
    e_y, e_psi, v, a = x
    return np.array([e_y + DT * v * e_psi, e_psi + DT * v * (u[0] - kappa_ref), v + DT * a,
                     a + DT / TAU * (u[1] - a)])


def closed_loop(m, x, s, kappa_fn, v_ref_fn, ticks, times=None):
    """1-tick delay: the command computed at tick k is applied during tick k+1."""
    u_prev, log = (0.0, 0.0), []
    for _ in range(ticks):
        t0 = time.perf_counter()
        k, a, ok = m.solve(x[0], x[1], x[2], s, kappa_fn, v_ref_fn, u_prev)
        if times is not None:
            times.append(time.perf_counter() - t0)
        assert ok and np.isfinite([k, a]).all()
        x, s = plant(x, u_prev, kappa_fn(s)), s + DT * x[2]
        u_prev = (k, a)
        log.append(x.copy())
    return np.array(log)


def test_converges_on_straight():
    log = closed_loop(FrenetMPC(V, CFG), np.array([1.0, 0, 40.0, 0]), 0.0,
                      lambda s: 0.0, lambda s: 40.0, ticks=120)
    ey = log[:, 0]
    assert np.abs(ey[int(2.0 / DT) - 1:]).max() < 0.1  # within 2 s, and stays there
    assert ey.min() > -0.2  # overshoot


def test_respects_input_bounds():
    rng = np.random.default_rng(0)
    for _ in range(200):
        v = rng.uniform(0, 70)
        m = FrenetMPC(V, CFG)  # fresh: a_act starts at u_prev[1] = 0, so the delayed-step speed is v
        kref = lambda s, c=rng.uniform(-0.05, 0.05): c * np.sin(s / 30.0)
        k, a, ok = m.solve(rng.uniform(-2, 2), rng.uniform(-0.3, 0.3), v, rng.uniform(0, 5000),
                           kref, lambda s, vr=rng.uniform(0, 70): vr, (rng.uniform(-0.05, 0.05), 0.0))
        assert np.isfinite([k, a]).all()
        assert abs(k) <= V.kappa_max(v) + 1e-3
        assert A_MIN - 1e-3 <= a <= V.a_drive(v) + 1e-3


S_C = 500.0  # corner entry
CORNER = lambda s: 0.0 if s < S_C else 1 / 40.0


def corner_ref():
    s, v = oracle_profile(CORNER, lambda s: V.a_lat_max, V.a_drive, V.a_brake * V.brake_scale,
                          V.p["v_max"], S_C + 400, 1.0, v0=60.0)
    return lambda x: float(np.interp(x, s, v))


def test_brakes_before_corner():
    v_ref = corner_ref()
    v_c = np.sqrt(V.a_lat_max * 40)
    d_stop = (60 ** 2 - v_c ** 2) / (2 * V.a_brake * V.brake_scale)
    for d in np.linspace(5, d_stop, 10):
        _, a, ok = FrenetMPC(V, CFG).solve(0, 0, 60.0, S_C - d, CORNER, v_ref, (0.0, 0.0))
        assert ok and a < 0, (d, a)
    _, a, ok = FrenetMPC(V, CFG).solve(0, 0, 60.0, S_C - 4 * d_stop, CORNER, v_ref, (0.0, 0.0))
    assert ok and a > 0  # far from the corner it still drives


def test_fallback_on_solver_failure(monkeypatch):
    m = FrenetMPC(V, CFG)
    m.solve(0.5, 0.0, 30.0, 0.0, lambda s: 0.0, lambda s: 30.0, (0.0, 0.0))
    import osqp
    bad = SimpleNamespace(x=np.full(m.nz, np.nan),
                          info=SimpleNamespace(status_val=osqp.SolverStatus.OSQP_MAX_ITER_REACHED))
    monkeypatch.setattr(m.solver, "solve", lambda *a, **kw: bad)
    k, a, ok = m.solve(0.5, 0.0, 30.0, 0.0, lambda s: 0.0, lambda s: 30.0, (0.0, 0.0))
    assert ok is False and np.isfinite([k, a]).all()
    assert k < 0  # car left of the line -> steer right, back toward it
    assert abs(k) <= V.kappa_max(30.0) + 1e-9


def corner_loop(ticks, times=None):
    vr, sc = corner_ref(), V.p["speed_scale"]  # shipped grip margin; at exactly the limit the car drifts wide
    return closed_loop(FrenetMPC(V, CFG), np.array([0.3, 0, 60.0, 0]), S_C - 150, CORNER,
                       lambda s: sc * vr(s), ticks, times)


def test_solve_time():
    times = []
    log = corner_loop(200, times)
    assert np.median(times) < 0.015
    assert np.abs(log[:, 0]).max() < CFG["ey_max"]  # tracks the braking zone and the R=40 corner


def test_deterministic():
    assert np.array_equal(corner_loop(60), corner_loop(60))
