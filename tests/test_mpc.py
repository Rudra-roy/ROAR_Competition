import os
import time

import numpy as np

from mpc import FrenetMPC
from profile_oracle import oracle_profile
from vehicle import Vehicle

V = Vehicle.from_json(os.path.join(os.path.dirname(__file__), "..", "competition_code", "data", "vehicle_params.json"))
CFG = V.p["mpc"]
DT = CFG["dt"]
TAU = V.p["brake_tau_s"]
TAU_K = V.p["steer_tau_s"]
A_MIN = -V.a_brake * V.brake_scale


def plant(x, u, kappa_ref):
    """Exact model the MPC predicts with: x = [e_y, e_psi, v, a_act, k_act], u = [kappa, a_cmd]."""
    e_y, e_psi, v, a, k = x
    return np.array([e_y + DT * v * e_psi, e_psi + DT * v * (k - kappa_ref), v + DT * a,
                     a + DT / TAU * (u[1] - a), k + DT / TAU_K * (u[0] - k)])


def closed_loop(m, x, s, kappa_fn, v_ref_fn, ticks, times=None, u_prev=(0.0, 0.0)):
    """1-tick delay: the command computed at tick k is applied during tick k+1."""
    log = []
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
    log = closed_loop(FrenetMPC(V, CFG), np.array([1.0, 0, 40.0, 0, 0]), 0.0,
                      lambda s: 0.0, lambda s: 40.0, ticks=120)
    ey = log[:, 0]
    assert np.abs(ey[int(2.0 / DT) - 1:]).max() < 0.1  # within 2 s, and stays there
    assert ey.min() > -0.2  # overshoot


def capture(m):
    """Record the raw OSQP solution of every solve, unscaled to physical units (before FrenetMPC clips it)."""
    raw, orig = [], m.solver.solve

    def wrapped(*a, **kw):
        res = orig(*a, **kw)
        raw.append(res.x * m.sz)
        return res
    m.solver.solve = wrapped
    return raw


def test_respects_input_bounds():
    rng = np.random.default_rng(0)
    fails = 0
    for _ in range(200):
        e_y, e_psi, v = rng.uniform(-2, 2), rng.uniform(-0.3, 0.3), rng.uniform(0, 70)
        m = FrenetMPC(V, CFG)  # fresh: a_act starts at u_prev[1] = 0, so the delayed-step speed is v
        raw = capture(m)
        kref = lambda s, c=rng.uniform(-0.05, 0.05): c * np.sin(s / 30.0)
        u_prev = (rng.uniform(-1, 1) * V.kappa_max(v), 0.0)  # a previous command is itself within the bound
        k, a, ok = m.solve(e_y, e_psi, v, rng.uniform(0, 5000), kref, lambda s, vr=rng.uniform(0, 70): vr, u_prev)
        assert np.isfinite([k, a]).all()
        assert abs(k) <= V.kappa_max(v) + 1e-3
        assert A_MIN - 1e-3 <= a <= V.a_drive(v) + 1e-3
        if abs(e_y + DT * v * e_psi) <= CFG["ey_max"]:  # delay-propagated state inside the soft bound
            assert ok, (e_y, e_psi, v)
        fails += not ok
        if ok:  # the QP itself, before the clip
            kq, aq = raw[-1][m.iu:m.iu + 2]
            assert abs(kq) <= V.kappa_max(v) + 1e-3
            # a: OSQP's inf-norm residual tolerance scales with the largest scaled variable, which is the e_y
            # slack (~100 m) in these beyond-grip states; measured worst raw overshoot 0.62 m/s^2 (0.46 before the
            # k_act state). The command itself is clipped exactly (asserted above).
            assert A_MIN - 0.7 <= aq <= V.a_drive(v) + 0.7
    assert fails <= 4  # 2 % of 200


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


def test_fallback_on_solver_failure():
    m = FrenetMPC(V, CFG)
    args = (0.0, 30.0, 0.0, lambda s: 0.0, lambda s: 30.0, (0.0, 0.0))
    assert m.solve(0.5, *args)[2]
    m.solver.update_settings(max_iter=3)  # a real OSQP failure: max_iter reached
    for e_y in (0.6, 0.7, 0.8):
        k, a, ok = m.solve(e_y, *args)
        assert ok is False and np.isfinite([k, a]).all()
        assert -V.kappa_max(30.0) - 1e-9 <= k < 0  # car left of the line -> steer right, back toward it
    m.solver.update_settings(max_iter=4000)
    assert m.solve(0.8, *args)[2]


def test_nan_input_does_not_poison():
    m = FrenetMPC(V, CFG)
    args = (0.0, 30.0, 0.0, lambda s: 0.0)
    assert m.solve(0.5, *args, lambda s: 30.0, (0.0, 0.0))[2]
    k, a, ok = m.solve(0.5, *args, lambda s: float("nan"), (0.0, 0.0))
    assert ok is False and np.isfinite([k, a]).all()
    for _ in range(5):
        k, a, ok = m.solve(0.5, *args, lambda s: 30.0, (k, a))
        assert ok and np.isfinite([k, a]).all()


def test_reset_clears_state():
    m = FrenetMPC(V, CFG)
    m.solve(0.5, 0.0, 30.0, 0.0, lambda s: 0.0, lambda s: 30.0, (0.0, 2.0))
    m.reset()
    assert m._v_pred is None and m._a_act is None
    assert m.solve(0.0, 0.0, 0.0, 0.0, lambda s: 0.0, lambda s: 0.0, (0.0, 0.0))[2]


def test_slows_for_tight_corner_at_limit():
    v_c = np.sqrt(V.a_lat_max * 40)  # v_ref exactly at the grip limit (speed_scale 1.0)
    log = closed_loop(FrenetMPC(V, CFG), np.array([0.0, 0, v_c, 0, 1 / 40.0]), 0.0, lambda s: 1 / 40.0,
                      lambda s: v_c, ticks=240, u_prev=(1 / 40.0, 0.0))  # already on the circle
    assert np.abs(log[:, 0]).max() <= 0.3
    vr = corner_ref()  # straight -> braking -> the same corner, oracle at scale 1.0
    log = closed_loop(FrenetMPC(V, CFG), np.array([0.0, 0, 60.0, 0, 0]), S_C - 150, CORNER, vr, ticks=300)
    assert np.abs(log[:, 0]).max() <= 0.3


def corner_loop(ticks, times=None):
    vr, sc = corner_ref(), V.p["speed_scale"]  # shipped grip margin; at exactly the limit the car drifts wide
    return closed_loop(FrenetMPC(V, CFG), np.array([0.3, 0, 60.0, 0, 0]), S_C - 150, CORNER,
                       lambda s: sc * vr(s), ticks, times)


def test_solve_time():
    times = []
    log = corner_loop(200, times)
    assert np.median(times) < 0.015
    assert np.abs(log[:, 0]).max() < CFG["ey_max"]  # tracks the braking zone and the R=40 corner


def test_deterministic():
    assert np.array_equal(corner_loop(60), corner_loop(60))


def test_no_weave_with_steering_lag():
    """40 m/s, 0.5 m off the line, plant with the steering -> curvature lag: e_y decays without a growing weave."""
    log = closed_loop(FrenetMPC(V, CFG), np.array([0.5, 0, 40.0, 0, 0]), 0.0, lambda s: 0.0, lambda s: 40.0,
                      ticks=200)
    ey = np.abs(log[:, 0])
    peaks = [ey[i:i + 40].max() for i in range(0, 200, 40)]  # 2 s windows
    assert all(b <= a + 1e-3 for a, b in zip(peaks, peaks[1:])), peaks
    assert ey[-40:].max() < 0.05
