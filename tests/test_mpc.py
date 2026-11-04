import os
import time

import numpy as np

from mpc import MAX_ITER, FrenetMPC
from profile_oracle import oracle_profile
from vehicle import Vehicle

V = Vehicle.from_json(os.path.join(os.path.dirname(__file__), "..", "competition_code", "data", "vehicle_params.json"))
CFG = V.p["mpc"]
DT = CFG["dt"]
TAU = CFG.get("a_tau_s", V.p["brake_tau_s"])  # the shipped config's longitudinal lag
OPTS = ("a_tau_s", "jerk_max", "ellipse", "a_lat_margin", "beta_k", "beta_tau_s")
BASE = {k: x for k, x in CFG.items() if k not in OPTS}  # options off
NO_BETA = {k: x for k, x in CFG.items() if k not in ("beta_k", "beta_tau_s")}
M = CFG.get("a_lat_margin", 1.0)


def kappa_max(v):
    """The MPC's curvature bound: steering, and the lateral-accel limit with the shipped headroom."""
    v = max(v, 1.0)
    return min(V.K(v), M * V.a_lat_max / v ** 2)
TAU_K = V.p["steer_tau_s"]
A_MIN = -V.a_brake * V.brake_scale


KB, TB = CFG.get("beta_k", 0.0), CFG.get("beta_tau_s", 1.0)  # shipped sideslip state (0 = off)


def plant(x, u, kappa_ref):
    """Exact model the MPC predicts with: x = [e_y, e_psi, v, a_act, k_act, b], u = [kappa, a_cmd]."""
    e_y, e_psi, v, a, k, b = x
    return np.array([e_y + DT * v * (e_psi - b), e_psi + DT * v * (k - kappa_ref), v + DT * a,
                     a + DT / TAU * (u[1] - a), k + DT / TAU_K * (u[0] - k),
                     b + DT / TB * (KB * v * v * (k - kappa_ref) - b)])


def closed_loop(m, x, s, kappa_fn, v_ref_fn, ticks, times=None, u_prev=(0.0, 0.0)):
    """1-tick delay: the command computed at tick k is applied during tick k+1."""
    log, x = [], np.r_[x, 0.0]  # + sideslip b
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
        u_prev = (rng.uniform(-1, 1) * kappa_max(v), 0.0)  # a previous command is itself within the bound
        k, a, ok = m.solve(e_y, e_psi, v, rng.uniform(0, 5000), kref, lambda s, vr=rng.uniform(0, 70): vr, u_prev)
        assert np.isfinite([k, a]).all()
        assert abs(k) <= kappa_max(v) + 1e-3
        assert A_MIN - 1e-3 <= a <= V.a_drive(v) + 1e-3
        if abs(e_y + DT * v * e_psi) <= CFG["ey_max"]:  # delay-propagated state inside the soft bound
            assert ok, (e_y, e_psi, v)
        fails += not ok
        if ok:  # the QP itself, before the clip
            kq, aq = raw[-1][m.iu:m.iu + 2]
            assert abs(kq) <= kappa_max(v) + 1e-3
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
    m.solver.update_settings(max_iter=MAX_ITER)
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


# ---- Task 8c options (mpc cfg keys a_tau_s, jerk_max, ellipse) ----

def braking_corner_solves(cfg, ticks=120, x0=(0.3, 0.0, 60.0, 0.0, 0.0), times=None):
    """Closed loop into the braking zone + R=40 corner; returns (MPC, raw QP solutions, commands)."""
    m = FrenetMPC(V, cfg)
    raw = capture(m)
    vr = corner_ref()
    x, s, u_prev, cmds = np.array(x0), S_C - 150, (0.0, 0.0), []
    tau = cfg.get("a_tau_s", V.p["brake_tau_s"])
    for _ in range(ticks):
        t0 = time.perf_counter()
        k, a, ok = m.solve(x[0], x[1], x[2], s, CORNER, vr, u_prev)
        if times is not None:
            times.append(time.perf_counter() - t0)
        assert ok and np.isfinite([k, a]).all()
        e_y, e_psi, v, aa, kk = x
        x = np.array([e_y + DT * v * e_psi, e_psi + DT * v * (kk - CORNER(s)), v + DT * aa,
                      aa + DT / tau * (u_prev[1] - aa), kk + DT / TAU_K * (u_prev[0] - kk)])
        s += DT * v
        u_prev = (k, a)
        cmds.append((k, a))
    return m, raw, np.array(cmds)


def horizon(m, z):
    xs = z[:m.nx].reshape(-1, 5)
    us = z[m.iu:m.iu + 2 * m.N].reshape(-1, 2)
    return xs, us


def test_a_tau_option_sets_longitudinal_lag():
    m, raw, _ = braking_corner_solves(dict(BASE, a_tau_s=0.225), ticks=20)
    xs, us = horizon(m, raw[-1])
    c = DT / 0.225
    np.testing.assert_allclose(xs[1:, 3], xs[:-1, 3] + c * (us[:, 1] - xs[:-1, 3]), atol=1e-3)


def test_jerk_limit_over_horizon_and_from_u_prev():
    J = 60.0
    m, raw, cmds = braking_corner_solves(dict(BASE, jerk_max=J))
    tol = 0.05 * J * DT
    for z in raw:
        _, us = horizon(m, z)
        assert np.abs(np.diff(us[:, 1])).max() <= J * DT + tol
    da = np.abs(np.diff(np.r_[0.0, cmds[:, 1]]))  # issued commands, from u_prev = 0
    assert da.max() <= J * DT + 1e-6
    assert cmds[:, 1].min() < -10  # still brakes hard for the corner
    _, _, free = braking_corner_solves(BASE)
    assert np.abs(np.diff(free[:, 1])).max() > J * DT  # the option is what limits it


def ellipse_use(cfg_m, z):
    xs, us = horizon(cfg_m, z)
    A = V.a_brake * V.brake_scale
    lat = xs[:-1, 2] ** 2 * us[:, 0] / (V.p["brake_ellipse_c"] * V.a_lat_max)
    return (np.minimum(us[:, 1], 0) / A) ** 2 + lat ** 2, us[:, 1] < -0.5


def test_ellipse_couples_braking_and_lateral():
    """Into the corner over speed (70 m/s): the free MPC plans full braking at high lateral accel."""
    x0 = (0.3, 0.0, 70.0, 0.0, 0.0)
    m, raw, _ = braking_corner_solves(dict(BASE, ellipse=True), ticks=150, x0=x0)
    worst = 0.0
    for z in raw:
        use, brk = ellipse_use(m, z)
        if brk.any():
            worst = max(worst, use[brk].max())
    assert worst <= 1.08  # tangent polygon at 30/60 deg + linearised v^2: outer bound ~7 %
    m0, raw0, _ = braking_corner_solves(BASE, ticks=150, x0=x0)
    worst0 = max(ellipse_use(m0, z)[0][ellipse_use(m0, z)[1]].max(initial=0) for z in raw0)
    assert worst0 > 1.2  # without the rows the plan exceeds the ellipse


def test_options_deterministic_and_fast():
    cfg = dict(BASE, a_tau_s=0.225, jerk_max=60.0, ellipse=True, a_lat_margin=1.1)
    times = []
    _, _, c1 = braking_corner_solves(cfg, times=times)
    _, _, c2 = braking_corner_solves(cfg)
    assert np.array_equal(c1, c2)
    print("median solve %.2f ms, max %.2f ms" % (1e3 * np.median(times), 1e3 * max(times)))
    assert np.median(times) < 0.02


def test_a_lat_margin_scales_lateral_limit():
    """On a circle at the grip limit, 0.5 m outside: the margin lets the MPC turn tighter instead of braking."""
    v = 40.0
    kc = V.a_lat_max / v ** 2
    out = {}
    for m in (1.0, 1.1):
        mpc = FrenetMPC(V, dict(BASE, a_lat_margin=m))
        raw = capture(mpc)
        k, a, ok = mpc.solve(-0.5, 0.0, v, 0.0, lambda s: kc, lambda s: v, (kc, 0.0))
        assert ok
        _, us = horizon(mpc, raw[-1])
        xs, _ = horizon(mpc, raw[-1])
        assert (np.abs(us[:, 0]) <= m * V.a_lat_max / np.maximum(xs[:-1, 2], 10.0) ** 2 * 1.01 + 1e-4).all()
        assert abs(k) <= m * V.a_lat_max / (v + DT * 0.0) ** 2 + 1e-6
        out[m] = (k, a)
    assert out[1.1][0] > 1.03 * kc  # uses the headroom
    assert out[1.1][1] > out[1.0][1]  # and brakes less


def test_fallback_uses_unscaled_a_lat():
    """a_lat_margin is MPC headroom only: the pure-pursuit fallback clips curvature at the profile's a_lat."""
    m = FrenetMPC(V, dict(CFG, a_lat_margin=1.2))
    m.solver.update_settings(max_iter=3)
    v = 50.0
    k, _, ok = m.solve(5.0, 0.0, v, 0.0, lambda s: 0.0, lambda s: v, (0.0, 0.0), a_lat_fn=lambda s: 20.0)
    assert ok is False
    assert 20.0 / v ** 2 < V.K(v)  # the a_lat bound binds, not steering
    assert abs(abs(k) - 20.0 / v ** 2) < 1e-9



# ---- Task 8e: sideslip state (cfg beta_k, beta_tau_s) ----
# Race logs (8d gate, straights): body sideslip beta = yaw - course follows the lateral accel with a lag,
# beta ~ 0.0071 * lag(v^2 * yaw curvature, 0.40 s) (R2 0.99). The MPC's e_y' = v*e_psi ignores it.
KB_PLANT, TB_PLANT = 0.0071, 0.40


def slip_loop(cfg, v=65.0, ticks=200):
    """Straight at v, 0.5 m off the line; plant = the MPC's yaw model + the sideslip lag (course = yaw - beta)."""
    m = FrenetMPC(V, cfg)
    ey, epsi, k, b, u, log = 0.5, 0.0, 0.0, 0.0, (0.0, 0.0), []
    for _ in range(ticks):
        kc, a, ok = m.solve(ey, epsi, v, 0.0, lambda s: 0.0, lambda s: v, u)
        assert ok
        ey, epsi, k, b = (ey + DT * v * (epsi - b), epsi + DT * v * k, k + DT / TAU_K * (u[0] - k),
                          b + DT / TB_PLANT * (KB_PLANT * v * v * k - b))
        u = (kc, a)
        log.append(ey)
    return np.array(log)


def test_sideslip_lag_makes_a_weave_without_the_state():
    ey = slip_loop(NO_BETA)
    assert -ey[:80].min() > 0.25  # 8d model: > 50 % overshoot, ~0.4 Hz, slow decay (the race weave)
    assert np.abs(ey[80:120]).max() > 0.1


def test_sideslip_state_damps_it():
    ey = slip_loop(dict(CFG, beta_k=KB_PLANT, beta_tau_s=TB_PLANT))
    assert -ey.min() < 0.1, -ey.min()
    assert np.abs(ey[80:120]).max() < 0.03


def test_sideslip_state_model_rows():
    """Raw horizon: b+ = b + dt/tau_b (k_b v^2 (k_act - kappa_ref) - b) and e_y+ = e_y + dt v (e_psi - b)."""
    kb, tb = 0.01, 0.6
    m = FrenetMPC(V, dict(CFG, beta_k=kb, beta_tau_s=tb))
    raw = capture(m)
    kref = 1 / 300.0
    k, a, ok = m.solve(0.3, 0.01, 50.0, 0.0, lambda s: kref, lambda s: 50.0, (0.0, 0.0))
    assert ok and m.NX == 6
    z = raw[-1]
    xs = z[:m.nx].reshape(-1, 6)
    vk = 50.0  # the linearisation speed: first call, v = v_ref, a = 0 -> v_k = 50 over the horizon
    np.testing.assert_allclose(xs[1:, 5], xs[:-1, 5] + DT / tb * (kb * vk ** 2 * (xs[:-1, 4] - kref) - xs[:-1, 5]),
                               atol=2e-4)
    np.testing.assert_allclose(xs[1:, 0], xs[:-1, 0] + DT * vk * (xs[:-1, 1] - xs[:-1, 5]), atol=2e-3)
    assert FrenetMPC(V, NO_BETA).NX == 5  # absent = off (the 8d model)


def test_sideslip_option_needs_tau_and_clamps_rate():
    import pytest
    with pytest.raises(ValueError):
        FrenetMPC(V, dict(NO_BETA, beta_k=0.007))
    m = FrenetMPC(V, dict(NO_BETA, beta_k=0.007, beta_tau_s=0.01))  # tau < dt: the lag row must not overshoot
    assert m.cb == 1.0
    assert m.solve(0.3, 0.0, 50.0, 0.0, lambda s: 0.0, lambda s: 50.0, (0.0, 0.0))[2]


def test_pedal_brake_model_bounds_a_by_speed():
    """brake_model pedal_map: the a_cmd lower bound is -a_brake_max(v_k), so far above v_ref the MPC brakes harder
    than the constant limit at high speed and never asks for more than the car has at low speed."""
    vb = Vehicle(dict(V.p, brake_model="pedal_map"))
    for v in (25.0, 45.0, 65.0):
        m = FrenetMPC(vb, CFG)
        raw = capture(m)
        a = None
        for _ in range(30):  # let the jerk limit ramp the command down
            k, a, ok = m.solve(0.0, 0.0, v, 0.0, lambda s: 0.0, lambda s: 5.0, (0.0, a or 0.0))
            assert ok
        lo = -vb.a_brake_max(v + DT * a)  # the bound is taken at the delay-propagated speed (v + dt*a_act)
        assert lo - 1e-2 <= a <= lo + 0.3, (v, a, lo)
        assert raw[-1][m.iu + 1] >= lo - 0.3
    assert vb.a_brake_max(65.0) > -A_MIN > vb.a_brake_max(25.0)
