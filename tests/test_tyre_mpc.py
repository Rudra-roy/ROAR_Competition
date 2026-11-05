import json
import os
import time

import numpy as np

from tyre_mpc import NX, Tyres, TyreMPC
from vehicle import Vehicle

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "competition_code", "data")
TYRE = dict(Cf=70.0, Cr=90.0, mu_f=15.0, mu_r=17.0, a=1.3, b=1.575, I=2.4, tau_d=0.12,
            g_v=[10.0, 25.0, 40.0, 55.0], g_k=[0.55, 0.45, 0.36, 0.3], dt=0.05)


def make(**cfg_over):
    with open(os.path.join(DATA, "vehicle_params.json")) as f:
        p = json.load(f)
    p["tyre"] = TYRE
    veh = Vehicle(p)
    cfg = dict(p["mpc"], model="tyre", r_da=0.5, q_v=2.0, **cfg_over)
    return veh, TyreMPC(veh, cfg)


def test_linearisation_matches_nonlinear_step_at_nominal():
    veh, m = make()
    ty, dt = m.ty, m.dt
    rng = np.random.default_rng(3)
    vl = rng.uniform(15, 60, m.N + 1)
    nom = np.c_[rng.normal(0, 0.03, m.N), rng.normal(0, 0.03, m.N), rng.normal(0, 0.2, m.N)]
    X = np.zeros((m.N + 1, NX))
    X[:, 2] = vl
    X[:m.N, 4:7] = nom
    X[m.N, 4:7] = nom[-1]
    c_beta, c_r, ay_c = m._write_dynamic(X, np.zeros(m.N))
    for k in (0, 7, m.N - 1):
        x = np.array([0.0, 0.0, vl[k], 0.0, nom[k, 0], nom[k, 1], nom[k, 2]])
        x1 = ty.step(x, (nom[k, 0], 0.0), 0.0, dt, m.c, m.cd)  # kappa_ref 0: the psidx entries are 0 here
        beta_lin = -(m.vals[m.latidx[k, 0, :]] @ nom[k]) + c_beta[k]
        r_lin = -(m.vals[m.latidx[k, 1, :]] @ nom[k]) + c_r[k]
        assert abs(beta_lin - x1[5]) < 1e-9
        assert abs(r_lin - x1[6]) < 1e-9
        af, ar = nom[k, 0] - nom[k, 1] - ty.a * nom[k, 2] / vl[k], -nom[k, 1] + ty.b * nom[k, 2] / vl[k]
        Ff, Fr, _, _ = ty.forces(af, ar)
        v1 = max(vl[k + 1], 10.0)
        n1 = nom[k + 1] if k + 1 < m.N else nom[-1]
        af1, ar1 = n1[0] - n1[1] - ty.a * n1[2] / v1, -n1[1] + ty.b * n1[2] / v1
        Ff1, Fr1, _, _ = ty.forces(af1, ar1)
        assert abs(m.vals[m.ayidx[k, 0, :]] @ n1 + ay_c[k] - (Ff1 + Fr1)) < 1e-9


def circle_solves(m, veh, R, v, e_y0=0.0, ticks=1, a_lat=None, v_ref=None):
    kappa = 1.0 / R
    d_ss, b_ss, r_ss = m.ty.steady(kappa, v)
    out = []
    u = (d_ss, 0.0)
    e_y, e_psi, beta, r, vv = e_y0, -b_ss, b_ss, r_ss, v  # steady state: course (yaw + beta) along the line
    a_act, d_act = 0.0, d_ss
    for t in range(ticks):
        d, a, ok = m.solve(e_y, e_psi, vv, beta, r, 100.0 + 0.05 * v * t, lambda s: kappa,
                           lambda s: v if v_ref is None else v_ref, u, lambda s: a_lat or veh.a_lat_max)
        out.append((d, a, ok))
        x = m.ty.step(np.array([e_y, e_psi, vv, a_act, d_act, beta, r]), (d, a), kappa, m.dt, m.c, m.cd)
        e_y, e_psi, vv, a_act, d_act, beta, r = x
        u = (d, a)
    return out, (d_ss, b_ss, r_ss)


def test_steady_circle_holds_steady_steer():
    veh, m = make()
    out, (d_ss, _, _) = circle_solves(m, veh, R=200.0, v=40.0)
    d, a, ok = out[0]
    assert ok
    assert abs(d - d_ss) < 0.15 * abs(d_ss) + 0.002
    assert abs(a) < 0.5


def test_offset_left_steers_right():
    veh, m = make()
    out, (d_ss, _, _) = circle_solves(m, veh, R=200.0, v=40.0, e_y0=0.5)
    assert out[0][2] and out[0][0] < d_ss - 0.002


def test_slip_limit_binds_and_car_brakes_over_speed():
    veh, m = make(alpha_max_deg=6.0)
    out, _ = circle_solves(m, veh, R=70.0, v=48.0, ticks=12, a_lat=25.0)  # needs 33 m/s^2: over the tyres and the grip
    assert all(o[2] for o in out)
    assert out[-1][1] < -8.0  # brakes hard
    z = m._z
    X = z[:m.nx].reshape(m.N + 1, NX)
    v1 = np.maximum(X[1:, 2], 10.0)
    af = X[1:, 4] - X[1:, 5] - m.ty.a * X[1:, 6] / v1
    assert np.abs(af[40:]).max() < np.radians(6.0) + 0.03  # once the car has slowed, the plan honours the slip limit
    assert X[-1, 2] < 44.0  # the plan ends slower than it started


def test_deterministic_and_fast():
    veh, m1 = make()
    _, m2 = make()
    t = []
    outs = []
    for mm in (m1, m2):
        res = []
        for e in (0.0, 0.3, -0.2, 0.1):
            t0 = time.perf_counter()
            res.append(circle_solves(mm, veh, R=150.0, v=45.0, e_y0=e, ticks=2)[0])
            t.append(time.perf_counter() - t0)
        outs.append(res)
    assert outs[0] == outs[1]
    assert np.median(t) / 2 < 0.03


def test_low_speed_steer_hold_and_cap():
    veh, m = make()
    kappa = 0.0
    for v, lim in ((0.1, 0.0), (4.0, 0.2), (30.0, m.ty.g(30.0))):
        m.reset()
        d, a, ok = m.solve(2.0, 0.03, v, 0.0, 0.0, 10.0, lambda s: kappa, lambda s: 20.0, (0.0, 0.0),
                           lambda s: veh.a_lat_max)
        assert ok
        assert abs(d) <= lim + 1e-9, (v, d)
    m.reset()
    d, _, _ = m.solve(2.0, 0.03, 4.0, 0.0, 0.0, 10.0, lambda s: kappa, lambda s: 20.0, (0.0, 0.0), lambda s: veh.a_lat_max)
    assert d < -0.03  # 2 m left of the line: steers right (first solve: one trust-region step)


def test_brake_ellipse_limits_braking_while_cornering():
    veh, m = make(brake_ellipse=1)
    _, m0 = make(brake_ellipse=0)
    kappa = 1.0 / 60.0
    v = 40.0  # a_y = 26.7: near the grip limit
    d_ss, b_ss, r_ss = m.ty.steady(kappa, v)
    args = (0.0, -b_ss, v, b_ss, r_ss, 100.0, lambda s: kappa, lambda s: 10.0, (d_ss, -14.0), lambda s: 28.0)
    a_el = m.solve(*args)[1]
    a_free = m0.solve(*args)[1]
    assert a_free < -12.0  # without coupling it brakes hard mid-corner
    assert a_el > a_free + 3.0  # with it, much less
    straight = (0.0, 0.0, v, 0.0, 0.0, 100.0, lambda s: 0.0, lambda s: 10.0, (0.0, -14.0), lambda s: 28.0)
    assert abs(m.solve(*straight)[1] - m0.solve(*straight)[1]) < 1.0  # straight line: unchanged
