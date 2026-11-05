import asyncio

import numpy as np

from submission import RoarCompetitionSolution


class Sensor:
    def __init__(self):
        self.val = np.zeros(3)

    def get_last_gym_observation(self):
        return self.val


class Car:
    async def apply_action(self, control):
        self.control = control


def test_respawn_resets_progress():
    loc, rpy, vel, car = Sensor(), Sensor(), Sensor(), Car()
    sol = RoarCompetitionSolution([], car, location_sensor=loc, velocity_sensor=vel, rpy_sensor=rpy)
    asyncio.run(sol.initialize())
    tr = sol.track
    i0 = int(np.argmin(np.abs(tr.s - sol.profile_start_s)))

    def put(i, v):
        i %= len(tr.s)
        loc.val = np.r_[tr.xy[i], 0.0]
        rpy.val = np.array([0.0, 0.0, tr.psi[i]])
        vel.val = v * np.r_[np.cos(tr.psi[i]), np.sin(tr.psi[i]), 0.0]
        asyncio.run(sol.step())
        return sol.last

    for k in range(101):  # 3 m per tick (60 m/s) for 300 m
        row = put(i0 + int(round(3 * k / tr.ds)), 60.0)
    assert abs(row["s_total"] - 300.0) < 2.0
    assert row["v_ref"] > 10.0
    for key in ("throttle", "brake", "steer"):
        assert np.isfinite(car.control[key])

    row = put(i0, 0.0)  # collision respawn: teleported back to the start
    assert abs(row["s_total"]) < 1.0
    assert abs(row["v_ref"] - sol.v_ref(0.0)) < 0.5
    assert np.isfinite(car.control["steer"])


def test_sideslip_feedforward_bounded_by_grip():
    from mpc import FrenetMPC
    loc, rpy, vel, car = Sensor(), Sensor(), Sensor(), Car()
    sol = RoarCompetitionSolution([], car, location_sensor=loc, velocity_sensor=vel, rpy_sensor=rpy)
    asyncio.run(sol.initialize())
    sol.tyre = False  # the feed-forward belongs to the kinematic controller (the shipped tyre MPC models beta)
    sol.mpc = FrenetMPC(sol.veh, dict(sol.veh.p["mpc"], **sol.veh.p.get("mpc_kinematic", {})))
    sol._off = None
    tr = sol.track
    i = int(np.argmax(np.abs(tr.kappa)))  # tightest corner, far over speed: v^2*kappa >> a_lat
    loc.val = np.r_[tr.xy[i], 0.0]
    rpy.val = np.array([0.0, 0.0, tr.psi[i]])
    vel.val = 80.0 * np.r_[np.cos(tr.psi[i]), np.sin(tr.psi[i]), 0.0]
    asyncio.run(sol.step())
    row = sol.last
    assert 80.0 ** 2 * abs(row["kappa_ref"]) > 2 * sol.veh.a_lat(row["line_s"])
    assert abs(row["e_psi"]) <= sol.veh.p["sideslip_k"] * sol.veh.a_lat(row["line_s"]) + 0.02
    assert np.isfinite(car.control["steer"])



def test_log_row_has_body_yaw():
    loc, rpy, vel, car = Sensor(), Sensor(), Sensor(), Car()
    sol = RoarCompetitionSolution([], car, location_sensor=loc, velocity_sensor=vel, rpy_sensor=rpy)
    asyncio.run(sol.initialize())
    i = 100
    loc.val = np.r_[sol.track.xy[i], 0.0]
    rpy.val = np.array([0.0, 0.0, sol.track.psi[i] + 0.1])
    vel.val = 20.0 * np.r_[np.cos(sol.track.psi[i]), np.sin(sol.track.psi[i]), 0.0]
    asyncio.run(sol.step())
    assert sol.last["yaw"] == rpy.val[2]  # the body yaw, for the sideslip fit (tools/fit_sideslip.py)


def _sol():
    loc, rpy, vel, car = Sensor(), Sensor(), Sensor(), Car()
    sol = RoarCompetitionSolution([], car, location_sensor=loc, velocity_sensor=vel, rpy_sensor=rpy)
    asyncio.run(sol.initialize())
    return sol, loc, rpy, vel, car


def test_line_offset_absent_is_identity(tmp_path):
    from submission import load_line_offset
    sol, *_ = _sol()
    assert load_line_offset(str(tmp_path / "nope.npz"), sol.track) is None
    sol._off = None
    s = 1234.0
    assert sol._ref(s, 0.7, 0.05) == (0.7, 0.05)
    assert sol._kappa_cmd_ref(s) == sol._kappa_ref(s)


def test_line_offset_constant_shift(tmp_path):
    from submission import load_line_offset
    sol, *_ = _sol()
    tr = sol.track
    f = tmp_path / "off.npz"
    np.savez(f, s=tr.s, d=np.full(len(tr.s), 1.0))
    sol._off = load_line_offset(str(f), tr)
    s_total = 1500.0
    ls = sol._line_s(s_total)
    e_y, e_psi = sol._ref(s_total, 1.0, 0.0)
    assert abs(e_y) < 1e-9 and abs(e_psi) < 1e-6  # car sitting on the shifted line, parallel to it
    k = float(np.interp(ls, tr.s, tr.kappa, period=tr.length))
    assert abs(sol._kappa_cmd_ref(s_total) - k / (1 - k * 1.0)) < 1e-4


def test_line_offset_ramp_heading_and_curvature(tmp_path):
    from submission import load_line_offset
    sol, *_ = _sol()
    tr = sol.track
    d = 0.5 * np.sin(2 * np.pi * tr.s / tr.length * 20)  # smooth periodic offset, 20 waves per lap
    f = tmp_path / "off.npz"
    np.savez(f, s=tr.s, d=d)
    sol._off = load_line_offset(str(f), tr)
    w = 2 * np.pi / tr.length * 20
    for s_total in (1100.0, 1777.0, 4000.0):
        ls = sol._line_s(s_total)
        e_y, e_psi = sol._ref(s_total, 0.0, 0.0)
        assert abs(e_y + 0.5 * np.sin(w * ls)) < 1e-3
        assert abs(e_psi + 0.5 * w * np.cos(w * ls)) < 1e-3
        k = float(np.interp(ls, tr.s, tr.kappa, period=tr.length))
        dd = 0.5 * np.sin(w * ls)
        assert abs(sol._kappa_cmd_ref(s_total) - (k / (1 - k * dd) - 0.5 * w * w * np.sin(w * ls))) < 2e-4


def test_line_offset_fades_in_after_start(tmp_path):
    from submission import OFFSET_FADE_M, load_line_offset
    sol, *_ = _sol()
    tr = sol.track
    f = tmp_path / "off.npz"
    np.savez(f, s=tr.s, d=np.full(len(tr.s), 1.0))
    sol._off = load_line_offset(str(f), tr)
    a, b = OFFSET_FADE_M
    assert sol._ref(0.5 * a, 0.0, 0.0)[0] == 0.0
    assert abs(sol._ref(0.5 * (a + b), 0.0, 0.0)[0] + 0.5) < 1e-9
    assert abs(sol._ref(b + 1, 0.0, 0.0)[0] + 1.0) < 1e-9


def test_tyre_model_option_measures_sideslip_and_yaw_rate():
    import json
    import os
    import submission
    data = os.path.join(os.path.dirname(submission.__file__), "data")
    p = json.load(open(os.path.join(data, "vehicle_params.json")))
    if "tyre" not in p:
        import pytest
        pytest.skip("no fitted tyre model in vehicle_params")
    loc, rpy, vel, car = Sensor(), Sensor(), Sensor(), Car()
    sol = RoarCompetitionSolution([], car, location_sensor=loc, velocity_sensor=vel, rpy_sensor=rpy)
    asyncio.run(sol.initialize())
    sol.tyre = True
    from tyre_mpc import TyreMPC
    sol.mpc = TyreMPC(sol.veh, dict(sol.veh.p["mpc"], model="tyre"))
    sol._off = None
    tr = sol.track
    i = 300
    psi = tr.psi[i]
    loc.val = np.r_[tr.xy[i], 0.0]
    rpy.val = np.array([0.0, 0.0, psi + 0.02])
    vel.val = 30.0 * np.r_[np.cos(psi), np.sin(psi), 0.0]  # moving along the line, nose 0.02 rad left of it
    asyncio.run(sol.step())
    assert abs(sol.last["beta"] + 0.02) < 1e-6  # course - yaw
    assert sol.last["r"] == 0.0  # first tick: no previous yaw
    j = i + 2
    loc.val = np.r_[tr.xy[j], 0.0]
    rpy.val = np.array([0.0, 0.0, psi + 0.02 + 0.01])
    asyncio.run(sol.step())
    assert abs(sol.last["r"] - 0.01 / sol.mpc.dt) < 1e-6
    assert sol.last["ok"] == 1
    assert np.isfinite(car.control["steer"]) and abs(car.control["steer"]) <= 1.0


def test_start_blend_slides_reference_onto_the_line():
    from submission import START_BLEND_M
    sol, loc, rpy, vel, car = _sol()
    sol._off = None
    tr = sol.track
    i = int(np.argmin(np.abs(tr.s - sol.profile_start_s)))
    n = np.array([-np.sin(tr.psi[i]), np.cos(tr.psi[i])])
    loc.val = np.r_[tr.xy[i] + 2.0 * n, 0.0]  # spawn 2 m left of the line
    rpy.val = np.array([0.0, 0.0, tr.psi[i]])
    vel.val = np.zeros(3)
    asyncio.run(sol.step())
    assert abs(sol._e_y0 - 2.0) < 0.05
    assert abs(sol._ref(0.0, 2.0, 0.0)[0]) < 1e-9  # no error at the start
    assert abs(sol._ref(0.5 * START_BLEND_M, 2.0, 0.0)[0] - 1.0) < 1e-9  # half way: 1 m of error
    assert sol._ref(START_BLEND_M + 1, 2.0, 0.0)[0] == 2.0  # blended out


def test_residual_policy_bounded_and_off_by_default(tmp_path):
    from submission import RESIDUAL_AHEAD_M, RESIDUAL_SCALE, ResidualPolicy
    nf = 7 + len(RESIDUAL_AHEAD_M)
    rng = np.random.default_rng(0)
    f = tmp_path / "res.npz"
    np.savez(f, W1=rng.normal(0, 5, (16, nf)), b1=np.zeros(16), W2=rng.normal(0, 5, (2, 16)), b2=np.zeros(2))
    pol = ResidualPolicy(str(f))
    for _ in range(50):
        dd, da = pol(rng.normal(0, 3, nf))
        assert abs(dd) <= RESIDUAL_SCALE[0] + 1e-12 and abs(da) <= RESIDUAL_SCALE[1] + 1e-12
    np.savez(f, W1=np.zeros((16, nf)), b1=np.zeros(16), W2=np.zeros((2, 16)), b2=np.zeros(2))
    assert ResidualPolicy(str(f))(np.ones(nf)) == (0.0, 0.0)
    import os
    import submission
    sol, *_ = _sol()
    shipped = os.path.exists(os.path.join(submission.DATA, "residual.npz"))
    assert (sol.residual is not None) == shipped  # loaded exactly when the file is shipped


def test_residual_ensemble_picks_the_section_network(tmp_path):
    from submission import RESIDUAL_AHEAD_M, ResidualPolicy
    nf, nh = 7 + len(RESIDUAL_AHEAD_M), 16
    K = 3
    W2 = np.zeros((K, 2, nh))
    b2 = np.array([[0.0, 0.5], [0.0, -0.5], [0.0, 0.9]])  # each section network: a constant accel correction
    f = tmp_path / "ens.npz"
    np.savez(f, W1=np.zeros((K, nh, nf)), b1=np.zeros((K, nh)), W2=W2, b2=b2, cuts=np.array([100.0, 1000.0, 3000.0]))
    pol = ResidualPolicy(str(f))
    x = np.zeros(nf)
    assert pol(x, 500.0)[1] > 0 and pol(x, 2000.0)[1] < 0 and pol(x, 4000.0)[1] > 1.0
    assert abs(pol(x, 50.0)[1] - pol(x, 4000.0)[1]) < 1e-12  # before the first cut = the last section (wraps)


def test_residual_ensemble_without_cuts_fails_at_load(tmp_path):
    import pytest
    from submission import RESIDUAL_AHEAD_M, ResidualPolicy
    nf = 7 + len(RESIDUAL_AHEAD_M)
    f = tmp_path / "bad.npz"
    np.savez(f, W1=np.zeros((3, 16, nf)), b1=np.zeros((3, 16)), W2=np.zeros((3, 2, 16)), b2=np.zeros((3, 2)))
    with pytest.raises(ValueError):
        ResidualPolicy(str(f))
