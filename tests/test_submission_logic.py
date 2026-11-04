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
    loc, rpy, vel, car = Sensor(), Sensor(), Sensor(), Car()
    sol = RoarCompetitionSolution([], car, location_sensor=loc, velocity_sensor=vel, rpy_sensor=rpy)
    asyncio.run(sol.initialize())
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


def test_pursuit_kappa_on_circle_and_straight():
    sol, *_ = _sol()
    sol._off = None
    tr = sol.track
    i = 2000
    p, yaw = tr.xy[i], tr.psi[i]
    s_total = (tr.s[i] - sol.progress.start_s) % tr.length
    k = sol._pursuit_kappa(p, yaw, s_total, 30.0)
    j = (i + int(30 / tr.ds)) % len(tr.s)
    kmean = tr.kappa[i:j].mean() if j > i else tr.kappa[i]
    assert abs(k - kmean) < 0.01 + 0.5 * abs(kmean)  # on the line: roughly the line's own curvature ahead
    k_left = sol._pursuit_kappa(p + 2.0 * np.r_[-np.sin(yaw), np.cos(yaw)], yaw, s_total, 30.0)
    assert k_left < k  # car 2 m left of the line steers back right
