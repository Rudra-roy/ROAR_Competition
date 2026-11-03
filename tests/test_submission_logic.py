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
