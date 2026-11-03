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
