"""One sysid lap through the official runner; logs every tick to logs/sysid.csv.

Run with the sim server up: python tools/sysid_run.py  (any cwd). Exits via os._exit(0) after one lap,
which leaves the server hung -- kill it afterwards (wineserver -k).
"""
import asyncio
import os
import sys

import carla
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(REPO, "logs")
sys.path.insert(0, os.path.join(REPO, "competition_code"))

_orig = carla.Client.set_timeout  # Wine ticks are slow (same patch as demo.py)
carla.Client.set_timeout = lambda self, t: _orig(self, max(t, 120.0))

import competition_runner  # noqa: E402
from track import Progress, Track  # noqa: E402

DT = 0.05
K_PP = 2.4           # steer per unit path curvature (~wheelbase 2.9 m / 70 deg max wheel angle)
KAPPA_STRAIGHT = 0.004
LOOK_M = 250.0
A_LAT_CORNER = 12.0  # m/s^2 -> corner speed sqrt(a/kappa), capped at V_CORNER
V_CORNER = 25.0


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class SysIdSolution:
    def __init__(self, maneuverable_waypoints, vehicle, camera_sensor=None, location_sensor=None,
                 velocity_sensor=None, rpy_sensor=None, occupancy_map_sensor=None, collision_sensor=None):
        self.wps = maneuverable_waypoints
        self.vehicle = vehicle
        self.location_sensor = location_sensor
        self.velocity_sensor = velocity_sensor
        self.rpy_sensor = rpy_sensor

    async def initialize(self):
        os.makedirs(LOGS, exist_ok=True)
        np.save(os.path.join(LOGS, "maneuverable_waypoints.npy"), np.array([w.location for w in self.wps]))
        self.track = tr = Track(np.load(os.path.join(REPO, "competition_code", "data", "racing_line.npz"))["xy"])
        self.progress = Progress(tr)
        ak, n = np.abs(tr.kappa), int(LOOK_M / tr.ds)
        self.kmax_ahead = np.max([np.roll(ak, -i) for i in range(n + 1)], axis=0)
        self.f = open(os.path.join(LOGS, "sysid.csv"), "w")
        self.f.write("t,x,y,yaw,v,throttle,brake,steer,yaw_rate\n")
        self.tick, self.prev_yaw, self.vmax = 0, None, 0.0

    async def step(self):
        tr = self.track
        p = self.location_sensor.get_last_gym_observation()
        yaw = float(self.rpy_sensor.get_last_gym_observation()[2])
        v = float(np.linalg.norm(self.velocity_sensor.get_last_gym_observation()))
        s_total, _, _, idx = self.progress.update(p, yaw)
        if s_total >= tr.length:
            self.f.close()
            print(f"SYSID lap done: ticks={self.tick} sim={self.tick * DT:.2f}s vmax={self.vmax:.2f}", flush=True)
            os._exit(0)

        # pure pursuit; positive steer turns RIGHT, so left curvature -> negative steer
        d = tr.xy[(idx + int(round(max(15.0, 0.5 * v) / tr.ds))) % len(tr.s)] - p[:2]
        curv = 2.0 * np.sin(_wrap(np.arctan2(d[1], d[0]) - yaw)) / max(np.hypot(*d), 1e-3)
        steer = -K_PP * curv

        km = self.kmax_ahead[idx]
        if km < KAPPA_STRAIGHT:
            thr, brk = 1.0, 0.0
            if v < 40.0:
                steer += 0.05 if (self.tick * DT) % 1.0 < 0.5 else -0.05
        else:
            vc = min(V_CORNER, np.sqrt(A_LAT_CORNER / km))
            if v > vc + 1.0:
                thr, brk = 0.0, 1.0
            else:
                thr, brk = float(np.clip(0.3 + 0.3 * (vc - v), 0.0, 1.0)), 0.0
        steer = float(np.clip(steer, -1.0, 1.0))
        await self.vehicle.apply_action({"throttle": thr, "steer": steer, "brake": brk,
                                         "hand_brake": 0.0, "reverse": 0, "target_gear": 0})

        yr = 0.0 if self.prev_yaw is None else _wrap(yaw - self.prev_yaw) / DT
        self.f.write(f"{self.tick * DT:.2f},{p[0]:.3f},{p[1]:.3f},{yaw:.5f},{v:.4f},"
                     f"{thr:.3f},{brk:.3f},{steer:.4f},{yr:.5f}\n")
        if self.tick % 200 == 0:
            print(f"SYSID t={self.tick * DT:.1f} s_total={s_total:.0f} v={v:.1f} km={km:.4f}", flush=True)
        self.prev_yaw, self.vmax = yaw, max(self.vmax, v)
        self.tick += 1


competition_runner.RoarCompetitionSolution = SysIdSolution
if __name__ == "__main__":
    asyncio.run(competition_runner.main())
