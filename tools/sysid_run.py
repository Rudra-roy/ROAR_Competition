"""One sysid lap through the official runner; logs every tick to logs/sysid.csv.

Run with the sim server up: python tools/sysid_run.py  (any cwd). Exits via os._exit(0) after one lap or at the first collision,
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
from track import Progress, load_track  # noqa: E402

DT = 0.05
LOOKAHEAD_KMH = [(90, 18.0), (110, 22.0), (130, 28.0), (160, 36.0), (180, 44.0), (200, 52.0),
                 (250, 60.0), (300, 70.0)]  # (speed bound km/h, lookahead m), winners' table
LOOKAHEAD_ELSE = 16.0
WHEELBASE_PP = 4.7
STEER_MUL = 1.5
A_LAT = 12.0         # m/s^2 -> corner speed sqrt(A_LAT/|kappa|)
V_CAP = 80.0
A_DECEL_PLAN = 10.0  # m/s^2 used for the backward braking pass (measured ~16)


def lookahead_m(v):
    kmh = v * 3.6
    return next((L for bound, L in LOOKAHEAD_KMH if kmh < bound), LOOKAHEAD_ELSE)


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
        self.collision_sensor = collision_sensor

    async def initialize(self):
        os.makedirs(LOGS, exist_ok=True)
        np.save(os.path.join(LOGS, "maneuverable_waypoints.npy"), np.array([w.location for w in self.wps]))
        self.track = tr = load_track(os.path.join(REPO, "competition_code", "data", "racing_line.npz"))
        self.progress = Progress(tr)
        vl = np.minimum(V_CAP, np.sqrt(A_LAT / np.maximum(np.abs(tr.kappa), 1e-9)))
        m = len(vl)
        for i in range(2 * m - 1, -1, -1):  # backward braking pass, twice round the loop for the wrap
            vl[i % m] = min(vl[i % m], np.sqrt(vl[(i + 1) % m] ** 2 + 2 * A_DECEL_PLAN * tr.ds))
        self.v_lim = vl
        self.f = open(os.path.join(LOGS, "sysid.csv"), "w")
        self.f.write("t,x,y,yaw,v,throttle,brake,steer,yaw_rate\n")
        self.tick, self.prev_yaw, self.vmax = 0, None, 0.0
        self.braking = False

    async def step(self):
        tr = self.track
        p = self.location_sensor.get_last_gym_observation()
        yaw = float(self.rpy_sensor.get_last_gym_observation()[2])
        v = float(np.linalg.norm(self.velocity_sensor.get_last_gym_observation()))
        hit = np.linalg.norm(self.collision_sensor.get_last_observation().impulse_normal) > 100.0
        jump = self.tick > 0 and np.hypot(*(p[:2] - self.prev_p)) > 6.0
        if hit or jump:  # stop at the first collision; nothing after it is logged
            self._finish(f"COLLISION at tick {self.tick}: hit={hit} jump={jump}")
        s_total, e_y, _, idx = self.progress.update(p, yaw)
        if s_total >= tr.length:
            self._finish("lap done")

        # winners' pure pursuit; alpha > 0 = target to the right -> positive steer = right
        L = lookahead_m(v)
        tx, ty = tr.xy[(idx + int(round(L / tr.ds))) % len(tr.s)]
        alpha = _wrap(yaw - np.arctan2(ty - p[1], tx - p[0]))
        steer = float(np.clip(STEER_MUL * np.arctan(2 * WHEELBASE_PP * np.sin(alpha) / L), -1.0, 1.0))

        vl = self.v_lim[idx]
        # brake latches: once on (v > v_lim + 1) it stays full until v <= v_lim, for clean full-brake data
        self.braking = v > vl + 1.0 or (self.braking and v > vl)
        if self.braking:
            thr, brk = 0.0, 1.0
        elif v <= vl - 0.5:
            thr, brk = 1.0, 0.0
        else:
            thr, brk = float(np.clip(0.3 + 0.3 * (vl - v), 0.0, 1.0)), 0.0
        await self.vehicle.apply_action({"throttle": thr, "steer": steer, "brake": brk,
                                         "hand_brake": 0.0, "reverse": 0, "target_gear": 0})

        yr = 0.0 if self.prev_yaw is None else _wrap(yaw - self.prev_yaw) / DT
        self.f.write(f"{self.tick * DT:.2f},{p[0]:.3f},{p[1]:.3f},{yaw:.5f},{v:.4f},"
                     f"{thr:.3f},{brk:.3f},{steer:.4f},{yr:.5f}\n")
        if self.tick % 200 == 0 or self.tick < 3:
            print(f"SYSID t={self.tick * DT:.1f} s_total={s_total:.0f} v={v:.1f} v_lim={vl:.1f} e_y={e_y:+.2f} alpha={alpha:+.3f} steer={steer:+.3f}",
                  flush=True)
        self.prev_yaw, self.prev_p, self.vmax = yaw, p[:2].copy(), max(self.vmax, v)
        self.tick += 1

    def _finish(self, why):
        self.f.close()
        print(f"SYSID {why}: ticks={self.tick} sim={self.tick * DT:.2f}s vmax={self.vmax:.2f}", flush=True)
        os._exit(0)


competition_runner.RoarCompetitionSolution = SysIdSolution
if __name__ == "__main__":
    asyncio.run(competition_runner.main())
