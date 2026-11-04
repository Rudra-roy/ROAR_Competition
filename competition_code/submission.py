"""Hybrid submission: offline TOPP-RA speed profile + LTV Frenet MPC tracking a fixed racing line.

step(): localize on the line (Progress) -> FrenetMPC (curvature, accel) -> steer / pedal maps.
"""

import os
from typing import List

import numpy as np
import roar_py_interface

from mpc import FrenetMPC
from track import TELEPORT_JUMP_M, Progress, load_track
from vehicle import Vehicle, load_profile

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
LOG_PATH = os.path.join(os.path.dirname(HERE), "logs", "run.csv")
LOG_COLS = ("tick", "s_total", "line_s", "e_y", "e_psi", "yaw", "v", "v_ref", "kappa_ref", "kappa_cmd", "a_cmd",
            "steer", "throttle", "brake", "ok", "d_ref")
START_S_TOL_M = 5.0


def load_line_offset(path, track):
    """Learned lateral reference offset (tools/ilc.py): d(s) metres left of the racing line, so that the DRIVEN path
    lands on the line. Returns (d, d', d'') on track.s, or None when the file is absent (= plain racing line)."""
    if not os.path.exists(path):
        return None
    f = np.load(path)
    d = np.interp(track.s, f["s"], f["d"], period=track.length)
    h = track.ds
    d1 = (np.roll(d, -1) - np.roll(d, 1)) / (2 * h)
    d2 = (np.roll(d, -1) - 2 * d + np.roll(d, 1)) / (h * h)
    return d, d1, d2


class RoarCompetitionSolution:
    def __init__(
        self,
        maneuverable_waypoints: List[roar_py_interface.RoarPyWaypoint],
        vehicle : roar_py_interface.RoarPyActor,
        camera_sensor : roar_py_interface.RoarPyCameraSensor = None,
        location_sensor : roar_py_interface.RoarPyLocationInWorldSensor = None,
        velocity_sensor : roar_py_interface.RoarPyVelocimeterSensor = None,
        rpy_sensor : roar_py_interface.RoarPyRollPitchYawSensor = None,
        occupancy_map_sensor : roar_py_interface.RoarPyOccupancyMapSensor = None,
        collision_sensor : roar_py_interface.RoarPyCollisionSensor = None,
    ) -> None:
        self.maneuverable_waypoints = maneuverable_waypoints
        self.vehicle = vehicle
        self.camera_sensor = camera_sensor
        self.location_sensor = location_sensor
        self.velocity_sensor = velocity_sensor
        self.rpy_sensor = rpy_sensor
        self.occupancy_map_sensor = occupancy_map_sensor
        self.collision_sensor = collision_sensor

    async def initialize(self) -> None:
        self.track = tr = load_track(os.path.join(DATA, "racing_line.npz"))
        self.veh = veh = Vehicle.from_json(os.path.join(DATA, "vehicle_params.json"))
        self.progress = Progress(tr)
        self.mpc = FrenetMPC(veh, veh.p["mpc"])
        prof = os.path.join(DATA, "speed_profile.npz")
        self.v_ref = load_profile(prof, speed_scale=veh.p["speed_scale"])
        self.profile_start_s = float(np.load(prof)["start_s"])
        # The profile starts from rest at v = 0: with the car stopped the whole MPC horizon sits at s = 0 and
        # v_ref = 0, so it would never pull away. A small floor gets it rolling onto the profile.
        v_min = veh.p["v_ref_min"]
        self._v_ref_mpc = lambda s: max(self.v_ref(s), v_min)
        self._kappa_ref = lambda s: float(np.interp(self._line_s(s), tr.s, tr.kappa, period=tr.length))
        self._a_lat = lambda s: veh.a_lat(self._line_s(s))
        self._off = load_line_offset(os.path.join(DATA, "line_offset.npz"), tr)
        self.u_prev = (0.0, 0.0)
        self.tick = 0
        self._p_prev = None
        self.last = None
        self._log = None
        if os.environ.get("ROAR_LOG") == "1":
            os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
            self._log = open(LOG_PATH, "w", buffering=1)
            self._log.write(",".join(LOG_COLS) + "\n")

    def _line_s(self, s_total):
        return (self.progress.start_s + s_total) % self.track.length

    def _off_at(self, s_total, k):
        tr = self.track
        return float(np.interp(self._line_s(s_total), tr.s, self._off[k], period=tr.length))

    def _ref(self, s_total, e_y, e_psi):
        """Errors w.r.t. the reference the MPC tracks: the racing line shifted by the learned offset (linearised)."""
        if self._off is None:
            return e_y, e_psi
        return e_y - self._off_at(s_total, 0), e_psi - self._off_at(s_total, 1)

    def _kappa_cmd_ref(self, s):
        """Curvature of the shifted reference: offset curve of the line plus the offset's own bending."""
        k = self._kappa_ref(s)
        if self._off is None:
            return k
        return k / (1.0 - k * self._off_at(s, 0)) + self._off_at(s, 2)

    async def step(self) -> None:
        p = np.asarray(self.location_sensor.get_last_gym_observation(), dtype=float)
        yaw = float(self.rpy_sensor.get_last_gym_observation()[2])
        v = float(np.linalg.norm(self.velocity_sensor.get_last_gym_observation()))
        # Same teleport test as Progress: a respawn restarts progress, so forget the MPC's state too.
        if self._p_prev is not None and np.hypot(*(p[:2] - self._p_prev)) > TELEPORT_JUMP_M:
            self.mpc.reset()
            self.u_prev = (0.0, 0.0)
        self._p_prev = p[:2]
        s_total, e_y, e_psi, _ = self.progress.update(p, yaw)
        a_lat = self._a_lat(s_total)
        # Sideslip feed-forward: the CARLA car's body points into the turn by beta ~ k*a_lat (tools/fit_sideslip.py),
        # so its velocity points outward. The MPC's e_y' = v*e_psi model assumes no slip: track the course instead by
        # asking for yaw = line heading + beta. A function of s and v only: feeding back the measured course
        # (velocity direction) made the car weave into the wall (Task 7, Task 8c course1).
        # Clipped to the grip limit: over speed at corner entry v^2*kappa would ask for a huge heading offset.
        e_psi -= self.veh.p["sideslip_k"] * float(np.clip(v * v * self._kappa_cmd_ref(s_total), -a_lat, a_lat))
        e_y_mpc, e_psi = self._ref(s_total, e_y, e_psi)
        if self.tick == 0:
            L = self.track.length
            d = (self.progress.start_s - self.profile_start_s + L / 2) % L - L / 2
            if abs(d) > START_S_TOL_M:
                print(f"WARNING: start_s {self.progress.start_s:.1f} is {d:+.1f} m from the profile's "
                      f"{self.profile_start_s:.1f}: v_ref is offset along the line")

        kappa, a, ok = self.mpc.solve(e_y_mpc, e_psi, v, s_total, self._kappa_cmd_ref, self._v_ref_mpc,
                                      self.u_prev, self._a_lat)
        self.u_prev = (kappa, a)
        steer = self.veh.steer_for(kappa, v)
        throttle, brake = self.veh.pedals_for(a, v)
        control = {
            "throttle": throttle,
            "steer": steer,
            "brake": brake,
            "hand_brake": 0.0,
            "reverse": 0,
            "target_gear": 0,
        }
        await self.vehicle.apply_action(control)

        self.last = dict(tick=self.tick, s_total=s_total, line_s=self._line_s(s_total), e_y=e_y, e_psi=e_psi,
                         yaw=yaw, v=v, v_ref=self.v_ref(s_total), kappa_ref=self._kappa_ref(s_total), kappa_cmd=kappa,
                         a_cmd=a, steer=steer, throttle=throttle, brake=brake, ok=int(ok),
                         d_ref=0.0 if self._off is None else self._off_at(s_total, 0))
        if self._log:
            self._log.write(",".join(f"{self.last[c]:.6g}" for c in LOG_COLS) + "\n")
        self.tick += 1
        return control
