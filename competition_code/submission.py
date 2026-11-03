"""Hybrid submission: offline TOPP-RA speed profile + LTV Frenet MPC tracking a fixed racing line.

step(): localize on the line (Progress) -> FrenetMPC (curvature, accel) -> steer / pedal maps.
"""

import os
from typing import List

import numpy as np
import roar_py_interface

from mpc import FrenetMPC
from track import TELEPORT_JUMP_M, Progress, Track
from vehicle import Vehicle, load_profile

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
LOG_PATH = os.path.join(os.path.dirname(HERE), "logs", "run.csv")
LOG_COLS = ("tick", "s_total", "line_s", "e_y", "e_psi", "v", "v_ref", "kappa_ref", "kappa_cmd", "a_cmd",
            "steer", "throttle", "brake", "ok")
START_S_TOL_M = 5.0


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
        self.track = tr = Track(np.load(os.path.join(DATA, "racing_line.npz"))["xy"])
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
        # Sideslip feed-forward: the CARLA car's body points into the turn by beta ~ k*a_lat (run 1 fit:
        # 0.003-0.005 rad per m/s^2), so its velocity points outward. The MPC's e_y' = v*e_psi model assumes no
        # slip: track the course instead by asking for yaw = line heading + beta.
        e_psi -= self.veh.p["sideslip_k"] * v * v * self._kappa_ref(s_total)
        if self.tick == 0:
            L = self.track.length
            d = (self.progress.start_s - self.profile_start_s + L / 2) % L - L / 2
            if abs(d) > START_S_TOL_M:
                print(f"WARNING: start_s {self.progress.start_s:.1f} is {d:+.1f} m from the profile's "
                      f"{self.profile_start_s:.1f}: v_ref is offset along the line")

        kappa, a, ok = self.mpc.solve(e_y, e_psi, v, s_total, self._kappa_ref, self._v_ref_mpc, self.u_prev,
                                      self._a_lat)
        # Phase lead on the curvature command: the car's yaw response lags the steer by ~0.2-0.3 s above 30 m/s,
        # which the MPC model leaves out.
        k_steer = kappa + self.veh.p["steer_lead_s"] * (kappa - self.u_prev[0]) / self.mpc.dt
        self.u_prev = (kappa, a)
        steer = self.veh.steer_for(k_steer, v)
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

        self.last = dict(tick=self.tick, s_total=s_total, line_s=self._line_s(s_total), e_y=e_y, e_psi=e_psi, v=v,
                         v_ref=self.v_ref(s_total), kappa_ref=self._kappa_ref(s_total), kappa_cmd=kappa, a_cmd=a,
                         steer=steer, throttle=throttle, brake=brake, ok=int(ok))
        if self._log:
            self._log.write(",".join(f"{self.last[c]:.6g}" for c in LOG_COLS) + "\n")
        self.tick += 1
        return control
