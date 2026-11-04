"""RL training worker: one persistent CARLA server + client per sim slot, scoring corner episodes.

  python tools/rl/worker.py <slot> <checkout dir>      (reads JSON jobs on stdin, writes JSON results on stdout)

Setup mirrors competition_runner.evaluate_solution (same car, location/velocity/rpy/collision sensors, 0.05 s
synchronous ticks), so behaviour transfers to the official race. An episode places the car at a corner's start state
(position, heading, speed taken from a clean reference race) with set_transform / set_linear_3d_velocity - training
only; the submission never does this - then runs the submission's own controller (with a residual policy's weights
installed) to the corner's end line and reports the time, or a collision. Job: {"weights": {W1,b1,W2,b2} | null,
"episodes": [[start_line_s, end_line_s], ...]}. Result: {"times": [...], "coll": [...], "exit_v": [...]}.
"""
import asyncio
import json
import os
import subprocess
import sys
import time

import numpy as np

SLOT, WT = int(sys.argv[1]), os.path.abspath(sys.argv[2])
PORT = 2000 + 3 * SLOT
PARENT = "/home/mt-labpc/Documents/roar_racing"
sys.path[:0] = [os.path.join(WT, "competition_code")]
import carla  # noqa: E402
import roar_py_carla  # noqa: E402
import roar_py_interface  # noqa: E402
import submission  # noqa: E402
from infrastructure import RoarCompetitionAgentWrapper  # noqa: E402
from track import load_track  # noqa: E402

REF_RUN = os.path.join(WT, "competition_code", "data", "rl_reference.run.csv")  # clean race: start states
TIMEOUT_S = 40.0
DT = 0.05


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def start_server():
    env = dict(os.environ)
    env["PATH"] = ":".join(p for p in env["PATH"].split(":") if "/.venv/" not in p)
    out = open(os.path.join(WT, "logs", "rl_server_%d.log" % SLOT), "w")
    subprocess.Popen([os.path.join(PARENT, "run_monza.sh"), "-RenderOffScreen", "-carla-rpc-port=%d" % PORT,
                      "-carla-streaming-port=%d" % (PORT + 1)], stdout=out, stderr=out, env=env,
                     start_new_session=True)
    import socket
    for _ in range(300):
        try:
            socket.create_connection(("127.0.0.1", PORT), 1).close()
            time.sleep(5)
            return
        except OSError:
            time.sleep(1)
    raise RuntimeError("server on %d did not come up" % PORT)


class Env:
    def __init__(self):
        client = carla.Client("127.0.0.1", PORT)
        client.set_timeout(120.0)
        self.inst = roar_py_carla.RoarPyCarlaInstance(client)
        self.world = w = self.inst.world
        w.set_control_steps(0.05, 0.005)
        w.set_asynchronous(False)
        self.wps = w.maneuverable_waypoints
        self.wz = np.array([p.location for p in self.wps])
        self.vehicle = v = w.spawn_vehicle("vehicle.tesla.model3", self.wps[0].location + np.array([0, 0, 1]),
                                           self.wps[0].roll_pitch_yaw, True)
        self.loc = v.attach_location_in_world_sensor()
        self.vel = v.attach_velocimeter_sensor()
        self.rpy = v.attach_roll_pitch_yaw_sensor()
        self.col = v.attach_collision_sensor(np.zeros(3), np.zeros(3))
        self.ref = np.genfromtxt(REF_RUN, delimiter=",", names=True)
        self.track = load_track(os.path.join(WT, "competition_code", "data", "racing_line.npz"))
        L = self.track.length
        lap2 = (self.ref["s_total"] >= L) & (self.ref["s_total"] < 2 * L)
        self.r2 = {k: self.ref[k][lap2] for k in self.ref.dtype.names}

    def start_state(self, line_s):
        """(xyz, rpy, velocity vector) of the reference race at line_s on its flying lap."""
        r, tr = self.r2, self.track
        i = int(np.argmin(np.abs(((r["line_s"] - line_s + tr.length / 2) % tr.length) - tr.length / 2)))
        ls, e_y, yaw, v, beta = r["line_s"][i], r["e_y"][i], r["yaw"][i], r["v"][i], r["beta"][i]
        x = np.interp(ls, tr.s, tr.xy[:, 0], period=tr.length)
        y = np.interp(ls, tr.s, tr.xy[:, 1], period=tr.length)
        psi = np.interp(ls, tr.s, np.unwrap(tr.psi), period=tr.length)
        p = np.array([x - e_y * np.sin(psi), y + e_y * np.cos(psi)])
        z = self.wz[int(np.argmin(np.linalg.norm(self.wz[:, :2] - p, axis=1))), 2] + 0.3
        return np.array([p[0], p[1], z]), np.array([0.0, 0.0, yaw]), v * np.array([np.cos(yaw + beta), np.sin(yaw + beta), 0.0])

    async def episode(self, weights, s0, s1):
        w, veh = self.world, self.vehicle
        xyz, rpy, vel = self.start_state(s0)
        veh.set_transform(xyz, rpy)
        veh.set_angular_velocity(np.zeros(3))
        veh.set_linear_3d_velocity(vel)
        await w.step()
        veh.set_linear_3d_velocity(vel)
        await w.step()
        await veh.receive_observation()
        sol = submission.RoarCompetitionSolution(self.wps, RoarCompetitionAgentWrapper(veh), None, self.loc, self.vel,
                                                 self.rpy, None, self.col)
        await sol.initialize()
        if weights is None:
            sol.residual = None
        else:
            sol.residual = submission.ResidualPolicy.__new__(submission.ResidualPolicy)
            sol.residual.W1, sol.residual.b1, sol.residual.W2, sol.residual.b2 = (np.asarray(weights[k], float)
                                                                                  for k in ("W1", "b1", "W2", "b2"))
        L = self.track.length
        prof0 = sol.profile_start_s
        target_total = (s0 - prof0) % L + L  # flying-lap profile position of the start point
        # localise once, then re-key progress to the race's own start so the speed profile, grip sections and learned
        # offset are read at the right place (in a real race the car gets here by driving from the start line)
        p0 = np.asarray(self.loc.get_last_gym_observation(), float)
        sol.progress.update(p0, float(self.rpy.get_last_gym_observation()[2]))
        s_car = sol.progress._s_prev
        sol.progress.start_s = prof0
        sol.progress._total = target_total + ((s_car - s0 + L / 2) % L - L / 2)
        sol._p_prev = p0[:2]
        sol.u_prev = (float(sol.mpc.ty.steady(np.interp(s_car, self.track.s, self.track.kappa, period=L),
                                              float(np.linalg.norm(vel)))[0]), 0.0)
        dist = (s1 - s0) % L
        t, ticks = 0.0, 0
        while True:
            await veh.receive_observation()
            if np.linalg.norm(self.col.get_last_observation().impulse_normal) > 100.0:
                return t, True, 0.0
            await sol.step()
            await w.step()
            ticks += 1
            t = ticks * DT
            done = sol.progress._total - target_total - ((s_car - s0 + L / 2) % L - L / 2)
            if done >= dist:
                return t, False, float(np.linalg.norm(self.vel.get_last_gym_observation()))
            if t > TIMEOUT_S or (t > 3 and np.linalg.norm(self.vel.get_last_gym_observation()) < 3.0):
                return TIMEOUT_S, True, 0.0


async def main():
    start_server()
    env = Env()
    log("worker %d ready on %d" % (SLOT, PORT))
    print(json.dumps({"ready": SLOT}), flush=True)
    for line in sys.stdin:
        job = json.loads(line)
        times, coll, exit_v = [], [], []
        for s0, s1 in job["episodes"]:
            try:
                t, c, ve = await env.episode(job.get("weights"), s0, s1)
            except Exception as e:  # keep the worker alive; a failed episode scores as a collision
                log("episode failed: %r" % e)
                t, c, ve = TIMEOUT_S, True, 0.0
            times.append(t), coll.append(bool(c)), exit_v.append(ve)
        print(json.dumps({"times": times, "coll": coll, "exit_v": exit_v}), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
