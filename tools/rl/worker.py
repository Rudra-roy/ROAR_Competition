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
VIEW = os.environ.get("ROAR_RL_VIEW") == str(SLOT)  # this slot opens a window with a chase camera
PORT = 2000 + 3 * SLOT
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.join(WT, "competition_code")]
import carla  # noqa: E402
import roar_py_carla  # noqa: E402
import roar_py_interface  # noqa: E402
import submission  # noqa: E402
from infrastructure import RoarCompetitionAgentWrapper  # noqa: E402
from track import load_track  # noqa: E402

REF_RUN = os.path.join(WT, "competition_code", "data", "rl_reference.run.csv")  # clean race: start states
TIMEOUT_S = 40.0
RISK_SLIP_DEG, RISK_EY_M = 12.0, 1.0  # near-miss thresholds: slip beyond 12 deg, more than 1 m off the line
DT = 0.05


class NoisyAgent(RoarCompetitionAgentWrapper):
    """Adds a slowly varying steering offset (AR(1), std `std`) to every command: trajectory variety in training
    without teleports, so a policy cannot fit one exact path."""

    def __init__(self, wrapped, std, seed):
        super().__init__(wrapped)
        self._rng, self._std, self._x = np.random.default_rng(seed), std, 0.0

    async def apply_action(self, action):
        if self._std > 0:
            self._x = 0.95 * self._x + np.sqrt(1 - 0.95 ** 2) * self._std * self._rng.normal()
            action = dict(action, steer=float(np.clip(action["steer"] + self._x, -1.0, 1.0)))
        return await super().apply_action(action)


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def start_server():
    env = dict(os.environ)
    env["PATH"] = ":".join(p for p in env["PATH"].split(":") if "/.venv/" not in p)
    out = open(os.path.join(WT, "logs", "rl_server_%d.log" % SLOT), "w")
    args = ["-RenderOffScreen", "-carla-rpc-port=%d" % PORT, "-carla-streaming-port=%d" % (PORT + 1)]
    subprocess.Popen([os.path.join(REPO, "tools", "sim", "run_monza.sh")] + args, stdout=out, stderr=out, env=env,
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


class TrainView:
    """Game-style training window (viewed slot only): a chase camera on the car, a fixed top status bar (run,
    generation, gain, penalty, exploration, FPS) and a bottom bar (speed, slip, pedal bars, policy corrections)."""

    W, H = 1280, 720

    def __init__(self, vehicle):
        import pygame
        self.pg = pygame
        self.screen = None  # the window opens at the first rendered frame (not while the sim boots)
        ext = vehicle.bounding_box.extent
        self.cam = vehicle.attach_camera_sensor(roar_py_interface.RoarPyCameraSensorDataRGB,
                                                np.array([-2.6 * ext[0], 0.0, 3.2 * ext[2]]),
                                                np.array([0.0, 12.0 / 180.0 * np.pi, 0.0]),
                                                image_width=self.W, image_height=self.H)
        self.clock = None

    def _open(self):
        pg = self.pg
        pg.init()
        self.screen = pg.display.set_mode((self.W, self.H))
        pg.display.set_caption("ROAR RL training - slot %d" % SLOT)
        self.font = pg.font.SysFont("dejavusansmono", 20, bold=True)
        self.big = pg.font.SysFont("dejavusansmono", 44, bold=True)
        self.clock = pg.time.Clock()

    def pump(self):
        """Keep the window responsive while waiting (between laps, for the next job)."""
        if self.screen is not None:
            self.pg.event.pump()

    def _bar(self, x, y, w, h, frac, color, label):
        pg = self.pg
        pg.draw.rect(self.screen, (60, 60, 60), (x, y, w, h))
        pg.draw.rect(self.screen, color, (x, y, int(w * max(0.0, min(frac, 1.0))), h))
        self.screen.blit(self.font.render(label, True, (240, 240, 240)), (x, y - 24))

    def render(self, sol, info):
        pg = self.pg
        if self.screen is None:
            self._open()
        for e in pg.event.get():
            if e.type == pg.QUIT:  # closing the window only hides the view; training continues
                pg.display.iconify()
        obs = self.cam.get_last_observation()
        if obs is not None:
            img = obs.get_image()
            self.screen.blit(pg.image.fromstring(img.tobytes(), img.size, img.mode).convert(), (0, 0))
        else:
            self.screen.fill((20, 20, 20))
        top = pg.Surface((self.W, 40), pg.SRCALPHA)
        top.fill((0, 0, 0, 170))
        self.screen.blit(top, (0, 0))
        self.clock.tick()
        self.screen.blit(self.font.render(info or "", True, (255, 255, 255)), (12, 9))
        fps = self.font.render("FPS %3.0f" % self.clock.get_fps(), True, (120, 255, 120))
        self.screen.blit(fps, (self.W - fps.get_width() - 12, 9))
        last = getattr(sol, "last", None)
        if last is not None:
            bot = pg.Surface((self.W, 96), pg.SRCALPHA)
            bot.fill((0, 0, 0, 170))
            self.screen.blit(bot, (0, self.H - 96))
            y0 = self.H - 96
            self.screen.blit(self.big.render("%3.0f km/h" % (last["v"] * 3.6), True, (255, 255, 255)), (16, y0 + 22))
            slip = np.degrees(last["beta"])
            col = (255, 90, 90) if abs(slip) > 12 else (255, 255, 255)
            self.screen.blit(self.font.render("SLIP %+5.1f deg" % slip, True, col), (290, y0 + 20))
            self.screen.blit(self.font.render("LAP %4.0f m" % (last["line_s"]), True, (200, 200, 200)), (290, y0 + 56))
            self._bar(560, y0 + 36, 180, 16, last["throttle"], (60, 200, 90), "THROTTLE")
            self._bar(770, y0 + 36, 180, 16, last["brake"], (220, 60, 60), "BRAKE")
            dd, da = getattr(sol, "_res_dd", 0.0), getattr(sol, "_res_da", 0.0)
            self.screen.blit(self.font.render("POLICY steer %+5.1f mrad" % (1000 * dd), True, (120, 200, 255)),
                             (980, y0 + 20))
            self.screen.blit(self.font.render("POLICY accel %+5.2f m/s2" % da, True, (120, 200, 255)),
                             (980, y0 + 56))
        pg.display.flip()


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
        self.view = TrainView(v) if VIEW else None
        self.info = ""
        self.ref = np.genfromtxt(REF_RUN, delimiter=",", names=True)
        self.track = load_track(os.path.join(WT, "competition_code", "data", "racing_line.npz"))
        L = self.track.length
        lap2 = (self.ref["s_total"] >= L) & (self.ref["s_total"] < 2 * L)
        self.r2 = {k: self.ref[k][lap2] for k in self.ref.dtype.names}

    def start_state(self, line_s, pert=(0.0, 0.0, 0.0)):
        """(xyz, rpy, velocity vector) of the reference race at line_s on its flying lap, perturbed by
        pert = (speed factor - 1, lateral offset m, heading rad): randomised starts so the policy meets varied
        situations instead of one exact trajectory per corner."""
        r, tr = self.r2, self.track
        i = int(np.argmin(np.abs(((r["line_s"] - line_s + tr.length / 2) % tr.length) - tr.length / 2)))
        ls, e_y, yaw, v, beta = r["line_s"][i], r["e_y"][i] + pert[1], r["yaw"][i] + pert[2], r["v"][i] * (1 + pert[0]), r["beta"][i]
        x = np.interp(ls, tr.s, tr.xy[:, 0], period=tr.length)
        y = np.interp(ls, tr.s, tr.xy[:, 1], period=tr.length)
        psi = np.interp(ls, tr.s, np.unwrap(tr.psi), period=tr.length)
        p = np.array([x - e_y * np.sin(psi), y + e_y * np.cos(psi)])
        z = self.wz[int(np.argmin(np.linalg.norm(self.wz[:, :2] - p, axis=1))), 2] + 0.3
        return np.array([p[0], p[1], z]), np.array([0.0, 0.0, yaw]), v * np.array([np.cos(yaw + beta), np.sin(yaw + beta), 0.0])

    async def lap(self, weights, s0, cuts, pert, noise_std, seed, warm=100.0):
        """One continuous flying lap: placed once at s0 (on a straight), warm up `warm` m, then timed for one lap
        with split times at the line_s values in `cuts` (in lap order from s0 + warm). -> (splits, coll_seg, impulse,
        frac): splits[k] is None for segments the car never reached; coll_seg = segment of a collision or -1."""
        w, veh, L = self.world, self.vehicle, self.track.length
        xyz, rpy, vel = self.start_state(s0, pert)
        veh.set_transform(xyz, rpy)
        veh.set_angular_velocity(np.zeros(3))
        veh.set_linear_3d_velocity(vel)
        await w.step()
        veh.set_linear_3d_velocity(vel)
        await w.step()
        await veh.receive_observation()
        sol = submission.RoarCompetitionSolution(self.wps, NoisyAgent(veh, noise_std, seed), None, self.loc,
                                                 self.vel, self.rpy, None, self.col)
        await sol.initialize()
        self._install(sol, weights)
        prof0 = sol.profile_start_s
        p0 = np.asarray(self.loc.get_last_gym_observation(), float)
        sol.progress.update(p0, float(self.rpy.get_last_gym_observation()[2]))
        s_car = sol.progress._s_prev
        sol.progress.start_s = prof0
        sol.progress._total = (s_car - prof0) % L + L
        sol._p_prev = p0[:2]
        sol.u_prev = (float(sol.mpc.ty.steady(np.interp(s_car, self.track.s, self.track.kappa, period=L),
                                              float(np.linalg.norm(vel)))[0]), 0.0)
        tot0 = sol.progress._total
        t_start = s_car + warm
        marks = sorted(((c - t_start) % L for c in cuts)) + [L]  # each cut, then the lap close (last segment)
        splits, k, t, ticks, timing, t_seg0 = [None] * len(marks), 0, 0.0, 0, False, 0.0
        risk = [0.0] * len(marks)  # near-miss integral per segment: (slip excess / 10 deg + e_y excess / 1 m) * dt
        while True:
            await veh.receive_observation()
            done = sol.progress._total - tot0 - warm  # metres since the timing start (negative while warming up)
            imp = float(np.linalg.norm(self.col.get_last_observation().impulse_normal))
            if imp > 100.0:
                if not timing:
                    return splits, 0, imp, 0.0, risk
                seg_len = marks[k] - (marks[k - 1] if k else 0.0)
                frac = (done - (marks[k - 1] if k else 0.0)) / max(seg_len, 1.0)
                return splits, k, imp, float(np.clip(frac, 0, 1)), risk
            await sol.step()
            if self.view is not None:
                self.view.render(sol, self.info)
            await w.step()
            if done >= 0 and not timing:
                timing, ticks = True, 0
            if timing:
                ticks += 1
                t = ticks * DT
                if sol.last is not None and k < len(risk):
                    risk[k] += DT * (max(0.0, abs(np.degrees(sol.last["beta"])) - RISK_SLIP_DEG) / 10.0
                                     + max(0.0, abs(sol.last["e_y"] - sol.last["d_ref"]) - RISK_EY_M))
                while k < len(marks) and done >= marks[k]:
                    splits[k], t_seg0, k = t - t_seg0, t, k + 1
                if k >= len(marks) or done >= L:
                    return splits, -1, 0.0, 1.0, risk
            if ticks * DT > 4 * TIMEOUT_S or (ticks > 60 and np.linalg.norm(self.vel.get_last_gym_observation()) < 3.0):
                return splits, k if timing else 0, 0.0, 0.0, risk

    def _install(self, sol, weights):
        if weights is None:
            sol.residual = None
        else:
            sol.residual = submission.ResidualPolicy.__new__(submission.ResidualPolicy)
            sol.residual.W1, sol.residual.b1, sol.residual.W2, sol.residual.b2 = (np.asarray(weights[k], float)
                                                                                  for k in ("W1", "b1", "W2", "b2"))
            sol.residual.cuts = np.asarray(weights["cuts"], float) if "cuts" in weights else None

    async def episode(self, weights, s0, s1, pert=(0.0, 0.0, 0.0)):
        w, veh = self.world, self.vehicle
        xyz, rpy, vel = self.start_state(s0, pert)
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
            sol.residual.cuts = np.asarray(weights["cuts"], float) if "cuts" in weights else None
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
        off0 = (s_car - s0 + L / 2) % L - L / 2
        t, ticks = 0.0, 0
        while True:
            await veh.receive_observation()
            imp = float(np.linalg.norm(self.col.get_last_observation().impulse_normal))
            if imp > 100.0:
                return t, True, imp, (sol.progress._total - target_total - off0) / dist
            await sol.step()
            if self.view is not None:
                self.view.render(sol, self.info)
            await w.step()
            ticks += 1
            t = ticks * DT
            done = sol.progress._total - target_total - off0
            if done >= dist:
                return t, False, 0.0, 1.0
            if t > TIMEOUT_S or (t > 3 and np.linalg.norm(self.vel.get_last_gym_observation()) < 3.0):
                return TIMEOUT_S, True, 0.0, done / dist


def kill_stale():
    """A server left on this slot's port by an earlier worker (Proton re-sessions it, so it can outlive us)."""
    import signal
    me = os.getpid()
    for pid in os.listdir("/proc"):
        if pid.isdigit() and int(pid) != me:
            try:
                if ("carla-rpc-port=%d" % PORT).encode() in open("/proc/%s/cmdline" % pid, "rb").read():
                    os.kill(int(pid), signal.SIGKILL)
            except OSError:
                pass
    time.sleep(3)


async def main():
    for attempt in range(3):  # a server can accept the port but hang loading the map under load: restart it
        kill_stale()
        start_server()
        try:
            env = Env()
            break
        except RuntimeError as e:
            log("server start attempt %d failed: %r" % (attempt + 1, e))
    else:
        raise RuntimeError("no working server on %d after 3 attempts" % PORT)
    log("worker %d ready on %d" % (SLOT, PORT))
    print(json.dumps({"ready": SLOT}), flush=True)
    import select
    while True:
        ready, _, _ = select.select([sys.stdin], [], [], 0.1)
        if not ready:
            if env.view is not None:
                env.view.pump()
            continue
        line = sys.stdin.readline()
        if not line:
            break
        job = json.loads(line)
        env.info = job.get("info", "")
        if "laps" in job:  # continuous flying laps: [[s0, pert, noise_std, seed], ...]
            out = []
            for s0, pert, nstd, seed in job["laps"]:
                try:
                    sp, cs, imp, fr, rk = await env.lap(job.get("weights"), s0, job["cuts"], tuple(pert), nstd, seed)
                except Exception as e:
                    log("lap failed: %r" % e)
                    sp, cs, imp, fr, rk = [None] * (len(job["cuts"]) + 1), 0, 0.0, 0.0, [0.0] * (len(job["cuts"]) + 1)
                out.append({"splits": sp, "coll_seg": cs, "impulse": imp, "frac": fr, "risk": rk})
            print(json.dumps({"laps": out}), flush=True)
            continue
        times, coll, imp, frac = [], [], [], []
        for ep in job["episodes"]:
            s0, s1 = ep[0], ep[1]
            pert = tuple(ep[2]) if len(ep) > 2 else (0.0, 0.0, 0.0)
            try:
                t, c, i, f = await env.episode(job.get("weights"), s0, s1, pert)
            except Exception as e:  # keep the worker alive; a failed episode scores as a collision
                log("episode failed: %r" % e)
                t, c, i, f = TIMEOUT_S, True, 0.0, 0.0
            times.append(t), coll.append(bool(c)), imp.append(i), frac.append(float(f))
        print(json.dumps({"times": times, "coll": coll, "impulse": imp, "frac": frac}), flush=True)
    kill_stale()  # stdin closed: take the server down with us


if __name__ == "__main__":
    asyncio.run(main())
