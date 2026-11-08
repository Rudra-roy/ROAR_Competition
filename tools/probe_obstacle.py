"""Map an obstacle at a corner apex with the simulator's own collision geometry.

The car body is teleported through the exact poses (position + heading) a reference race drove through a window of
line_s, each pose shifted toward the inside of the corner by d = 0, step, 2 step, ... m, placed --z m above the road.
After each placement the world ticks twice (the car is still falling, so low kerbs never touch: placed at 0.05 m the
shipped car's own Lesmo poses all 'touched') and the collision sensor is read. Per pose this gives the smallest shift that
touches something: the clearance the reference race had at that point, measured with the real collision mesh (not
tools/body_margin.py's 4.8 x 2.1 m box).

  python tools/probe_obstacle.py SLOT REF.run.csv A:B [--max 0.6] [--step 0.02] [--lap 2] [--line racing_line.npz]
         [--out probe.csv]

Output csv: line_s, station (centreline), d_touch (m; inf = no touch up to --max), impulse, body_in (inner body
edge offset from the centreline at d = 0, m), body_in_touch (at d_touch). Uses its own CARLA server on port
2000 + 3 * SLOT (stop other users of the slot first). Training-time tool: the submission never teleports.
"""
import argparse
import asyncio
import os
import socket
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "competition_code"), os.path.join(ROOT, "tools")]
import carla  # noqa: E402
import roar_py_carla  # noqa: E402
from roar_py_carla.utils.convert_coordinate import location_to_carla  # noqa: E402

import body_margin  # noqa: E402
from body_margin import corner_offsets  # noqa: E402
from track import load_track  # noqa: E402

DATA = os.path.join(ROOT, "competition_code", "data")


def start_server(port, log):
    env = dict(os.environ)
    env["PATH"] = ":".join(p for p in env["PATH"].split(":") if "/.venv/" not in p)
    args = ["-RenderOffScreen", "-carla-rpc-port=%d" % port, "-carla-streaming-port=%d" % (port + 1)]
    p = subprocess.Popen([os.path.join(ROOT, "tools", "sim", "run_monza.sh")] + args, stdout=log, stderr=log, env=env,
                         start_new_session=True)
    for _ in range(300):
        try:
            socket.create_connection(("127.0.0.1", port), 1).close()
            time.sleep(5)
            return p
        except OSError:
            time.sleep(1)
    raise RuntimeError("server on %d did not come up" % port)


def poses(run_csv, track, a, b, lap):
    """(line_s, xy, yaw, kappa) of the reference race's ticks inside [a, b] on lap `lap` (1-based)."""
    d = np.genfromtxt(run_csv, delimiter=",", names=True)
    L = track.length
    m = (d["s_total"] >= (lap - 1) * L - 300) & (d["s_total"] < lap * L - 300) & (d["line_s"] >= a) & (d["line_s"] <= b)
    s = d["line_s"][m] % L
    p = np.c_[np.interp(s, track.s, track.xy[:, 0], period=L), np.interp(s, track.s, track.xy[:, 1], period=L)]
    psi = np.interp(s, track.s, np.unwrap(track.psi), period=L)
    nrm = np.c_[-np.sin(psi), np.cos(psi)]
    xy = p + d["e_y"][m][:, None] * nrm
    kap = np.interp(s, track.s, track.kappa, period=L)
    return s, xy, nrm, psi + d["e_psi"][m], kap


async def probe(port, s, xy, nrm, yaw, kap, shifts, thresh=1.0, z_up=0.3):
    client = carla.Client("127.0.0.1", port)
    client.set_timeout(120.0)
    inst = roar_py_carla.RoarPyCarlaInstance(client)
    w = inst.world
    w.set_control_steps(0.05, 0.005)
    w.set_asynchronous(False)
    wps = w.maneuverable_waypoints
    wz = np.array([q.location for q in wps])
    veh = w.spawn_vehicle("vehicle.tesla.model3", wps[0].location + np.array([0, 0, 1]), wps[0].roll_pitch_yaw, True)
    col = veh.attach_collision_sensor(np.zeros(3), np.zeros(3))
    for _ in range(10):
        await w.step()
    out = []
    for i in range(len(s)):
        inward = np.sign(kap[i]) * nrm[i] if kap[i] != 0 else nrm[i]
        z0 = wz[int(np.argmin(np.linalg.norm(wz[:, :2] - xy[i], axis=1))), 2]
        hit, imp = np.inf, 0.0
        for d in shifts:
            q = xy[i] + d * inward
            # road height under this placement (the nearest waypoint can sit tens of cm off on the Lesmo slope)
            gp = w.carla_world.ground_projection(location_to_carla(np.array([q[0], q[1], z0 + 5.0])), 20.0)
            z = (gp.location.z if gp is not None else z0) + z_up
            veh.set_transform(np.array([q[0], q[1], z]), np.array([0.0, 0.0, yaw[i]]))
            veh.set_linear_3d_velocity(np.zeros(3))
            veh.set_angular_velocity(np.zeros(3))
            await col.receive_observation()  # drop anything from the move
            await w.step()
            await w.step()
            o = await col.receive_observation()
            imp = float(np.linalg.norm(o.impulse_normal))
            # park high above the track and drain: a touch's event can arrive a tick late and would otherwise be
            # credited to the next placement (seen as isolated 0.00 touches right after a real touch)
            veh.set_transform(np.array([xy[i][0], xy[i][1], z + 60.0]), np.array([0.0, 0.0, yaw[i]]))
            veh.set_linear_3d_velocity(np.zeros(3))
            for _ in range(3):
                await w.step()
                await col.receive_observation()
            if imp > thresh:
                hit = d
                break
        out.append((hit, imp))
        print("line_s %7.1f  d_touch %s  impulse %.0f" % (s[i], "%.2f" % hit if np.isfinite(hit) else " none", imp),
              flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("slot", type=int)
    ap.add_argument("ref")
    ap.add_argument("window")
    ap.add_argument("--max", type=float, default=0.6)
    ap.add_argument("--step", type=float, default=0.02)
    ap.add_argument("--lap", type=int, default=2)
    ap.add_argument("--line", default=os.path.join(DATA, "racing_line.npz"))
    ap.add_argument("--out", default="probe.csv")
    ap.add_argument("--z", type=float, default=0.3, help="placement height above the road, m: high enough that "
                    "low kerbs do not touch in the two ticks before the car lands, so only tall geometry counts")
    a = ap.parse_args(argv)
    port = 2000 + 3 * a.slot
    track = load_track(a.line)
    lo, hi = map(float, a.window.split(":"))
    s, xy, nrm, yaw, kap = poses(a.ref, track, lo, hi, a.lap)
    shifts = np.round(np.arange(0.0, a.max + 1e-9, a.step), 4)
    log = open(os.path.splitext(a.out)[0] + ".server.log", "w")
    srv = start_server(port, log)
    try:
        res = asyncio.run(probe(port, s, xy, nrm, yaw, kap, shifts, z_up=a.z))
    finally:
        os.killpg(srv.pid, 9)
        for p in subprocess.run(["pgrep", "-f", "carla-rpc-port=%d" % port], capture_output=True, text=True).stdout.split():
            subprocess.run(["kill", "-9", p])
    centre = np.load(os.path.join(DATA, "centreline.npy"))
    body_margin.UPSAMPLE = 1  # one offset per pose
    st, b_lo, b_hi = corner_offsets(xy, yaw, centre)
    inward_sign = np.sign(kap)  # + : the inside is to the left (+ offset)
    rows = []
    for i, (d, imp) in enumerate(res):
        edge = b_lo[i] if inward_sign[i] < 0 else b_hi[i]
        if np.isfinite(d):
            q = xy[i] + d * np.sign(kap[i]) * nrm[i]
            st2, lo2, hi2 = corner_offsets(q[None], np.array([yaw[i]]), centre)
            edge_t = lo2[0] if inward_sign[i] < 0 else hi2[0]
        else:
            edge_t = np.nan
        rows.append((s[i], st[i], d, imp, edge, edge_t))
    np.savetxt(a.out, np.array(rows), delimiter=",", header="line_s,station,d_touch,impulse,body_in,body_in_touch",
               comments="", fmt="%.4f")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
