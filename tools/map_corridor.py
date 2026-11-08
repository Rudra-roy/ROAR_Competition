"""Map the walls on both sides of the whole lap with CARLA ray casts (the track's real corridor).

At every --step m of the centreline, horizontal rays go from the centreline outward along its left and right normals
at heights 0.3, 0.7 and 1.2 m above the ground-projected road; the first hit that is not road surface (Roads,
RoadLines) is a wall. Per side the nearest of the three heights is kept. Checked against tools/probe_obstacle.py (the
car's collision mesh): Lesmo 1 fence at 5.37-5.55 m (0.7 m ray) vs the probed body edge 5.33-5.53 m; final hairpin
within ~0.1 m. The 0.3 m ray hits a fence's sloped foot ~0.35 m nearer; the car body touched where the 0.7 m ray hits,
so --heights default 0.7,1.2 (0.3 m is kept in the file as wall_low).

  python tools/map_corridor.py OUT.npz [--slot 1] [--step 1.0] [--reach 12]

OUT.npz: station (centreline arc length, m), left, right (wall distance from the centreline, m, > 0; nan = none
within --reach), left_low, right_low (the 0.3 m ray), label_left, label_right.
"""
import argparse
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "competition_code"), os.path.join(ROOT, "tools")]
DATA = os.path.join(ROOT, "competition_code", "data")
ROAD = ("Roads", "RoadLines")


def first_wall(world, p, n, z, h, reach, carla_mod, to_carla, from_carla):
    """(distance, label) of the first non-road hit along p + t n (t in (0, reach]) at height z + h."""
    a = to_carla(np.array([p[0], p[1], z + h]))
    b = to_carla(np.r_[p + reach * n, z + h])
    for hp in world.cast_ray(a, b):
        lab = str(hp.label)
        if lab in ROAD:
            continue
        d = float(np.dot(from_carla(hp.location)[:2] - p, n))
        if d > 0.05:
            return d, lab
    return np.nan, ""


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--slot", type=int, default=1)
    ap.add_argument("--step", type=float, default=1.0)
    ap.add_argument("--reach", type=float, default=12.0)
    ap.add_argument("--heights", default="0.7,1.2")
    a = ap.parse_args(argv)
    import carla
    from roar_py_carla.utils.convert_coordinate import location_from_carla, location_to_carla
    import probe_obstacle as po
    port = 2000 + 3 * a.slot
    centre = np.load(os.path.join(DATA, "centreline.npy"))
    seg = np.linalg.norm(np.roll(centre, -1, 0) - centre, axis=1)
    st0 = np.r_[0, np.cumsum(seg)[:-1]]
    L = seg.sum()
    stations = np.arange(0.0, L, a.step)
    heights = [float(x) for x in a.heights.split(",")]
    srv = po.start_server(port, open(os.path.splitext(a.out)[0] + ".server.log", "w"))
    try:
        cl = carla.Client("127.0.0.1", port)
        cl.set_timeout(120.0)
        w = cl.get_world()
        res = {k: [] for k in ("left", "right", "left_low", "right_low", "label_left", "label_right")}
        for k, s in enumerate(stations):
            i = int(np.searchsorted(st0, s, side="right") - 1) % len(centre)
            j = (i + 1) % len(centre)
            t = centre[j] - centre[i]
            t /= np.linalg.norm(t)
            p = centre[i] + (s - st0[i]) * t
            nl = np.array([-t[1], t[0]])
            gp = w.ground_projection(location_to_carla(np.array([p[0], p[1], 80.0])), 200.0)
            z = gp.location.z if gp is not None else 0.0
            for side, n in (("left", nl), ("right", -nl)):
                hits = [first_wall(w, p, n, z, h, a.reach, carla, location_to_carla, location_from_carla) for h in heights]
                d = [x[0] for x in hits]
                m = int(np.nanargmin(d)) if np.isfinite(d).any() else 0
                res[side].append(d[m])
                res["label_" + side].append(hits[m][1])
                res[side + "_low"].append(first_wall(w, p, n, z, 0.3, a.reach, carla, location_to_carla,
                                                     location_from_carla)[0])
            if k % 500 == 0:
                print("station %6.0f / %.0f  left %.2f right %.2f" % (s, L, res["left"][-1], res["right"][-1]),
                      flush=True)
    finally:
        os.killpg(srv.pid, 9)
        for q in subprocess.run(["pgrep", "-f", "carla-rpc-port=%d" % port], capture_output=True, text=True).stdout.split():
            subprocess.run(["kill", "-9", q])
    np.savez(a.out, station=stations, **{k: np.array(v) for k, v in res.items()})
    print("wrote", a.out, "stations", len(stations))


if __name__ == "__main__":
    main()
