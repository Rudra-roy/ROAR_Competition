"""Correct the ray-cast wall map (tools/map_corridor.py) with collision-probe edges (tools/probe_obstacle.py).

The 0.3 / 0.7 / 1.2 m rays miss fence feet that stick out lower, by 0.1-0.25 m at every spot probed so far (Lesmo 1,
final hairpin, c1329, first-chicane exit). A probe of a CLEAN race's poses gives, per pose, the body-edge offset at
the first touch (body_in_touch, from the centreline; the inside of the corner): that is the real wall there. The map's
wall distance at the pose's centreline station (+-1 m) becomes the nearer of the old value and the probed edge.

  python tools/correct_corridor.py tools/corridor_map.npz logs/probe/kink_ship_gb4.csv [more.csv ...]

Idempotent (only ever moves a wall nearer). Rows with d_touch = 0.00 are skipped: on a clean race's own pose that is a
touch event credited to the wrong pose, not an obstacle (the probe drains the sensor between poses now, but older
probe files have them). Only probe CLEAN races: a crash replay touches at ~0 by construction.
"""
import sys

import numpy as np


def correct(z, probe_csvs, min_d=0.01):
    """z: dict of the map arrays (station, left, right, ...). -> (corrected copy, number of station-sides moved)."""
    z = {k: np.array(v, copy=True) for k, v in z.items()}
    st = z["station"]
    step = st[1] - st[0]
    n = len(st)
    moved = 0
    for f in probe_csvs:
        a = np.genfromtxt(f, delimiter=",", names=True)
        a = a[np.isfinite(a["d_touch"]) & (a["d_touch"] > min_d)]
        for r in np.atleast_1d(a):
            k = int(round(r["station"] / step)) % n
            e = float(r["body_in_touch"])
            for kk in ((k - 1) % n, k, (k + 1) % n):
                if e < 0:  # inside of a right-hander: the wall is on the right, at distance -e
                    old = z["right"][kk] if np.isfinite(z["right"][kk]) else np.inf
                    if -e < old:
                        z["right"][kk], moved = -e, moved + 1
                elif e < z["left"][kk]:
                    z["left"][kk], moved = e, moved + 1
    z["probe_corrected"] = np.array(True)
    return z, moved


def main(argv):
    path, probes = argv[0], argv[1:]
    z, moved = correct(dict(np.load(path)), probes)
    np.savez(path, **z)
    print("moved %d station-sides in %s from %d probe file(s)" % (moved, path, len(probes)))


if __name__ == "__main__":
    main(sys.argv[1:])
