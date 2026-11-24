"""Pick, from a tools/capture.py folder, the frame where the car is nearest a corner's apex on lap 2. Training-time tool.

  python tools/pick_stills.py CAPTURE_DIR [--view chase] [--lap 2]

frames.csv holds CARLA-frame positions (y is the ROAR y negated); the racing line is in the ROAR frame. Prints, per
corner, the file name, simulated time, speed and the distance from the apex.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from media_data import DATA  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "competition_code"))
from track import load_track  # noqa: E402

APEX = {"first_chicane": 982.0, "c1329": 1339.0, "lesmo1": 1692.0, "kink": 2698.0, "final_hairpin": 5408.0}


def pick(capdir, view="chase", lap=2):
    tr = load_track(os.path.join(DATA, "racing_line.npz"))
    fr = np.genfromtxt(os.path.join(capdir, "frames.csv"), delimiter=",", names=True, dtype=None, encoding=None)
    fr = fr[fr["view"] == view]
    xy = np.c_[fr["x"], -fr["y"]]  # CARLA -> ROAR frame
    s = np.array([tr.s[np.argmin(np.linalg.norm(tr.xy - p, axis=1))] for p in xy])
    out = {}
    for name, a in APEX.items():
        near = np.where(np.abs(s - a) < 40)[0]
        if not len(near):
            continue
        groups = np.split(near, np.where(np.diff(fr["t"][near]) > 20)[0] + 1)  # one group per lap
        g = groups[min(lap - 1, len(groups) - 1)] if len(groups) >= lap else groups[-1]
        k = g[np.argmin(np.abs(s[g] - a))]
        out[name] = (view + "_%07.2f.png" % fr["t"][k], float(fr["t"][k]), float(fr["speed"][k]), float(s[k] - a), len(groups))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("capdir")
    ap.add_argument("--view", default="chase")
    ap.add_argument("--lap", type=int, default=2)
    a = ap.parse_args(argv)
    for name, (f, t, v, ds, n) in pick(a.capdir, a.view, a.lap).items():
        print("%-14s %s  t=%.1f s  v=%.1f m/s  %+.1f m from the apex  (%d passes seen)" % (name, f, t, v, ds, n))


if __name__ == "__main__":
    main()
