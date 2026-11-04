"""Wall margin of raced laps from ROAR_LOG run.csv files (Task 8f).

  python tools/wall_margin.py [--line=racing_line.npz] logs/<label>_*.run.csv

The car centre is rebuilt from line_s + e_y on the racing line the run used (--line, default the current
competition_code/data/racing_line.npz; e.g. `git show 3dfa9a2:competition_code/data/racing_line.npz` for 8d logs) and
measured signed against centreline.npy (point-to-polyline, + left). Per run:
  chicane  max right offset (-d) within NEAR m of the chicane right wall where mc8d hit 11 times at d -3.96..-4.00
  lap max  max |d| and its line_s
  corners  max |d| per corner window (line_s from apex_s0 - 20 to apex_s1 + 80, the corner_diag.py CORNERS)
"""
import os
import sys

import numpy as np

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(R, "competition_code"), os.path.join(R, "tools")]
from export_line import offsets  # noqa: E402
from track import Track  # noqa: E402

DATA = os.path.join(R, "competition_code", "data")
CHICANE_WALL = np.array([[267.2, 906.9], [274.5, 907.6]])  # mc8d impact xy span (8d line_s 864-871)
NEAR = 8.0
CORNERS = [(795, 937), (1281, 1329), (1619, 1697), (2545, 2695), (3719, 3835), (3935, 3993), (4093, 4113),
           (5239, 5413)]  # copy of corner_diag.CORNERS (that module runs on import)


def seg_dist(p, ab):
    a, b = ab
    t = np.clip((p - a) @ (b - a) / ((b - a) @ (b - a)), 0, 1)
    return np.linalg.norm(p - (a + t[:, None] * (b - a)), axis=1)


def margins(run_csv, line_xy, centre, wall=CHICANE_WALL, corners=CORNERS):
    d = np.genfromtxt(run_csv, delimiter=",", names=True)
    tr = Track(line_xy)
    ls = d["line_s"]
    I = lambda a: np.interp(ls, tr.s, a, period=tr.length)  # noqa: E731
    psi = I(np.unwrap(tr.psi))
    xy = np.c_[I(tr.xy[:, 0]), I(tr.xy[:, 1])] + d["e_y"][:, None] * np.c_[-np.sin(psi), np.cos(psi)]
    dc = offsets(xy, centre)[0]
    near = seg_dist(xy, wall) <= NEAR
    k = int(np.argmax(np.abs(dc)))
    win = lambda a, b: np.abs(dc[(ls >= a) & (ls <= b)]).max()  # noqa: E731
    return dict(chicane_right=float(-dc[near].min()) if near.any() else np.nan,
                lap_max=float(abs(dc[k])), lap_max_at=float(ls[k]),
                corners=[float(win(a - 20, b + 80)) for a, b in corners])


if __name__ == "__main__":
    line = os.path.join(DATA, "racing_line.npz")
    files = []
    for a in sys.argv[1:]:
        if a.startswith("--line="):
            line = a[7:]
        else:
            files.append(a)
    lxy, centre = np.load(line)["xy"], np.load(os.path.join(DATA, "centreline.npy"))
    print("| run | chicane right | lap max\\|d\\| @ line_s | " + " | ".join("c%d" % b for _, b in CORNERS) + " |")
    print("|" + "---|" * (3 + len(CORNERS)))
    for f in files:
        m = margins(f, lxy, centre)
        print("| %s | %.2f | %.2f @ %.0f | %s |" % (os.path.basename(f), m["chicane_right"], m["lap_max"],
                                                  m["lap_max_at"], " | ".join("%.2f" % c for c in m["corners"])))
