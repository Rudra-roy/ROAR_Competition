"""High-speed weave metrics from ROAR_LOG run.csv files (Task 8e).

  python tools/weave_diag.py logs/<label>_*.run.csv

Straights: v > 55 m/s and |kappa_ref| < 0.002. Per file: ticks, steer zero-crossings per second (counted inside
contiguous straight stretches), e_y std / max|e_y|, max |delta steer| per tick.
"""
import sys

import numpy as np

DT = 0.05


def weave(path, v_min=55.0, kref_max=0.002):
    d = np.genfromtxt(path, delimiter=",", names=True)
    m = (d["v"] > v_min) & (np.abs(d["kappa_ref"]) < kref_max)
    idx = np.flatnonzero(m)
    segs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
    st = d["steer"]
    zc = sum(int(np.sum(np.sign(st[s][1:]) != np.sign(st[s][:-1]))) for s in segs)
    both = m[1:] & m[:-1]
    return dict(n=int(m.sum()), zc=zc / (m.sum() * DT), ey_std=float(d["e_y"][m].std()),
                ey_max=float(np.abs(d["e_y"][m]).max()), dsteer=float(np.abs(np.diff(st))[both].max()))


if __name__ == "__main__":
    for p in sys.argv[1:]:
        w = weave(p)
        print("%-28s n %4d  steer zc %.2f /s  e_y std %.3f max %.2f  max|dsteer| %.3f" % (
            p.split("/")[-1], w["n"], w["zc"], w["ey_std"], w["ey_max"], w["dsteer"]))
