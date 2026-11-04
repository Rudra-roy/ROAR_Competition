"""Proven corridor envelope (Task 8h): where a car centre has been driven without hitting anything.

  python tools/envelope.py [--line=OLD.npz] [car.hud.csv ...]   per run: max |d| and excursion beyond the car bounds

Per centreline station (the vertices of centreline.npy, ~2 m): offsets d (+ left, point-to-polyline) of our current
racing line and of the S25 car over every pass in its HUD log (x, y in the ROAR frame).
  raw  lo_raw / hi_raw = min / max of {our line, every S25 pass}: the interval between two driven offsets has been
       traversed by a car centre, so it is treated as clear.
  line lo / hi: on every side where S25 (not our line) is the extreme, its bound moved SHRINK m toward the
       centreline, not past it (our car tracks worse than S25); a bound on the centre side of zero is left as driven
       (moving it toward the centreline would widen it). Never past our own line, and |d| <= CAP.
The line generator uses lo / hi. The car check of a raced run (check_bounds) is never further out than our current
line, S25 or our current car (the e115 gate runs, E115_RUNS) went, with 0.3 m allowed on the side facing the
centreline. The S25 log is telemetry data only (no code from ../ref or the S25 sandbox is used).
"""
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(R, "competition_code", "data")
S25_HUD = os.path.join(os.path.dirname(R), "sandbox_s25", "logs", "s25_xy.hud.csv")
SHRINK = 0.3
CAP = 4.0
LINE_8E = os.path.join(R, "tools", "line_e115.npz")  # the line the e115 runs drove (racing_line.npz at 8d1f3a4)
E115_RUNS = [os.path.join(R, "logs", "final8g_%d.hud.csv" % i) for i in range(1, 6)]


def station(p, centre):
    """-> (s, d): arc length of the foot point along the closed polyline `centre` and the signed offset (+ left)."""
    p = np.asarray(p, float)[:, :2]
    n = len(centre)
    seg = np.roll(centre, -1, 0) - centre
    el = np.linalg.norm(seg, axis=1)
    s0 = np.r_[0.0, np.cumsum(el)[:-1]]
    _, i = cKDTree(centre).query(p)
    best_d, best_s = np.full(len(p), np.inf), np.zeros(len(p))
    for a in ((i - 1) % n, i):
        r = p - centre[a]
        u = np.clip((r * seg[a]).sum(1) / el[a] ** 2, 0, 1)
        dist = np.linalg.norm(r - u[:, None] * seg[a], axis=1)
        sgn = np.sign(seg[a, 0] * r[:, 1] - seg[a, 1] * r[:, 0])
        take = dist < np.abs(best_d)
        best_d = np.where(take, sgn * dist, best_d)
        best_s = np.where(take, s0[a] + u * el[a], best_s)
    return best_s % el.sum(), best_d


def passes(xy, centre, st):
    """Offsets of a driven trace at stations st, one row per (partial) pass around the loop; NaN where not covered."""
    L = np.linalg.norm(np.roll(centre, -1, 0) - centre, axis=1).sum()
    s, d = station(xy, centre)
    su = np.maximum.accumulate(np.unwrap(s * 2 * np.pi / L) * L / (2 * np.pi))  # forward progress only
    rows = []
    for k in range(int(np.floor(su[0] / L)), int(np.floor(su[-1] / L)) + 1):
        q = st + k * L
        rows.append(np.where((q >= su[0]) & (q <= su[-1]), np.interp(q, su, d), np.nan))
    return np.array(rows)


def envelope(line_xy, s25_xy, centre, shrink=SHRINK, cap=CAP):
    """-> stations, lo, hi (line bounds), lo_raw, hi_raw (see module doc)."""
    seg = np.linalg.norm(np.roll(centre, -1, 0) - centre, axis=1)
    st, L = np.r_[0.0, np.cumsum(seg)[:-1]], seg.sum()
    s, d = station(line_xy, centre)
    o = np.argsort(s)
    ours = np.interp(st, s[o], d[o], period=L)
    car = passes(s25_xy, centre, st)
    s_lo, s_hi = np.fmin.reduce(car, 0), np.fmax.reduce(car, 0)  # NaN = no S25 pass covers the station: ours only
    lo_raw, hi_raw = np.fmin(ours, s_lo), np.fmax(ours, s_hi)
    lo = np.maximum(np.fmin(ours, s_lo + np.clip(-s_lo, 0, shrink)), -cap)
    hi = np.minimum(np.fmax(ours, s_hi - np.clip(s_hi, 0, shrink)), cap)
    return st, lo, hi, lo_raw, hi_raw


def check_bounds(line_xy, traces, centre, widen=SHRINK):
    """Car rule bounds: per station min / max over our line and every pass of every driven trace (S25, our e115 gate
    runs), the side facing the centreline (lo > 0 / hi < 0) widened by `widen` m toward it -> stations, lo, hi."""
    seg = np.linalg.norm(np.roll(centre, -1, 0) - centre, axis=1)
    st, L = np.r_[0.0, np.cumsum(seg)[:-1]], seg.sum()
    s, d = station(line_xy, centre)
    o = np.argsort(s)
    rows = np.vstack([np.interp(st, s[o], d[o], period=L)] + [passes(x, centre, st) for x in traces])
    lo, hi = np.fmin.reduce(rows, 0), np.fmax.reduce(rows, 0)
    return st, lo - widen * (lo > 0), hi + widen * (hi < 0)


def outside(car_xy, st, lo, hi, centre):
    """Per car point: metres beyond [lo, hi] at its station (0 inside)."""
    L = np.linalg.norm(np.roll(centre, -1, 0) - centre, axis=1).sum()
    s, d = station(car_xy, centre)
    a, b = np.interp(s, st, lo, period=L), np.interp(s, st, hi, period=L)
    return np.maximum(0.0, np.maximum(d - b, a - d))


def outward(car_xy, st, lo, hi, centre):
    """Per car point: metres further from the centreline than the bounds reach on its side (0 when the car is inside
    them or nearer the centreline). The binding part of the car rule: never further out than anyone drove."""
    L = np.linalg.norm(np.roll(centre, -1, 0) - centre, axis=1).sum()
    s, d = station(car_xy, centre)
    a, b = np.interp(s, st, lo, period=L), np.interp(s, st, hi, period=L)
    return np.where(d > 0, np.maximum(0.0, d - np.maximum(b, 0)), np.maximum(0.0, np.minimum(a, 0) - d))


def load_xy(hud_csv):
    d = np.genfromtxt(hud_csv, delimiter=",", names=True)
    return np.c_[d["x"], d["y"]]


def main(argv):
    line = LINE_8E
    files = []
    for a in argv:
        if a.startswith("--line="):
            line = a[7:]
        else:
            files.append(a)
    centre = np.load(os.path.join(DATA, "centreline.npy"))
    st, lo, hi = check_bounds(np.load(line)["xy"], [load_xy(S25_HUD)] + [load_xy(f) for f in E115_RUNS], centre)
    print("| run | max beyond car bounds m @ centreline | ticks > 0.05 m | max further out m @ centreline "
          "| max abs(d) @ centreline |")
    print("|---|---|---|---|---|")
    for f in files:
        xy = load_xy(f)
        ex = outside(xy, st, lo, hi, centre)
        s, d = station(xy, centre)
        out = outward(xy, st, lo, hi, centre)
        k, j, o = int(np.argmax(ex)), int(np.argmax(np.abs(d))), int(np.argmax(out))
        print("| %s | %.2f @ %.0f | %d | %.2f @ %.0f | %.2f @ %.0f |" % (
            os.path.basename(f), ex[k], s[k], int((ex > 0.05).sum()), out[o], s[o], abs(d[j]), s[j]))


if __name__ == "__main__":
    main(sys.argv[1:])
