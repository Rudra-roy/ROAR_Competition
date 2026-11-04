"""Per-corner comparison of raced laps (Task 8g): where and how does one car gain on another?

  python tools/corner_compare.py NAME=log.csv [NAME=log.csv ...]           per-corner table
  python tools/corner_compare.py --grip NAME=log.csv [...]                   p95 |a_lat| per a_lat section

A log is a HUD csv with x, y columns (logs/hud.csv from ROAR_HUD_LOG=1, any controller) or a ROAR_LOG run.csv (our
controller; the car xy is rebuilt from line_s + e_y on the current racing line). Position = distance along
centreline.npy, unwrapped from the first sample. Laps 2 and 3 (flying) only; each cell is the median of those laps.

Sectors: one per corner, from the plan's top-speed point before it to the one after it. Every position is in centreline metres mod lap. Per sector:
  time       sector time (s)
  lift       first point (before the min-speed point) with throttle < 0.5 or brake > 0: the entry lift/brake point
  decel      peak deceleration, -dv/dt over a 5-tick mean (m/s^2)
  vmin       minimum speed (km/h) @ its position
  alat       peak |v^2 kappa| on the car's own path (course rate * v, 5-tick mean) (m/s^2)
  exit       start of the last full-throttle (>= 0.95) run of the sector, i.e. where the car commits to the exit
  plan       our plan's minimum in the sector at this car's positions (HUD plan / run.csv v_ref column), km/h
  d_apex     the car's signed offset from the centreline at the min-speed point (m, + left)
"""
import os
import sys

import numpy as np

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(R, "competition_code"), os.path.join(R, "tools")]
DATA = os.path.join(R, "competition_code", "data")
DT = 0.05
LAPS = (1, 2)  # 0-based: the flying laps 2 and 3
# corner apex ranges in line_s of the current racing line (copy of corner_diag.CORNERS; that module runs on import)
CORNERS = [(795, 937), (1281, 1329), (1619, 1697), (2545, 2695), (3719, 3835), (3935, 3993), (4093, 4113),
           (5239, 5413)]


def _mean5(a):
    return np.convolve(a, np.ones(5) / 5, "same")


def path_a_lat(xy, v):
    """Signed lateral acceleration on the driven path: v * d(course)/dt, + left. 5-tick means."""
    d = np.gradient(xy, axis=0)
    chi = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    return _mean5(v * np.gradient(chi) / DT)


def corner_metrics(D, a, b):
    """Metrics of one pass through the sector [a, b) (D arrays restricted to it are taken here)."""
    m = (D["s"] >= a) & (D["s"] < b)
    s, v, thr, brk = D["s"][m], D["v"][m], D["thr"][m], D["brk"][m]
    k = int(np.argmin(v))
    lift = np.flatnonzero((thr[:k + 1] < 0.5) | (brk[:k + 1] > 0))
    ex = np.flatnonzero(thr[k:] < 0.95)  # exit = start of the full-throttle run that ends the sector
    dec = -_mean5(np.gradient(v) / DT)
    plan = D["plan"][m]
    return dict(time=float(np.ptp(D["t"][m]) + DT), lift_s=float(s[lift[0]]) if len(lift) else np.nan,
                decel_max=float(dec[2:-2].max()) if len(v) > 5 else np.nan, v_min=float(v[k]), v_min_s=float(s[k]),
                a_lat_max=float(np.abs(D["a_lat"][m]).max()), exit_thr_s=float(s[min(k + ex[-1] + 1, len(s) - 1)]) if len(ex) else float(s[k]),
                plan_min=float(np.nanmin(plan)) if np.isfinite(plan).any() else np.nan, k=int(np.flatnonzero(m)[k]))


def load(path):
    """-> dict of arrays t, s (centreline, unwrapped), v, thr, brk, xy, a_lat, plan."""
    from hud import LapTracker, Polyline
    d = np.genfromtxt(path, delimiter=",", names=True)
    n = d.dtype.names
    if "x" in n:
        xy, t, v, thr, brk = np.c_[d["x"], d["y"]], d["t"], d["v"], d["thr"], d["brk"]
        plan = d["plan"]
    else:  # our run.csv
        from track import Track
        tr = Track(np.load(os.path.join(DATA, "racing_line.npz"))["xy"])
        I = lambda a: np.interp(d["line_s"], tr.s, a, period=tr.length)  # noqa: E731
        psi = I(np.unwrap(tr.psi))
        xy = np.c_[I(tr.xy[:, 0]), I(tr.xy[:, 1])] + d["e_y"][:, None] * np.c_[-np.sin(psi), np.cos(psi)]
        t, v, thr, brk, plan = d["tick"] * DT, d["v"], d["throttle"], d["brake"], d["v_ref"]
    lt = LapTracker(Polyline(np.load(os.path.join(DATA, "centreline.npy"))))
    s = np.array([lt.update(p)[0] for p in xy])
    s0 = lt.line.project(xy[0])[1]
    s = s + (s0 if s0 < lt.line.length / 2 else s0 - lt.line.length)
    return dict(t=t, s=s, v=v, thr=thr, brk=brk, xy=xy, a_lat=path_a_lat(xy, v), plan=plan, L=lt.line.length)


def sectors():
    """[(name, a, b)] in centreline metres, a in [0, L), b > a, contiguous over one lap. Boundaries: the plan's
    (lap 2 of speed_profile.npz) top-speed point between consecutive corners, i.e. just before each braking zone."""
    from hud import Polyline
    from track import Track
    tr = Track(np.load(os.path.join(DATA, "racing_line.npz"))["xy"])
    cl = Polyline(np.load(os.path.join(DATA, "centreline.npy")))
    pr = np.load(os.path.join(DATA, "speed_profile.npz"))
    lap2 = (pr["s"] >= tr.length) & (pr["s"] < 2 * tr.length)
    ls, vp = (pr["start_s"] + pr["s"][lap2]) % tr.length, pr["v"][lap2]
    bounds = []
    for i, (_, b0) in enumerate(CORNERS):
        a1 = CORNERS[(i + 1) % len(CORNERS)][0]
        m = ((ls - b0) % tr.length) < ((a1 - b0) % tr.length)
        bounds.append(ls[m][np.argmax(vp[m])])
    c = [cl.project(tr.xy[int(round(x / tr.ds)) % len(tr.s)])[1] for x in bounds]  # end of corner i
    out = []
    for i, (_, b) in enumerate(CORNERS):
        st, en = c[i - 1], c[i]
        out.append(("c%d" % b, st, en if en > st else en + cl.length))
    return sorted(out, key=lambda r: r[1])


def compare(logs, secs=None):
    """logs: [(name, D)] -> {(sector name, log name): metrics (median over LAPS)}; metric 'd_apex' added."""
    from export_line import offsets
    centre = np.load(os.path.join(DATA, "centreline.npy"))
    secs = secs or sectors()
    res = {}
    for name, D in logs:
        L = D["L"]
        for sn, a, b in secs:
            ms = [corner_metrics(D, a + n * L, b + n * L) for n in LAPS if D["s"].max() >= b + n * L]
            if not ms:
                continue
            for m in ms:
                m["d_apex"] = float(offsets(D["xy"][m["k"]:m["k"] + 1], centre)[0][0])
                for k in ("lift_s", "v_min_s", "exit_thr_s"):
                    m[k] = m[k] % L
            res[sn, name] = {k: float(np.nanmedian([m[k] for m in ms])) for k in ms[0] if k != "k"}
    return res, secs


def section_grip(D, starts_c, p=95.0, v_min=15.0):
    """p-th percentile of |a_lat| (flying laps, v > v_min) per section; starts_c = section starts in centreline m,
    ascending mod lap (the last section wraps into the first)."""
    L, s = D["L"], D["s"]
    fly = (s >= LAPS[0] * L) & (s < (LAPS[-1] + 1) * L) & (D["v"] > v_min)
    st = np.asarray(starts_c) % L
    sec = (np.searchsorted(st, s % L, side="right") - 1) % len(st)
    return [float(np.percentile(np.abs(D["a_lat"][fly & (sec == i)]), p)) for i in range(len(st))]


def grip_table(logs, p=95.0):
    """a_lat sections of vehicle_params.json vs the p-th percentile |a_lat| each log reached in them."""
    import json
    from hud import Polyline
    from track import Track
    tr = Track(np.load(os.path.join(DATA, "racing_line.npz"))["xy"])
    cl = Polyline(np.load(os.path.join(DATA, "centreline.npy")))
    secs = json.load(open(os.path.join(DATA, "vehicle_params.json")))["a_lat_sections"]
    st = [cl.project(tr.xy[int(round(a / tr.ds)) % len(tr.s)])[1] for a, _ in secs]
    order = np.argsort(st)
    g = {name: section_grip(D, np.asarray(st)[order], p) for name, D in logs}
    print("| sec | line_s | centreline | a_lat (json) | " + " | ".join("%s p%g" % (n, p) for n, _ in logs) + " |")
    print("|" + "---|" * (4 + len(logs)))
    for j, i in enumerate(order):
        print("| %d | %.1f | %.0f | %.3f | " % (i, secs[i][0], st[i], secs[i][1])
              + " | ".join("%.1f" % g[n][j] for n, _ in logs) + " |")
    return g


COLS = (("time", "%.2f", 1), ("lift_s", "%.0f", 1), ("decel_max", "%.1f", 1), ("v_min", "%.0f", 3.6),
        ("v_min_s", "%.0f", 1), ("a_lat_max", "%.1f", 1), ("exit_thr_s", "%.0f", 1), ("plan_min", "%.0f", 3.6),
        ("d_apex", "%+.2f", 1))


def main(argv):
    grip = "--grip" in argv
    logs = [(a.split("=", 1)[0], load(a.split("=", 1)[1])) for a in argv if not a.startswith("--")]
    if grip:
        return grip_table(logs)
    res, secs = compare(logs)
    print("| sector (centreline m) | car | time s | lift @ | decel max | vmin km/h | @ | alat max | exit thr @ "
          "| plan min km/h | d_apex m |")
    print("|" + "---|" * 11)
    for sn, a, b in secs:
        for name, _ in logs:
            m = res.get((sn, name))
            if m:
                print("| %s %.0f-%.0f | %s | " % (sn, a, b % logs[0][1]["L"], name)
                      + " | ".join(f % (m[k] * sc) for k, f, sc in COLS) + " |")
    tot = {name: sum(res[sn, name]["time"] for sn, _, _ in secs if (sn, name) in res) for name, _ in logs}
    print("lap time (median of flying laps, sum of sectors): " + ", ".join("%s %.2f" % kv for kv in tot.items()))


if __name__ == "__main__":
    main(sys.argv[1:])
