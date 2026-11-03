"""Fit the steady-state pedal map a_ss(v, pedal) from per-tick logs; write it into vehicle_params.json.

  python tools/fit_pedals.py [--write] [csv ...]    (default: logs/*.run.csv + logs/sysid*.csv)

Plant model (dynamic, so transient ticks count too): the measured accel a[k] = (v[k+1] - v[k]) / dt is a
first-order lag (time constant tau) of a_ss(v, u) applied DELAY ticks late, u = throttle - brake (exclusive
pedals). a_ss is bilinear on (V_GRID x U_GRID), so a[k] = sum_j h_j * basis(v[k-j], u[k-DELAY-j]) . table is
linear in the table: delay and tau by grid search, table by ridge-regularised least squares (2nd differences,
so cells without data extrapolate smoothly). Only straight, teleport-free windows are used (|kappa| < 0.004).
--write stores "pedal_map" and refits "a_drive_lines" (fit_sysid.fit_a_drive on steady full-throttle straight
samples: >= 9 ticks of throttle 1) in vehicle_params.json.
"""
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARAMS = os.path.join(ROOT, "competition_code", "data", "vehicle_params.json")
DT = 0.05
V_GRID = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
BRK_GRID = [0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
THR_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
U_GRID = [-b for b in BRK_GRID[:0:-1]] + THR_GRID
TAPS = 16
KAPPA_STRAIGHT = 0.004
LAM = 3.0  # smoothness weight (per 2nd difference, in m/s^2 units against ~4e4 samples)


def load(path):
    d = np.genfromtxt(path, delimiter=",", names=True)
    v = np.asarray(d["v"], float)
    u = np.asarray(d["throttle"], float) - np.asarray(d["brake"], float)
    if "kappa_ref" in d.dtype.names:  # controller log
        kap = np.asarray(d["kappa_ref"], float)
        jump = np.r_[np.abs(np.diff(d["s_total"])) > 20, False]
    else:  # sysid log
        kap = np.asarray(d["yaw_rate"], float) / np.maximum(v, 1.0)
        jump = np.r_[np.hypot(np.diff(d["x"]), np.diff(d["y"])) > 6, False]
    a = np.r_[np.diff(v) / DT, np.nan]
    ok = (np.abs(kap) < KAPPA_STRAIGHT) & ~jump & np.isfinite(a)
    return v, u, a, ok


def basis(v, u):
    """Bilinear weights of (v, u) on the grid, flattened index i_v * len(U) + i_u."""
    vg, ug = np.asarray(V_GRID), np.asarray(U_GRID)
    iv = np.clip(np.searchsorted(vg, v) - 1, 0, len(vg) - 2)
    iu = np.clip(np.searchsorted(ug, u) - 1, 0, len(ug) - 2)
    fv = np.clip((v - vg[iv]) / (vg[iv + 1] - vg[iv]), 0, 1)
    fu = np.clip((u - ug[iu]) / (ug[iu + 1] - ug[iu]), 0, 1)
    nu = len(ug)
    idx = np.stack([iv * nu + iu, iv * nu + iu + 1, (iv + 1) * nu + iu, (iv + 1) * nu + iu + 1], 1)
    w = np.stack([(1 - fv) * (1 - fu), (1 - fv) * fu, fv * (1 - fu), fv * fu], 1)
    return idx, w


def design(logs, delay, tau):
    """Rows for every sample whose whole lag window is straight and teleport-free."""
    alpha = 1 - np.exp(-DT / tau)
    h = alpha * (1 - alpha) ** np.arange(TAPS)
    h /= h.sum()
    ncol = len(V_GRID) * len(U_GRID)
    rows, ys, sup = [], [], np.zeros(ncol)
    for v, u, a, ok in logs:
        n = len(v)
        win = delay + TAPS
        okc = np.convolve(ok.astype(float), np.ones(win + 1), "full")[:n] == win + 1  # ok over [k-win, k]
        ks = np.flatnonzero(okc)
        ks = ks[ks >= win]
        X = np.zeros((len(ks), ncol))
        for j in range(TAPS):
            m = ks - delay - j
            idx, w = basis(v[m], u[m])
            for c in range(4):
                np.add.at(X, (np.arange(len(ks)), idx[:, c]), h[j] * w[:, c])
        rows.append(X)
        ys.append(a[ks])
    X, y = np.vstack(rows), np.concatenate(ys)
    return X, y


def smooth_rows():
    nv, nu = len(V_GRID), len(U_GRID)
    R = []
    for iv in range(nv):
        for iu in range(1, nu - 1):
            r = np.zeros(nv * nu)
            r[[iv * nu + iu - 1, iv * nu + iu, iv * nu + iu + 1]] = [1, -2, 1]
            R.append(r)
    for iu in range(nu):
        for iv in range(1, nv - 1):
            r = np.zeros(nv * nu)
            r[[(iv - 1) * nu + iu, iv * nu + iu, (iv + 1) * nu + iu]] = [1, -2, 1]
            R.append(r)
    return np.array(R)


def solve(X, y):
    R = smooth_rows() * LAM
    A = np.vstack([X, R])
    b = np.r_[y, np.zeros(len(R))]
    t = np.linalg.lstsq(A, b, rcond=None)[0]
    res = y - X @ t
    return t, float(np.sqrt(np.mean(res ** 2))), res


def fit(paths):
    logs = [load(p) for p in paths]
    best = None
    for delay in (0, 1, 2):
        for tau in (0.05, 0.1, 0.15, 0.2, 0.225, 0.25, 0.3):
            X, y = design(logs, delay, tau)
            t, rms, _ = solve(X, y)
            print("delay %d tau %.3f: rms %.3f (%d samples)" % (delay, tau, rms, len(y)))
            if best is None or rms < best[0]:
                best = (rms, delay, tau, t, X)
    rms, delay, tau, t, X = best
    table = t.reshape(len(V_GRID), len(U_GRID))
    support = X.sum(0).reshape(table.shape)
    return dict(delay=delay, tau=tau, rms=rms, table=table, support=support)


def steady_full_throttle(logs, hold=9):
    vs, acc = [], []
    for v, u, a, ok in logs:
        m = (u == 1) & ok
        held = np.convolve(m.astype(float), np.ones(hold), "full")[:len(v)] == hold
        vs.append(v[held])
        acc.append(a[held])
    return np.concatenate(vs), np.concatenate(acc)


def pedal_map(table):
    nb = len(BRK_GRID) - 1
    a_brk = table[:, nb::-1]  # u = 0, -0.02, ..., -1
    a_thr = table[:, nb:]
    return {"v": V_GRID, "thr": THR_GRID, "brk": BRK_GRID,
            "a_thr": np.round(a_thr, 3).tolist(), "a_brk": np.round(a_brk, 3).tolist()}


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    paths = args or sorted(glob.glob(os.path.join(ROOT, "logs", "*.run.csv")) +
                           glob.glob(os.path.join(ROOT, "logs", "sysid*.csv")))
    r = fit(paths)
    print("best: delay %d ticks, tau %.3f s, rms %.3f m/s^2" % (r["delay"], r["tau"], r["rms"]))
    np.set_printoptions(linewidth=200, suppress=True)
    print("u grid:", U_GRID)
    for iv, v in enumerate(V_GRID):
        print("v %4.0f a_ss" % v, " ".join("%6.1f" % x for x in r["table"][iv]))
        print("       sup ", " ".join("%6.0f" % x for x in r["support"][iv]))
    pm = pedal_map(r["table"])
    pm["fit"] = {"delay_ticks": r["delay"], "tau_s": r["tau"], "rms": round(r["rms"], 3)}
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    from fit_sysid import fit_a_drive
    vs, acc = steady_full_throttle([load(p) for p in paths])
    lines = fit_a_drive(vs, acc)
    print("a_drive_lines refit (%d steady full-throttle samples): %s" % (len(vs), lines))
    for v in range(20, 75, 5):
        m = np.abs(vs - v) < 1.5
        print("  v %d: lines %.2f  table %.2f  data median %s" % (
            v, min(a - b * v * v for a, b in lines), np.interp(v, V_GRID, r["table"][:, -1]),
            "%.2f (%d)" % (np.median(acc[m]), m.sum()) if m.sum() else "--"))
    print("a_brake check: table at brk=1:", np.round(r["table"][:, 0], 2).tolist())
    if "--write" in sys.argv:
        with open(PARAMS) as f:
            p = json.load(f)
        p["pedal_map"] = pm
        p["a_drive_lines"] = lines
        with open(PARAMS, "w") as f:
            json.dump(p, f, indent=2)
        print("wrote", PARAMS)


if __name__ == "__main__":
    main()
