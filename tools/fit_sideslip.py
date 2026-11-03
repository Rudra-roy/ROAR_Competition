"""Fit the sideslip feed-forward beta = k * a_lat_ref from ROAR_LOG run.csv files.

  python tools/fit_sideslip.py logs/final8b_*.run.csv [--write]

beta = body yaw - course (direction of the logged path), + into the turn. Body yaw is the logged `yaw` column,
else e_psi + the feed-forward the run used (FF_K_LOGGED * a_lat_ref; every Task 8/8b run used 0.004).
a_lat_ref = clip(v^2 kappa_ref, +-a_lat(line_s)) is what submission.py feeds forward; brake_ref = max(-a_ref, 0) /
A_brk with a_ref = d(v_ref^2)/(2 ds) of the logged v_ref (optional braking term kb * a_lat_ref * brake_ref). Both are reference quantities (functions of s and v only),
so the feed-forward adds no loop dynamics (Task 7 and Task 8c: feeding the measured course back made the car weave).
The regressors are taken `lag` ticks earlier (grid search). Only v > 20 m/s, after the launch, no respawn ticks.
--write stores the lag-0, no-braking-term k as sideslip_k in vehicle_params.json (the form submission.py uses; on
the Task 8b logs lag and braking term cut the residual only 0.042 -> 0.036 rad).
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "competition_code", "data")
DT = 0.05
FF_K_LOGGED = 0.004


def samples(path, tr, veh):
    d = np.genfromtxt(path, delimiter=",", names=True)
    L = tr.length
    ls, v = d["line_s"], d["v"]
    al = np.array([veh.a_lat(x) for x in ls])
    alr = np.clip(v ** 2 * d["kappa_ref"], -al, al)
    psi = np.interp(ls, tr.s, np.unwrap(tr.psi), period=L)
    yaw = np.unwrap(d["yaw"]) if "yaw" in d.dtype.names else psi + d["e_psi"] + FF_K_LOGGED * alr
    xy = (np.c_[np.interp(ls, tr.s, tr.xy[:, 0], period=L), np.interp(ls, tr.s, tr.xy[:, 1], period=L)]
          + d["e_y"][:, None] * np.c_[-np.sin(psi), np.cos(psi)])
    dxy = np.gradient(xy, axis=0)
    chi = np.unwrap(np.arctan2(dxy[:, 1], dxy[:, 0]))
    beta = (yaw - chi + np.pi) % (2 * np.pi) - np.pi
    ds = np.maximum(np.gradient(d["s_total"]), 1e-3)
    a_ref = np.gradient(d["v_ref"] ** 2) / (2 * ds)
    brk = np.maximum(-a_ref, 0) / (veh.a_brake * veh.brake_scale)
    jump = np.abs(np.gradient(d["s_total"])) > 6.0
    ok = (v > 20) & (d["s_total"] > 300) & ~jump
    return beta, alr, brk, ok


def fit(sets, lag, braking):
    X, Y = [], []
    for beta, alr, brk, ok in sets:
        a, b = np.roll(alr, lag), np.roll(brk, lag)
        m = ok & np.roll(ok, lag)
        m[:lag] = False
        X.append(np.c_[a, a * b][m] if braking else a[m, None])
        Y.append(beta[m])
    X, Y = np.vstack(X), np.concatenate(Y)
    coef = np.linalg.lstsq(X, Y, rcond=None)[0]
    return coef, float(np.sqrt(np.mean((Y - X @ coef) ** 2))), float(np.sqrt(np.mean(Y ** 2))), len(Y)


def main():
    sys.path.insert(0, os.path.join(ROOT, "competition_code"))
    from track import Track
    from vehicle import Vehicle
    files = [a for a in sys.argv[1:] if not a.startswith("--")]
    tr = Track(np.load(os.path.join(DATA, "racing_line.npz"))["xy"])
    veh = Vehicle.from_json(os.path.join(DATA, "vehicle_params.json"))
    sets = [samples(f, tr, veh) for f in files]
    print("%d runs" % len(sets))
    best = {}
    for braking in (False, True):
        for lag in (0, 2, 4, 6, 8, 10):
            coef, rms, rms0, n = fit(sets, lag, braking)
            print("braking term %-5s lag %2d ticks: coef %s rms %.4f rad (beta rms %.4f, n %d)"
                  % (braking, lag, np.round(coef, 5).tolist(), rms, rms0, n))
            if braking not in best or rms < best[braking][1]:
                best[braking] = (coef, rms, lag)
    k = float(fit(sets, 0, False)[0][0])
    per_run = [round(float(fit([s], 0, False)[0][0]), 5) for s in sets]
    print("sideslip_k (lag 0, no braking term) %.5f; per run %s" % (k, per_run))
    if "--write" in sys.argv:
        path = os.path.join(DATA, "vehicle_params.json")
        with open(path) as f:
            p = json.load(f)
        p["sideslip_k"] = round(k, 5)
        with open(path, "w") as f:
            json.dump(p, f, indent=2)
        print("wrote sideslip_k %.5f" % p["sideslip_k"])


if __name__ == "__main__":
    main()
