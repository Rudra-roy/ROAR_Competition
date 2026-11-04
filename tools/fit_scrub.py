"""Fit the tyre-scrub model from race logs: at full throttle the car accelerates S(|a_lat|) less than a_drive(v).

  python tools/fit_scrub.py RUN.csv [RUN.csv ...] [--write]

Samples: throttle >= 0.97 for >= 0.5 s, no brake, 15 < v < 70; |a_lat| = |v * yaw rate|; loss = a_drive(v) - dv/dt.
Model (convex, so the planner's secant chords are conservative): S = k x^2 up to x_sat, then + m (x - x_sat).
Fit on bin medians (robust to the transient spikes). --write stores {"x": breakpoints, "s": S(x)} as
vehicle_params["scrub"].
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "competition_code"))
from vehicle import Vehicle  # noqa: E402

PARAMS = os.path.join(ROOT, "competition_code", "data", "vehicle_params.json")
X_SAT = 26.0
BREAKS = [0, 5, 10, 15, 20, 26, 32, 40]


def samples(run_csv, veh):
    d = np.genfromtxt(run_csv, delimiter=",", names=True)
    v = d["v"]
    yr = np.gradient(np.unwrap(d["yaw"])) / 0.05
    th = d["throttle"] >= 0.97
    held = np.convolve(th, np.ones(10), "full")[:len(th)] >= 10
    ok = held & (d["brake"] == 0) & (v > 15) & (v < 70)
    ad = np.array([veh.a_drive(x) for x in v[ok]])
    return np.abs(v * yr)[ok], ad - (np.gradient(v) / 0.05)[ok]


def fit(x, loss, x_sat=X_SAT):
    edges = np.arange(0, 36, 3.0)
    c, m = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        k = (x >= a) & (x < b)
        if k.sum() > 30:
            c.append(0.5 * (a + b))
            m.append(np.median(loss[k]))
    c, m = np.array(c), np.array(m)
    lo = c <= x_sat
    k2 = max(float((m[lo] * c[lo] ** 2).sum() / (c[lo] ** 4).sum()), 0.0)
    hi = ~lo
    base = k2 * x_sat ** 2
    slope = max(float(((m[hi] - base) * (c[hi] - x_sat)).sum() / ((c[hi] - x_sat) ** 2).sum()), 2 * k2 * x_sat) \
        if hi.any() else 2 * k2 * x_sat
    return k2, slope, c, m


def model(x, k2, slope, x_sat=X_SAT):
    x = np.asarray(x, float)
    return np.where(x <= x_sat, k2 * x * x, k2 * x_sat ** 2 + slope * (x - x_sat))


if __name__ == "__main__":
    runs = [a for a in sys.argv[1:] if not a.startswith("--")]
    with open(PARAMS) as f:
        p = json.load(f)
    p.pop("scrub", None)
    veh = Vehicle(p)
    xs, ls = zip(*[samples(r, veh) for r in runs])
    k2, slope, c, m = fit(np.concatenate(xs), np.concatenate(ls))
    for ci, mi in zip(c, m):
        print("  |a_lat| %4.1f  loss median %5.2f  model %5.2f" % (ci, mi, model(ci, k2, slope)))
    print("k2 %.5f, slope beyond %.0f m/s^2: %.3f" % (k2, X_SAT, slope))
    if "--write" in sys.argv:
        p["scrub"] = {"x": BREAKS, "s": [round(float(model(b, k2, slope)), 4) for b in BREAKS]}
        with open(PARAMS, "w") as f:
            json.dump(p, f, indent=2)
        print("wrote vehicle_params scrub:", p["scrub"])
