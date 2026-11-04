"""Steering response from race logs (ROAR_LOG run.csv): lag and high-speed gain of steer -> yaw curvature.

  python tools/fit_steer.py logs/final8d_*.run.csv [--write]

Model (the MPC's k_act state): k+ = k + dt/tau (K(v) * -steer - k), the command one tick late, against the
measured yaw curvature yaw_rate / v. Near-straight ticks only (|kappa_ref| < KREF_MAX: no sideslip build-up), after the
launch, no respawn ticks. tau: one shared value, grid search, gain free per speed band (>= 45 m/s, where the weave
is; on the 8d logs a 30 m/s floor gives 0.30, but under-modelled lag is what weaves and over-modelled lag is benign
at low speed in the offline closed loop). K: per band above the sysid table (vehicle.K extrapolates there with the
low-speed shape and was 1.4-1.65x low at 55-72 m/s), written as extra steer_gain_v/k points at the band's mean speed.
--write sets steer_tau_s and replaces the table points above the sysid range.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARAMS = os.path.join(ROOT, "competition_code", "data", "vehicle_params.json")
DT = 0.05
KREF_MAX = 0.004
SYSID_V_MAX = 43.6  # top of the sysid steer table (Task 3); later points come from this tool
BANDS = (45, 50, 55, 60, 65, 72)
TAUS = np.arange(0.10, 0.80001, 0.05)


def load(path):
    d = np.genfromtxt(path, delimiter=",", names=True)
    v = d["v"]
    ky = np.gradient(np.unwrap(d["yaw"])) / DT / np.maximum(v, 1.0)
    jump = np.abs(np.gradient(d["s_total"])) > 6.0
    ok = (np.abs(d["kappa_ref"]) < KREF_MAX) & (d["s_total"] > 300) & ~jump
    return -d["steer"], v, ky, ok


def lag(x, tau):
    a, f = min(DT / tau, 1.0), np.zeros_like(x)
    for k in range(1, len(x)):
        f[k] = f[k - 1] + a * (x[k - 1] - f[k - 1])
    return f


def fit(runs, bands=BANDS, taus=TAUS):
    """-> (tau, [(v_mean, K, n, r2)] per band) with tau minimising the summed per-band normalised residual."""
    best = None
    for tau in taus:
        out, cost = [], 0.0
        filt = [lag(u, tau) for u, _, _, _ in runs]
        for lo, hi in zip(bands[:-1], bands[1:]):
            m = [ok & (v >= lo) & (v < hi) for _, v, _, ok in runs]
            X = np.concatenate([f[mm] for f, mm in zip(filt, m)])
            Y = np.concatenate([r[2][mm] for r, mm in zip(runs, m)])
            V = np.concatenate([r[1][mm] for r, mm in zip(runs, m)])
            if len(Y) < 50:
                continue
            g = float(X @ Y / (X @ X))
            res = float(np.mean((Y - g * X) ** 2) / Y.var())
            cost += res
            out.append((float(V.mean()), g, len(Y), 1 - res))
        if best is None or cost < best[0]:
            best = (cost, float(tau), out)
    return best[1], best[2]


def main():
    paths = [a for a in sys.argv[1:] if not a.startswith("--")]
    tau, bands = fit([load(p) for p in paths])
    print("steer lag tau = %.2f s" % tau)
    for v, k, n, r2 in bands:
        print("  v %5.1f  K %.4f  n %5d  R2 %.3f" % (v, k, n, r2))
    if "--write" in sys.argv:
        with open(PARAMS) as f:
            p = json.load(f)
        keep = [i for i, v in enumerate(p["steer_gain_v"]) if v <= SYSID_V_MAX + 1e-6]
        hs = [(v, k) for v, k, _, _ in bands if v > SYSID_V_MAX]
        p["steer_gain_v"] = [p["steer_gain_v"][i] for i in keep] + [v for v, _ in hs]
        p["steer_gain_k"] = [p["steer_gain_k"][i] for i in keep] + [k for _, k in hs]
        p["steer_tau_s"] = round(tau, 3)
        with open(PARAMS, "w") as f:
            json.dump(p, f, indent=2)
        print("wrote steer_tau_s %.2f and %d high-speed K points" % (tau, len(hs)))


if __name__ == "__main__":
    main()
