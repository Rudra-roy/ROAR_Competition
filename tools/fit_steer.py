"""Lateral response from race logs (ROAR_LOG run.csv, Task 8e): what the MPC's steering model misses at speed.

  python tools/fit_steer.py [--line LINE.npz] logs/final8d_*.run.csv [--write]

Near-straight ticks only (|kappa_ref| < KREF_MAX), v >= 45 m/s, after the launch, no respawn ticks.
1. Yaw: k+ = k + dt/tau (K * -steer - k), command one tick late, against yaw_rate / v; tau shared, K per speed
   band. On the 8d logs: tau 0.40, K ~0.28 flat (vehicle.K extrapolates 0.17-0.23 above the sysid table, which
   ends at 43.6 m/s). --write-yaw sets steer_tau_s and replaces the steer_gain_v/k points above the sysid table.
2. Sideslip (the weave): beta = yaw - course (direction of the logged path on LINE) against
   b = lag(v^2 * k_act, tau_b), k_act = the curvature the MPC's model predicts for the logged steer with the
   CURRENT params: lag(K(v) * -steer, steer_tau_s). So refit after --write-yaw. --write stores k_b, tau_b as the
   mpc options beta_k / beta_tau_s.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "competition_code", "data")
PARAMS = os.path.join(DATA, "vehicle_params.json")
DT = 0.05
KREF_MAX = 0.004
BANDS = (45, 50, 55, 60, 65, 72)
SYSID_V_MAX = 43.6  # top of the sysid steer table (Task 3); points above it come from this tool
TAUS = np.arange(0.10, 1.50001, 0.05)


def lag(x, tau):
    """First-order lag, input one tick late: f[k] = f[k-1] + dt/tau (x[k-1] - f[k-1])."""
    a, f = min(DT / tau, 1.0), np.zeros_like(x)
    for k in range(1, len(x)):
        f[k] = f[k - 1] + a * (x[k - 1] - f[k - 1])
    return f


def load(path, tr, veh):
    """-> dict of per-tick arrays: u (-steer), v, ky (yaw curvature), beta, x_b (v^2 k_act), ok mask."""
    d = np.genfromtxt(path, delimiter=",", names=True)
    L, v = tr.length, d["v"]
    ls = d["line_s"]
    psi = np.interp(ls, tr.s, np.unwrap(tr.psi), period=L)
    xy = (np.c_[np.interp(ls, tr.s, tr.xy[:, 0], period=L), np.interp(ls, tr.s, tr.xy[:, 1], period=L)]
          + d["e_y"][:, None] * np.c_[-np.sin(psi), np.cos(psi)])
    dxy = np.gradient(xy, axis=0)
    yaw = np.unwrap(d["yaw"])
    beta = (yaw - np.unwrap(np.arctan2(dxy[:, 1], dxy[:, 0])) + np.pi) % (2 * np.pi) - np.pi
    jump = np.abs(np.gradient(d["s_total"])) > 6.0
    ok = (np.abs(d["kappa_ref"]) < KREF_MAX) & (d["s_total"] > 300) & ~jump & (v >= BANDS[0])
    return dict(u=-d["steer"], v=v, ky=np.gradient(yaw) / DT / np.maximum(v, 1.0), beta=beta,
                x_b=v * v * lag(np.array([veh.K(x) for x in v]) * -d["steer"], veh.p["steer_tau_s"]), ok=ok)


def fit_gain(X, Y):
    g = float(X @ Y / (X @ X))
    return g, 1 - float(np.mean((Y - g * X) ** 2) / Y.var())


def fit_yaw(runs, bands=BANDS, taus=TAUS):
    """-> (tau, [(v_mean, K, n, r2)]) with tau minimising the summed per-band normalised residual."""
    best = None
    for tau in taus:
        filt, out, cost = [lag(r["u"], tau) for r in runs], [], 0.0
        for lo, hi in zip(bands[:-1], bands[1:]):
            m = [r["ok"] & (r["v"] >= lo) & (r["v"] < hi) for r in runs]
            Y = np.concatenate([r["ky"][mm] for r, mm in zip(runs, m)])
            if len(Y) < 50:
                continue
            g, r2 = fit_gain(np.concatenate([f[mm] for f, mm in zip(filt, m)]), Y)
            cost += 1 - r2
            out.append((float(np.concatenate([r["v"][mm] for r, mm in zip(runs, m)]).mean()), g, len(Y), r2))
        if best is None or cost < best[0]:
            best = (cost, float(tau), out)
    return best[1], best[2]


def fit_beta(runs, taus=TAUS):
    """-> (k_b, tau_b, r2): beta = k_b * lag(v^2 k_act, tau_b) on the ok ticks."""
    Y = np.concatenate([r["beta"][r["ok"]] for r in runs])
    best = None
    for tau in taus:
        g, r2 = fit_gain(np.concatenate([lag(r["x_b"], tau)[r["ok"]] for r in runs]), Y)
        if best is None or r2 > best[2]:
            best = (g, float(tau), r2)
    return best


def main():
    sys.path.insert(0, os.path.join(ROOT, "competition_code"))
    from track import Track
    from vehicle import Vehicle
    args = sys.argv[1:]
    line = os.path.join(DATA, "racing_line.npz")
    if "--line" in args:
        line = args[args.index("--line") + 1]
        args.remove("--line"), args.remove(line)
    with open(PARAMS) as f:
        p = json.load(f)
    tr = Track(np.load(line)["xy"])
    runs = [load(a, tr, Vehicle(p)) for a in args if not a.startswith("--")]
    tau, bands = fit_yaw(runs)
    print("yaw: lag tau = %.2f s" % tau)
    for v, k, n, r2 in bands:
        print("  v %5.1f  K %.4f (vehicle.K %.4f)  n %5d  R2 %.3f" % (v, k, Vehicle(p).K(v), n, r2))
    kb, tb, r2 = fit_beta(runs)
    print("sideslip: beta = %.5f * lag(v^2 k_act, %.2f s), R2 %.3f" % (kb, tb, r2))
    if "--write-yaw" in args:
        keep = [i for i, v in enumerate(p["steer_gain_v"]) if v <= SYSID_V_MAX + 1e-6]
        hs = [(v, k) for v, k, _, _ in bands if v > SYSID_V_MAX]
        p["steer_gain_v"] = [p["steer_gain_v"][i] for i in keep] + [v for v, _ in hs]
        p["steer_gain_k"] = [p["steer_gain_k"][i] for i in keep] + [k for _, k in hs]
        p["steer_tau_s"] = round(tau, 3)
        print("wrote steer_tau_s %.2f and %d high-speed K points (refit the sideslip now)" % (tau, len(hs)))
    elif "--write" in args:
        p["mpc"]["beta_k"], p["mpc"]["beta_tau_s"] = round(kb, 5), round(tb, 3)
        print("wrote mpc.beta_k %.5f, mpc.beta_tau_s %.2f" % (kb, tb))
    if "--write-yaw" in args or "--write" in args:
        with open(PARAMS, "w") as f:
            json.dump(p, f, indent=2)


if __name__ == "__main__":
    main()
