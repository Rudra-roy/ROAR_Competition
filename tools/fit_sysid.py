"""Fit vehicle_params.json from a sysid log.

Usage: fit_sysid.py logs/sysid.csv logs/maneuverable_waypoints.npy competition_code/data/vehicle_params.json
"""
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Winners' per-section mu, at these maneuverable-waypoint section starts (same order).
SECTION_WP = [2611, 322, 557, 739, 1158, 1317, 1516, 1881, 1944, 2359]
SECTION_MU = [3.05, 2.97, 3.35, 3.36, 3.14, 2.98, 3.30, 2.85, 2.98, 2.10]
BRAKE_SETTLE = 11  # ticks of brake=1 (incl. this one) before a sample counts
KNOBS = {
    "speed_scale": 0.85,
    "brake_scale": 1.0,
    "mpc": {"N": 60, "dt": 0.05, "q_ey": 10, "q_epsi": 50, "q_v": 0.5,
            "r_dkappa": 2e5, "r_da": 0.01, "ey_max": 1.0},
}


def a_drive(p, v):
    return float(min(a - b * v * v for a, b in p["a_drive_lines"]))


def _steady(m):
    """Mask true only where the sample and both neighbours satisfy m (centered difference spans them)."""
    out = m.copy()
    out[1:-1] &= m[:-2] & m[2:]
    out[[0, -1]] = False
    return out


def _lines(x, y, k, iters=50):
    """Least-squares fit of y ~ min_j(alpha_j + slope_j * x) with k lines (k-plane regression)."""
    groups = np.array_split(np.arange(len(x)), k)
    lab = np.empty(len(x), int)
    for j, g in enumerate(groups):
        lab[g] = j
    for _ in range(iters):
        coef = [np.polyfit(x[lab == j], y[lab == j], 1) for j in np.unique(lab) if (lab == j).sum() >= 2]
        pred = np.array([np.polyval(c, x) for c in coef])
        new = pred.argmin(0)
        if np.array_equal(new, lab):
            break
        lab = new
    return coef, float(((pred.min(0) - y) ** 2).sum())


def fit_a_drive(v, a, bin_w=1.0, min_count=5, max_lines=4):
    edges = np.arange(0.0, v.max() + bin_w, bin_w)
    b = np.digitize(v, edges)
    vb, ab = [], []
    for i in np.unique(b):
        sel = b == i
        if sel.sum() >= min_count:
            vb.append(np.median(v[sel]))
            ab.append(np.median(a[sel]))
    x, y = np.array(vb) ** 2, np.array(ab)
    best = None
    for k in range(1, max_lines + 1):
        coef, sse = _lines(x, y, k)
        if best is None or sse < 0.9 * best[1]:  # extra line must cut the error by >10 %
            best = (coef, sse)
    return sorted([[float(c[1]), float(-c[0])] for c in best[0]], key=lambda l: l[1])


def fit_steer_gain(v, steer, yaw_rate, bin_w=5.0, min_count=20):
    m = (np.abs(steer) > 0.02) & (v > 5)
    v, k = v[m], -yaw_rate[m] / (v[m] * steer[m])
    b = np.round(v / bin_w)
    vs, ks = [], []
    for i in np.unique(b):
        sel = b == i
        if sel.sum() >= min_count:
            vs.append(float(np.median(v[sel])))
            ks.append(float(np.median(k[sel])))
    return vs, ks


def fit(log):
    t, v = log["t"], log["v"]
    thr, brk, steer = log["throttle"], log["brake"], log["steer"]
    acc = np.gradient(v, t)
    full = _steady((thr == 1) & (brk == 0) & (np.abs(steer) < 0.1)) & (v > 1)
    # skip each brake application's first 0.5 s: CARLA ramps decel 0 -> full over ~10 ticks
    on = np.convolve(brk == 1, np.ones(BRAKE_SETTLE), "full")[:len(v)] == BRAKE_SETTLE
    hard = _steady(on) & (v > 30)
    gv, gk = fit_steer_gain(v, steer, log["yaw_rate"])
    return {
        "a_drive_lines": fit_a_drive(v[full], acc[full]),
        "a_brake": float(np.percentile(-acc[hard], 10)),
        "v_max": float(v.max()),
        "steer_gain_v": gv,
        "steer_gain_k": gk,
    }


def a_lat_sections(wp_xyz):
    sys.path.insert(0, os.path.join(REPO, "competition_code"))
    from track import Track
    tr = Track(np.load(os.path.join(REPO, "competition_code", "data", "racing_line.npz"))["xy"])
    secs = [[tr.project(wp_xyz[i])[1], mu * 9.81] for i, mu in zip(SECTION_WP, SECTION_MU)]
    return sorted([[round(float(s), 1), round(a, 3)] for s, a in secs])


def main(csv_path, wp_path, out_path):
    d = np.genfromtxt(csv_path, delimiter=",", names=True)
    p = fit({k: np.asarray(d[k], float) for k in d.dtype.names})
    p["a_lat_sections"] = a_lat_sections(np.load(wp_path))
    p.update(KNOBS)
    with open(out_path, "w") as f:
        json.dump(p, f, indent=2)
    print(json.dumps(p, indent=2))
    print("a_drive:", {v: round(a_drive(p, v), 2) for v in range(10, 80, 10)})


if __name__ == "__main__":
    main(*sys.argv[1:4])
