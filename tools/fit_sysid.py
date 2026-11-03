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
LIMITER_TICKS, LIMITER_ACC = 40, 0.3  # top speed reached: mean accel < 0.3 over >= 2 s of full throttle
V_MAX_CAP = 75.0
BRAKE_WIN = (3, 16)  # brake-response fit window: ticks before / after each brake onset
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


def fit_k_smooth(vs, ks):
    """K(v) = k0 / (1 + c v^2)  <=>  1/K = 1/k0 + (c/k0) v^2, both coefficients kept >= 0."""
    from scipy.optimize import nnls
    vs, ks = np.asarray(vs), np.asarray(ks)
    (icpt, slope), _ = nnls(np.c_[np.ones_like(vs), vs ** 2], 1.0 / ks)
    icpt = max(icpt, 1e-6)  # ponytail: K->inf at v=0 if the data say so; never hit with real bins
    return float(1.0 / icpt), float(slope / icpt)


def fit_brake_response(t, v, thr, brk):
    """Dead time + first-order lag of decel after each brake onset (forward differences, interval midpoints)."""
    a = np.diff(v) / np.diff(t)
    pre, post = BRAKE_WIN
    ts, ys, grp = [], [], []
    for k0 in np.where((brk[1:] == 1) & (brk[:-1] == 0))[0] + 1:
        if k0 < pre or k0 + post >= len(a) or v[k0] < 20 or not (brk[k0:k0 + post + 1] == 1).all():
            continue
        ks = np.arange(k0 - pre, k0 + post + 1)
        ts.append((ks - k0 + 0.5) * (t[1] - t[0]))
        ys.append(a[ks])
        grp.append(np.full(len(ks), len(grp)))
    if not ts:
        return None, None
    ts, ys, grp = np.concatenate(ts), np.concatenate(ys), np.concatenate(grp)
    onehot = (grp[:, None] == np.arange(grp.max() + 1)).astype(float)  # per-onset pre-brake level
    best = (np.inf, None, None)
    for delay in np.arange(0.0, 0.3001, 0.005):
        for tau in np.arange(0.02, 0.8001, 0.005):
            g = 1 - np.exp(-np.clip(ts - delay, 0, None) / tau)
            X = np.c_[onehot * (1 - g)[:, None], g]  # a = a0_j (1-g) + a_inf g
            r = ys - X @ np.linalg.lstsq(X, ys, rcond=None)[0]
            if r @ r < best[0]:
                best = (r @ r, float(delay), float(tau))
    return round(best[1], 3), round(best[2], 3)


def v_max_estimate(t, v, thr, brk, acc, lines):
    """Observed max if the car sat on the limiter, else the root of a_drive (capped)."""
    full = ((thr == 1) & (brk == 0)).astype(float)
    n = LIMITER_TICKS
    if len(v) >= n:
        cnt = np.convolve(full, np.ones(n), "valid")
        mean_acc = np.convolve(acc * full, np.ones(n), "valid") / n
        near_top = np.convolve(v < 0.9 * v.max(), np.ones(n), "valid") == 0  # not e.g. stuck on a wall
        if ((cnt == n) & (mean_acc < LIMITER_ACC) & near_top).any():
            return float(v.max()), "limiter"
    roots = [np.sqrt(al / be) for al, be in lines if be > 0 and al > 0]
    return float(min(roots + [V_MAX_CAP])), "a_drive_root"


def fit(log):
    t, v = log["t"], log["v"]
    thr, brk, steer = log["throttle"], log["brake"], log["steer"]
    acc = np.gradient(v, t)
    full = _steady((thr == 1) & (brk == 0) & (np.abs(steer) < 0.1)) & (v > 1)
    # skip each brake application's first 0.5 s: CARLA ramps decel 0 -> full over ~10 ticks
    on = np.convolve(brk == 1, np.ones(BRAKE_SETTLE), "full")[:len(v)] == BRAKE_SETTLE
    hard = _steady(on) & (v > 30)
    gv, gk = fit_steer_gain(v, steer, log["yaw_rate"])
    k0, c = fit_k_smooth(gv, gk)
    lines = fit_a_drive(v[full], acc[full])
    v_max, v_src = v_max_estimate(t, v, thr, brk, acc, lines)
    delay, tau = fit_brake_response(t, v, thr, brk)
    return {
        "a_drive_lines": lines,
        "a_brake": float(np.percentile(-acc[hard], 10)),
        "brake_delay_s": delay,
        "brake_tau_s": tau,
        "v_max": v_max,
        "v_max_source": v_src,
        "steer_gain_v": gv,
        "steer_gain_k": gk,
        "steer_gain_k0": k0,
        "steer_gain_c": c,
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
