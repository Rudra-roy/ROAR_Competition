"""Minimum-curvature racing line -> competition_code/data/racing_line.npz.

  python tools/mincurv_line.py competition_code/data/racing_line.npz [--straights OLD.npz | --widen OLD.npz
                                [--factors 1,1,1,1]] [--remap-from OLD.npz]

Free mode (default) is the full min-curvature line in the waypoint corridor. Raced in Task 8e (mc8d): the lane
polygon is not the physical corridor (hits at |d_centre| 2.8 and 4.0 in the first km), so it does not ship.
--straights OLD.npz (shipped): the same QP in the raced line's own frame, moving only its straights (see
smooth_straights); its corners stay exactly where the car has raced them.

Method (ported from analysis/path_study.py, qp_offsets "curv"): offsets a along the normals of the smoothed
centreline (Track of the official waypoints, 2 m), p = c + a n, minimise sum |kappa|^2 ds with kappa ~ second
differences scaled by the local spacing (ds^-1.5 weight: the integral), iterated QP (spacing re-linearised), OSQP,
closed loop. Corridor = the runner's road polygon (waypoints +- lane_width/2, as RoarPyWaypoint.line_representation)
minus the car half-width + YAW_ALLOW, intersected with |a| <= MAX_OFFSET - SLACK (the cap is checked against the
centreline polyline, which the smoothed centreline differs from by a few cm).
The waypoints file is read with np.load only (data). The old line stays reproducible with tools/export_line.py.
--widen OLD.npz (Task 8h): the same QP along the old line's normals, free only in the corner WINDOWS (braking point
-> corner exit, line_s of OLD, cosine-blended to 0 over BLEND m each side), bounds = the proven envelope
(tools/envelope.py: our line + the S25 car's driven offsets, S25 side shrunk 0.3 m, |d| <= 4). --factors scales each
window's shift (0 = that corner as before, 0.5 = half way). Track's smoothing spline is global (refitting moves
every corner's kappa by up to ~80 %), so the output npz carries its own spline (tck_t / tck_c, read by
track.load_track): the old fit with only the coefficients whose support lies inside a window refitted to the new
points; outside the windows the line the car follows is the old one exactly.
--remap-from: a_lat_sections in vehicle_params.json are keyed by line_s of the OLD line; each start is converted to
xy on the old line and projected onto the new one (values kept, list re-sorted).
"""
import json
import os
import sys

import numpy as np
import osqp
import scipy.sparse as sp
import transforms3d as tr3d

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "competition_code"), os.path.join(ROOT, "tools")]
from export_line import offsets  # noqa: E402
from track import Track, fit_spline  # noqa: E402
from scipy.ndimage import gaussian_filter1d  # noqa: E402

WP = os.path.join(os.path.dirname(ROOT), "ROAR_PY", "roar_py_carla", "roar_py_carla", "assets", "waypoints", "Monza.npz")
DATA = os.path.join(ROOT, "competition_code", "data")
HALF_W = 1.0817  # Tesla Model 3 bbox half-width (sim)
YAW_ALLOW = 0.3  # body corner reach at ~0.1 rad heading + tracking slop
MAX_OFFSET = 4.0  # same wall cap as the export_line line
ENV_SHRINK = 0.3  # m, envelope pulled in where S25 is the extreme (--shrink)
SLACK = 0.05  # QP bound below the cap: smoothed vs polyline centreline + the runtime Track resampling
DS = 2.0
STRAIGHT_DELTA = 0.5  # m, max lateral move of the raced line on its straights
STRAIGHT_KAPPA = 0.002  # 1/m, |kappa| (15 m smoothed) below which the raced line counts as straight
WALL_OUTER = 3.5  # m |d_centre|: no move toward the near wall from here out (ramped over 0.5 m)
WALL_IN = 3.95  # m |d_centre|: raced-line points beyond may move in by up to the excess, never out (cap 4.0 checked)
ITERS = 4
# corner windows, line_s of the 8e/e115 line (racing_line.npz at 8d1f3a4): braking point (plan's top speed) -> exit
# (|kappa| back to straight); names = corner_compare sectors c937 chicane, c1697, c3835, c5413 hairpin
WINDOWS = [(780.0, 1060.0), (1600.0, 1780.0), (3720.0, 3860.0), (5240.0, 5500.0)]
# Raced (Task 8h): the car slid further out than any driven path with c937 (0.5: +0.85 m at its exit) and c3835 (1.0:
# +0.9 m, 0.5: +0.56 m); c1697 is pinned at its apex by the envelope (S25 only ~0.3 m wider). Shipped: hairpin only.
SHIP_FACTORS = (1.0, 1.0, 1.0, 1.0)  # shipped (w5, tyre MPC): every window, envelope shrink 0.15, cap 4.6 m
SHIP_SHRINK, SHIP_MAX_OFFSET = 0.15, 4.6
# every corner (c1329, the c2695/2812/2899 complex and c3993/c4113 added), for --full-width: bounds = the corridor
# edges minus the car, per-window caps (--caps) that the race loop lowers 0.2 m at the corner that collided
WINDOWS_ALL = [(780.0, 1060.0), (1250.0, 1420.0), (1600.0, 1780.0), (2540.0, 2960.0), (3720.0, 3860.0),
               (3930.0, 4160.0), (5240.0, 5500.0)]
TEST_FACTORS = (1.0, 0.0, 1.0, 1.0)  # the tool's tests: every corner whose min radius the envelope lets rise
BLEND = 60.0  # m, cosine ramp of the free band each side of a window
KNOT_DS = 6.0  # m, spline knot spacing inside the windows (the old fit has ~20-45 m knots there)
MARGIN = 40.0  # m beyond the blend: spline patch with the old curve as target (room to meet its coefficients)
QP_EPS = 1e-7


def edges_from_waypoints(path=WP):
    d = np.load(path, allow_pickle=False)
    loc, rot, w = d["locations"], d["rotations"], d["lane_widths"]
    left, right = [], []
    for p, r, lw in zip(loc, rot, w):  # same as RoarPyWaypoint.line_representation
        off = tr3d.euler.euler2mat(*r).dot([0, lw / 2, 0])
        left.append((p + off)[:2])
        right.append((p - off)[:2])
    return np.array(left), np.array(right), loc[:, :2]


def signed_offset(p, poly):
    """Signed distance of points p from closed polyline poly (+ = left of its direction)."""
    from scipy.spatial import cKDTree
    _, i = cKDTree(poly).query(p)
    n = len(poly)
    best = np.full(len(p), np.inf)
    for a, b in ((i - 1) % n, i), (i, (i + 1) % n):
        seg = poly[b] - poly[a]
        r = p - poly[a]
        u = np.clip((r * seg).sum(1) / np.maximum((seg * seg).sum(1), 1e-12), 0, 1)
        d = np.linalg.norm(r - u[:, None] * seg, axis=1)
        sgn = np.sign(seg[:, 0] * r[:, 1] - seg[:, 1] * r[:, 0])
        best = np.where(d < np.abs(best), sgn * d, best)
    return best


def qp_offsets(c, nrm, lo, hi, iters=ITERS):
    """Offsets a (p = c + a n) minimising sum |kappa|^2 ds, closed loop."""
    n = len(c)
    I = np.arange(n)
    a = np.zeros(n)
    rows = [((I - 1) % n, 1.0), (I, -2.0), ((I + 1) % n, 1.0)]
    for _ in range(iters):
        p = c + a[:, None] * nrm
        ds = 0.5 * (np.linalg.norm(p - np.roll(p, 1, 0), axis=1) + np.linalg.norm(np.roll(p, -1, 0) - p, axis=1))
        wgt = 1.0 / np.maximum(ds, 0.2) ** 1.5  # |d2p|/ds^2 = kappa, times sqrt(ds) for the integral
        blocks, b = [], np.zeros(2 * n)
        for k in (0, 1):
            data, cols = [], []
            for j, coef in rows:
                cols.append(j)
                data.append(wgt * coef * nrm[j, k])
                b[k * n:(k + 1) * n] += wgt * coef * c[j, k]
            blocks.append(sp.csr_matrix((np.concatenate(data), (np.tile(I, 3), np.concatenate(cols))), shape=(n, n)))
        A = sp.vstack(blocks).tocsc()
        P, q = sp.triu(A.T @ A).tocsc(), A.T @ b
        s = osqp.OSQP()
        s.setup(P=P, q=q, A=sp.identity(n, format="csc"), l=lo, u=hi, verbose=False,
                eps_abs=QP_EPS, eps_rel=QP_EPS, max_iter=200000, polish=True)
        r = s.solve(raise_error=False)
        # OSQP 1.x can hold a converged iterate to max_iter (or "solved inaccurate") on its duality-gap test: the
        # straights QP stops at 200k iterations with primal 8e-9 / dual 2e-8, one free-line iteration reports
        # "solved inaccurate" at primal 3.9e-7 (tol 5e-7). Accept those only when the residuals meet OSQP's own
        # primal/dual criteria (eps_abs + eps_rel * scale, A = I here); anything else raises.
        P_full = (A.T @ A).tocsc()
        tol_p = QP_EPS * (1 + np.abs(r.x).max())
        tol_d = QP_EPS * (1 + max(np.abs(P_full @ r.x).max(), np.abs(q).max(), np.abs(r.y).max()))
        near = (osqp.SolverStatus.OSQP_MAX_ITER_REACHED, osqp.SolverStatus.OSQP_SOLVED_INACCURATE)
        stuck = r.info.status_val in near and r.info.prim_res <= tol_p and r.info.dual_res <= tol_d
        if r.info.status_val != osqp.SolverStatus.OSQP_SOLVED and not stuck:
            raise RuntimeError("line QP not solved: %s (primal %.1e / %.1e, dual %.1e / %.1e)"
                               % (r.info.status, r.info.prim_res, tol_p, r.info.dual_res, tol_d))
        a = np.clip(r.x, lo, hi)
    return c + a[:, None] * nrm, a


def build(slack=SLACK):
    left, right, wp = edges_from_waypoints()
    ctr = Track(wp, ds=DS)  # smooth centreline, uniform 2 m
    c, psi = ctr.xy, ctr.psi
    nrm = np.c_[-np.sin(psi), np.cos(psi)]
    margin = HALF_W + YAW_ALLOW
    lo = np.maximum(-(np.abs(signed_offset(c, right)) - margin), -(MAX_OFFSET - slack))
    hi = np.minimum(np.abs(signed_offset(c, left)) - margin, MAX_OFFSET - slack)
    return qp_offsets(c, nrm, lo, hi)[0]


def straight_weight(old_xy, kthr=STRAIGHT_KAPPA):
    """Per raced-line point: 1 on its straights, 0 in corners, ramped over ~10 m; and the point's Track index."""
    tr = Track(old_xy)
    idx = np.array([tr.project(p)[0] for p in old_xy])
    step = np.linalg.norm(np.roll(old_xy, -1, 0) - old_xy, axis=1).mean()
    ks = gaussian_filter1d(np.abs(tr.kappa), 15 / tr.ds, mode="wrap")[idx]
    return np.clip(2 * gaussian_filter1d((ks < kthr).astype(float), 10 / step, mode="wrap") - 1, 0, 1), tr, idx


def smooth_straights(old_xy, centre, delta=STRAIGHT_DELTA):
    """Min-curvature QP with offsets a along the raced line's own normals: |a| <= delta * w (w = straight_weight),
    no move toward the near wall beyond WALL_OUTER; a point beyond WALL_IN may move in by up to its excess (its bound
    is [excess inward, 0]: the QP may leave it where it is; main() checks the 4.0 cap). Removes the raced line's
    curvature wiggle on the straights (the weave the MPC follows, Task 8e) and leaves its corners as raced."""
    w, tr, idx = straight_weight(old_xy)
    nrm = np.c_[-np.sin(tr.psi[idx]), np.cos(tr.psi[idx])]  # left normal, same sense as the centreline's
    dc = offsets(old_xy, centre)[0]
    out = delta * w * np.clip((WALL_OUTER - np.abs(dc)) / 0.5, 0, 1) + np.minimum(WALL_IN - np.abs(dc), 0)
    lo = np.where(dc > 0, -delta * w, -out)  # the near wall is on the +n side when dc > 0
    hi = np.where(dc > 0, out, delta * w)
    lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
    return qp_offsets(old_xy, nrm, lo, hi)[0]


def window_weight(ls, L, windows=WINDOWS, blend=BLEND):
    """Per line_s: 1 inside a window, cosine ramp to 0 over `blend` m each side, 0 elsewhere; and the window index
    (-1 outside)."""
    w, k = np.zeros(len(ls)), np.full(len(ls), -1)
    for j, (a, b) in enumerate(windows):
        da = (ls - a + L / 2) % L - L / 2  # signed distance past a (closed loop)
        db = (ls - b + L / 2) % L - L / 2
        r = np.where(da < 0, -da, np.where(db > 0, db, 0.0))  # distance outside [a, b]
        wj = np.where(r < blend, 0.5 * (1 + np.cos(np.pi * r / blend)), 0.0)
        k = np.where(wj > 0, j, k)
        w = np.maximum(w, wj)
    return w, k


def chord_u(xy):
    """splprep's parameter of each point of a closed line: normalised cumulative chord length (0 at xy[0])."""
    pts = np.vstack([xy, xy[:1]])
    u = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    return (u / u[-1])[:-1], u[-1]


def patch_spline(tck, u, target, free, knot_ds_u):
    """Spline tck with knots added every knot_ds_u inside each run of `free` points (exact knot insertion: the curve
    is unchanged) and the coefficients whose support covers free points only refitted (least squares) to the targets
    at parameters u. Every other coefficient stays as it was, so outside the runs the curve is exactly the old one."""
    from scipy.interpolate import BSpline, insert
    k = tck[2]
    edges = np.flatnonzero(np.diff(np.r_[0, free.astype(int), 0]))  # runs [edges[2i], edges[2i+1])
    for a, b in zip(edges[::2], edges[1::2]):
        for x in np.arange(u[a] + knot_ds_u, u[b - 1] - knot_ds_u / 2, knot_ds_u):
            if np.abs(tck[0] - x).min() > knot_ds_u / 4:
                tck = insert(x, tck, 1, per=1)
    t = tck[0]
    c = [np.array(ci[:len(t) - k - 1], float) for ci in tck[1]]
    nb = len(c[0])
    fj = [j for j in range(nb - k) if t[j] >= 0 and t[j + k + 1] <= 1  # not the periodic wrap-around copies
          and free[(u >= t[j]) & (u <= t[j + k + 1])].all()]
    if fj:
        rows = np.flatnonzero(np.any([(u >= t[j]) & (u <= t[j + k + 1]) for j in fj], axis=0))
        B = BSpline.design_matrix(u[rows], t, k).toarray()
        fixed = np.setdiff1d(np.arange(nb), fj)
        for d in (0, 1):
            c[d][fj] = np.linalg.lstsq(B[:, fj], target[rows, d] - B[:, fixed] @ c[d][fixed], rcond=None)[0]
    return t, c, k


def widen_corners(old_xy, centre, s25_xy, factors=(1.0, 1.0, 1.0, 1.0), windows=WINDOWS, full_width=False,
                  caps=None):
    """-> (points, tck). The QP runs on the old line as the car follows it (its own smoothing spline sampled at each
    point's parameter, not the noisy points): min-curvature offsets along its normals, free only in the WINDOWS
    (cosine blended), within the proven envelope; window j's shift scaled by factors[j]. The spline is the old fit
    with dense knots and refitted coefficients over window +- (BLEND + MARGIN); the target in the margin is the old
    curve itself, so the patch meets the untouched coefficients smoothly and the curve outside window +- (BLEND +
    MARGIN) is the old one exactly. points (for tools that read xy) = the new curve's samples inside the patch, the
    old points elsewhere."""
    from scipy.interpolate import splev
    from envelope import envelope, station
    tck = fit_spline(old_xy)
    u, chord = chord_u(old_xy)
    base = np.array(splev(u, tck)).T
    dx, dy = splev(u, tck, der=1)
    nrm = np.c_[-dy, dx] / np.hypot(dx, dy)[:, None]  # left normal
    tr = Track(old_xy, tck=tck)
    ls = np.array([tr.project(p)[1] for p in base])
    keep = [fi > 0 for fi in factors]  # a 0-factor corner is not touched at all
    wins = [wi for wi, k in zip(windows, keep) if k]
    fac = np.array([fi for fi in factors if fi > 0], float)
    capw = np.array([c for c, k in zip(caps if caps is not None else [MAX_OFFSET] * len(windows), keep) if k], float)
    w, kw = window_weight(ls, tr.length, wins)
    Lc = np.linalg.norm(np.roll(centre, -1, 0) - centre, axis=1).sum()
    sc, dc = station(base, centre)
    if full_width:  # the road itself: edges minus the car (+ YAW_ALLOW), then the per-window caps
        left, right, _ = edges_from_waypoints()
        margin = HALF_W + YAW_ALLOW
        lo = -(np.abs(signed_offset(base, right)) - margin)
        hi = np.abs(signed_offset(base, left)) - margin
        cap = np.where(kw >= 0, capw[np.maximum(kw, 0)], MAX_OFFSET) - SLACK
    else:
        st, lo_s, hi_s, _, _ = envelope(old_xy, s25_xy, centre, shrink=ENV_SHRINK, cap=MAX_OFFSET)
        lo, hi = np.interp(sc, st, lo_s, period=Lc), np.interp(sc, st, hi_s, period=Lc)
        cap = MAX_OFFSET - SLACK
    # allowed change of the centreline offset (|d| <= cap, never forcing the old line to move); a move a along the
    # line normal changes it by a*cos(heading gap) <= a
    lo_p = np.minimum(np.maximum(lo, -cap) - dc, 0)
    hi_p = np.maximum(np.minimum(hi, cap) - dc, 0)
    new = qp_offsets(base, nrm, w * lo_p, w * hi_p)[0]
    f = np.where(kw >= 0, fac[np.maximum(kw, 0)], 0.0) if len(fac) else np.zeros(len(ls))
    new = base + f[:, None] * (new - base)
    patch = window_weight(ls, tr.length, wins, blend=BLEND + MARGIN)[0] > 0
    return np.where(patch[:, None], new, old_xy), patch_spline(tck, u, new, patch, KNOT_DS / chord)


def remap_s(old_tr, new_tr, s):
    p = [np.interp(s % old_tr.length, old_tr.s, old_tr.xy[:, k], period=old_tr.length) for k in (0, 1)]
    return new_tr.project(np.array(p))[1]


def remap_sections(old_xy, new_xy, sections):
    """[[old line_s start, value], ...] -> [[new line_s start, value], ...] sorted by the new start."""
    to = old_xy if isinstance(old_xy, Track) else Track(old_xy)
    tn = new_xy if isinstance(new_xy, Track) else Track(new_xy)
    rows = sorted((round(float(remap_s(to, tn, s)), 1), s, a) for s, a in sections)
    for s1, s0, a in rows:
        print("  section %7.1f -> %7.1f  a_lat %.3f" % (s0, s1, a))
    return [[s1, a] for s1, _, a in rows]


def main(out, remap_from=None, straights=None, widen=None, factors=(1.0, 1.0, 1.0, 1.0), windows=WINDOWS,
         full_width=False, caps=None):
    centre = np.load(os.path.join(DATA, "centreline.npy"), allow_pickle=False)
    tck = None
    dmax_of = lambda xy: max(np.abs(offsets(xy, centre)[0]).max(),  # noqa: E731
                             np.abs(offsets(Track(xy, tck=tck).xy, centre)[0]).max())
    if widen is not None:
        from envelope import S25_HUD, load_xy
        xy, tck = widen_corners(widen, centre, load_xy(S25_HUD), factors, windows, full_width, caps)
        mode = "widen %s%s" % (",".join("%g" % f for f in factors), " full-width caps %s" % caps if full_width else "")
        dmax = dmax_of(xy)
    elif straights is not None:  # no slack in this mode: the bounds come from the raced line
        xy, mode = smooth_straights(straights, centre), "straights"
        dmax = dmax_of(xy)
    else:
        for slack in (SLACK, 2 * SLACK, 4 * SLACK):  # tighten until the resampled line is within the cap
            xy, mode = build(slack), "slack %.2f" % slack
            dmax = dmax_of(xy)
            if dmax <= MAX_OFFSET:
                break
    if dmax > MAX_OFFSET:
        raise RuntimeError("line exceeds the %.1f m cap: %.3f" % (MAX_OFFSET, dmax))
    tr = Track(xy, ds=1.0, tck=tck)
    print("min-curvature line: %d points, L %.1f m, max|d_centre| %.3f (%s), |k|max %.4f, int k^2 %.4f"
          % (len(xy), tr.length, dmax, mode, np.abs(tr.kappa).max(), (tr.kappa ** 2).sum() * tr.ds))
    if remap_from:
        path = os.path.join(DATA, "vehicle_params.json")
        with open(path) as f:
            p = json.load(f)
        from track import load_track
        p["a_lat_sections"] = remap_sections(load_track(remap_from), tr, p["a_lat_sections"])
        with open(path, "w") as f:
            json.dump(p, f, indent=2)
    if tck is None:
        np.savez(out, xy=xy)
    else:
        np.savez(out, xy=xy, tck_t=tck[0], tck_c=np.array(tck[1]))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--remap-from")
    ap.add_argument("--straights", metavar="OLD_LINE.npz")
    ap.add_argument("--widen", metavar="OLD_LINE.npz")
    ap.add_argument("--factors", default="1,1,1,1")
    ap.add_argument("--max-offset", type=float, default=MAX_OFFSET,
                    help="centreline cap (user lifted the 4.0 m cap 2026-11-04; the proven envelope still bounds it)")
    ap.add_argument("--shrink", type=float, default=ENV_SHRINK)
    ap.add_argument("--all-corners", action="store_true", help="use WINDOWS_ALL (7 corners)")
    ap.add_argument("--full-width", action="store_true", help="bounds = the road edges minus the car, not the envelope")
    ap.add_argument("--caps", help="per-window |d_centre| caps, m (full-width mode)")
    a = ap.parse_args()
    MAX_OFFSET, ENV_SHRINK = a.max_offset, a.shrink
    wins = WINDOWS_ALL if a.all_corners else WINDOWS
    facs = [float(x) for x in a.factors.split(",")]
    if len(facs) != len(wins):
        facs = [1.0] * len(wins)
    caps = [float(x) for x in a.caps.split(",")] if a.caps else None
    main(a.out, a.remap_from, a.straights and np.load(a.straights)["xy"], a.widen and np.load(a.widen)["xy"],
         facs, wins, a.full_width, caps)
