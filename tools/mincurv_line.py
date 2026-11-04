"""Minimum-curvature racing line -> competition_code/data/racing_line.npz.

  python tools/mincurv_line.py competition_code/data/racing_line.npz [--straights OLD.npz] [--remap-from OLD.npz]

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
from track import Track  # noqa: E402
from scipy.ndimage import gaussian_filter1d  # noqa: E402

WP = os.path.join(os.path.dirname(ROOT), "ROAR_PY", "roar_py_carla", "roar_py_carla", "assets", "waypoints", "Monza.npz")
DATA = os.path.join(ROOT, "competition_code", "data")
HALF_W = 1.0817  # Tesla Model 3 bbox half-width (sim)
YAW_ALLOW = 0.3  # body corner reach at ~0.1 rad heading + tracking slop
MAX_OFFSET = 4.0  # same wall cap as the export_line line
SLACK = 0.05  # QP bound below the cap: smoothed vs polyline centreline + the runtime Track resampling
DS = 2.0
STRAIGHT_DELTA = 0.5  # m, max lateral move of the raced line on its straights
STRAIGHT_KAPPA = 0.002  # 1/m, |kappa| (15 m smoothed) below which the raced line counts as straight
WALL_OUTER = 3.5  # m |d_centre|: no move toward the near wall from here out (ramped over 0.5 m)
WALL_IN = 3.95  # m |d_centre|: raced-line points beyond are pulled in to it (cap 4.0 incl. the Track resampling)
ITERS = 4


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
        s = osqp.OSQP()
        s.setup(P=sp.triu(A.T @ A).tocsc(), q=A.T @ b, A=sp.identity(n, format="csc"), l=lo, u=hi, verbose=False,
                eps_abs=1e-7, eps_rel=1e-7, max_iter=200000, polish=True)
        a = np.clip(s.solve(raise_error=False).x, lo, hi)
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
    no move toward the near wall beyond WALL_OUTER, points beyond WALL_IN pulled in to it. Removes the raced line's
    curvature wiggle on the straights (the weave the MPC follows, Task 8e) and leaves its corners as raced."""
    w, tr, idx = straight_weight(old_xy)
    nrm = np.c_[-np.sin(tr.psi[idx]), np.cos(tr.psi[idx])]  # left normal, same sense as the centreline's
    dc = offsets(old_xy, centre)[0]
    out = delta * w * np.clip((WALL_OUTER - np.abs(dc)) / 0.5, 0, 1) + np.minimum(WALL_IN - np.abs(dc), 0)
    lo = np.where(dc > 0, -delta * w, -out)  # the near wall is on the +n side when dc > 0
    hi = np.where(dc > 0, out, delta * w)
    lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
    return qp_offsets(old_xy, nrm, lo, hi)[0]


def remap_s(old_tr, new_tr, s):
    p = [np.interp(s % old_tr.length, old_tr.s, old_tr.xy[:, k], period=old_tr.length) for k in (0, 1)]
    return new_tr.project(np.array(p))[1]


def remap_sections(old_xy, new_xy, sections):
    """[[old line_s start, value], ...] -> [[new line_s start, value], ...] sorted by the new start."""
    to, tn = Track(old_xy), Track(new_xy)
    rows = sorted((round(float(remap_s(to, tn, s)), 1), s, a) for s, a in sections)
    for s1, s0, a in rows:
        print("  section %7.1f -> %7.1f  a_lat %.3f" % (s0, s1, a))
    return [[s1, a] for s1, _, a in rows]


def main(out, remap_from=None, straights=None):
    centre = np.load(os.path.join(DATA, "centreline.npy"), allow_pickle=False)
    for slack in (SLACK, 2 * SLACK, 4 * SLACK):  # tighten until the resampled line is within the cap
        xy = build(slack) if straights is None else smooth_straights(straights, centre)
        dmax = max(np.abs(offsets(xy, centre)[0]).max(), np.abs(offsets(Track(xy).xy, centre)[0]).max())
        if dmax <= MAX_OFFSET:
            break
        if straights is not None:
            raise RuntimeError("line exceeds the %.1f m cap: %.3f" % (MAX_OFFSET, dmax))
    else:
        raise RuntimeError("line exceeds the %.1f m cap: %.3f" % (MAX_OFFSET, dmax))
    tr = Track(xy, ds=1.0)
    print("min-curvature line: %d points, L %.1f m, max|d_centre| %.3f (slack %.2f), |k|max %.4f, int k^2 %.4f"
          % (len(xy), tr.length, dmax, slack, np.abs(tr.kappa).max(), (tr.kappa ** 2).sum() * tr.ds))
    if remap_from:
        path = os.path.join(DATA, "vehicle_params.json")
        with open(path) as f:
            p = json.load(f)
        p["a_lat_sections"] = remap_sections(np.load(remap_from)["xy"], xy, p["a_lat_sections"])
        with open(path, "w") as f:
            json.dump(p, f, indent=2)
    np.savez(out, xy=xy)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--remap-from")
    ap.add_argument("--straights", metavar="OLD_LINE.npz")
    a = ap.parse_args()
    main(a.out, a.remap_from, a.straights and np.load(a.straights)["xy"])
