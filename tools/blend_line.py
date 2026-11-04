"""Blend a donor path into our racing line outside chosen windows (keep OUR line inside them).

  python tools/blend_line.py OUT.npz OURS.npz DONOR.npy --keep a:b[,a:b...] [--blend 60] [--sigma 0.06]

Frenet blend on our line: donor offset e(s) from our line (projection), w(s) = 1 inside a keep window with cosine
ramps of --blend m, result = our line + (1 - w) e. Used for the s25b line (2026-11-04): ROAR_1_S25's driven path
everywhere except the chicane, where our car collided 0.8 m inside it (an unmapped obstacle; our line is proven there).
"""
import argparse
import os
import sys

import numpy as np
from scipy.interpolate import splev

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "competition_code"))
from track import Track, fit_spline, load_track  # noqa: E402


def keep_weight(s, L, windows, blend):
    w = np.zeros_like(s)
    for a, b in windows:
        x = (s - a) % L  # distance past the window start, periodic
        n = (b - a) % L
        inside = x <= n
        before = np.minimum((a - s) % L, L)  # distance before the window start
        after = np.where(inside, 0.0, x - n)
        d = np.where(inside, 0.0, np.minimum(before, after))
        w = np.maximum(w, np.where(d < blend, 0.5 * (1 + np.cos(np.pi * d / blend)), 0.0))
    return w


def blend(ours, donor_xy, windows, blend_m=60.0, sigma=0.06):
    e = np.array([ours.project(p) for p in donor_xy], dtype=object)
    s, ey = np.array([q[1] for q in e], float), np.array([q[2] for q in e], float)
    o = np.argsort(s)
    E = np.interp(ours.s, s[o], ey[o], period=ours.length)
    w = keep_weight(ours.s, ours.length, windows, blend_m)
    nrm = np.c_[-np.sin(ours.psi), np.cos(ours.psi)]
    xy = ours.xy + ((1 - w) * E)[:, None] * nrm
    tck = fit_spline(xy, sigma=sigma)
    return np.array(splev(np.linspace(0, 1, len(xy) + 1)[:-1], tck)).T, tck, w, E


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("ours"); ap.add_argument("donor")
    ap.add_argument("--keep", default=""); ap.add_argument("--blend", type=float, default=60.0)
    ap.add_argument("--sigma", type=float, default=0.06)
    a = ap.parse_args()
    wins = [tuple(float(x) for x in k.split(":")) for k in a.keep.split(",") if k]
    xy, tck, w, E = blend(load_track(a.ours), np.load(a.donor), wins, a.blend, a.sigma)
    np.savez(a.out, xy=xy, tck_t=tck[0], tck_c=np.array(tck[1]))
    t = Track(xy, tck=tck)
    print("blend: kept ours on %.0f m, donor |e| max %.2f m; L %.1f max|k| %.4f" % (
        (w > 0.99).sum() * load_track(a.ours).ds, np.abs(E).max(), t.length, np.abs(t.kappa).max()))
