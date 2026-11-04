"""Iterative learning of the lateral reference offset (Task 8i).

The sim is deterministic (run-to-run e_y(s) p95 ~0.09 m vs the error itself p95 ~1.7 m), so the car's lateral error
along the line is a repeatable function of line_s. Each clean race updates the offset the MPC tracks:

    d_{k+1}(s) = d_k(s) - gamma * G_sigma[e_y,k - t](s)    (e_y = driven path minus racing line, + = left)

so the driven path converges onto the target t(s): the racing line, pulled in toward the centreline wherever the line
itself is more than --target-cap from it (wall margin: a car exactly on a 3.9x m line still shows ~4.0 m by noise). The runtime (competition_code/submission.py) reads
competition_code/data/line_offset.npz {s, d}; no file = plain racing line.

Usage: python tools/ilc.py OUT.npz RUN.csv [RUN.csv ...] [--prev PREV.npz] [--gamma 0.5] [--sigma 8] [--cap 2.5]
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "competition_code"))
from track import load_track  # noqa: E402

LINE = os.path.join(ROOT, "competition_code", "data", "racing_line.npz")
CENTRE = os.path.join(ROOT, "competition_code", "data", "centreline.npy")
TARGET_CAP = 4.60  # m, max |offset from centreline| the driven path is steered to
FLYING_LAPS = (2, 3)


def smooth_periodic(x, ds, sigma):
    """Zero-phase Gaussian low-pass of a periodic signal sampled every ds metres."""
    if sigma <= 0:
        return np.asarray(x, float).copy()
    n = len(x)
    f = np.fft.rfftfreq(n, d=ds)
    return np.fft.irfft(np.fft.rfft(x) * np.exp(-0.5 * (2 * np.pi * f * sigma) ** 2), n)


def lap_error(run_csv, s_grid, length, laps=FLYING_LAPS):
    """Mean e_y(s) over the given laps of one race, on s_grid."""
    d = np.genfromtxt(run_csv, delimiter=",", names=True)
    out = []
    for lap in laps:
        m = (d["s_total"] >= (lap - 1) * length) & (d["s_total"] < lap * length)
        if m.sum() < 100:
            continue
        ls, ey = d["line_s"][m], d["e_y"][m]
        o = np.argsort(ls)
        out.append(np.interp(s_grid, ls[o], ey[o], period=length))
    if not out:
        raise ValueError("%s: no complete laps among %s" % (run_csv, laps))
    return np.mean(out, axis=0)


def target(line_offset, cap=TARGET_CAP):
    """Lateral target relative to the line: move toward the centreline by the amount the line exceeds cap."""
    o = np.asarray(line_offset, float)
    return -np.sign(o) * np.maximum(np.abs(o) - cap, 0.0)


def update(prev_d, err, ds, gamma=0.5, sigma=8.0, cap=2.5):
    d = np.asarray(prev_d, float) - gamma * smooth_periodic(err, ds, sigma)
    return smooth_periodic(np.clip(d, -cap, cap), ds, sigma / 2)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--prev")
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--sigma", type=float, default=8.0)
    ap.add_argument("--cap", type=float, default=2.5)
    ap.add_argument("--target-cap", type=float, default=TARGET_CAP)
    a = ap.parse_args(argv)
    tr = load_track(LINE)
    prev = np.zeros(len(tr.s))
    if a.prev:
        f = np.load(a.prev)
        prev = np.interp(tr.s, f["s"], f["d"], period=tr.length)
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    from mincurv_line import signed_offset
    t = smooth_periodic(target(signed_offset(tr.xy, np.load(CENTRE)[:, :2]), a.target_cap), tr.ds, a.sigma)
    err = np.mean([lap_error(r, tr.s, tr.length) for r in a.runs], axis=0) - t
    d = update(prev, err, tr.ds, a.gamma, a.sigma, a.cap)
    d2 = (np.roll(d, -1) - 2 * d + np.roll(d, 1)) / tr.ds ** 2
    np.savez(a.out, s=tr.s, d=d)
    print("e_y in: rms %.3f p95 %.3f max %.3f m | d out: max|d| %.2f m, max|d''| %.4f 1/m (line max|kappa| %.4f)"
          % (np.sqrt((err ** 2).mean()), np.percentile(np.abs(err), 95), np.abs(err).max(), np.abs(d).max(),
             np.abs(d2).max(), np.abs(tr.kappa).max()))
    return d


if __name__ == "__main__":
    main()
