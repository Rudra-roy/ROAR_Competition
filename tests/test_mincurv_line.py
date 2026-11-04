import os

import numpy as np
import pytest

import export_line
import mincurv_line
from export_line import KINK_THR, kink_metric, offsets
from track import Track

CENTRE = os.path.join(os.path.dirname(__file__), "..", "competition_code", "data", "centreline.npy")
K2_8D = 0.2189  # integral of kappa^2 ds of the Task 8d line (Track(xy, ds=1), analysis/path_study summary)
R = os.path.join(os.path.dirname(__file__), "..", "..", "ref", "roar2026", "competition_code")


@pytest.fixture(scope="module")
def old_line(tmp_path_factory):
    ref, prim = os.path.join(R, "WaypointLine.py"), os.path.join(R, "waypoints", "waypointsPrimary.npz")
    if not (os.path.exists(ref) and os.path.exists(prim)):
        pytest.skip("reference files not found: %s" % R)
    out = str(tmp_path_factory.mktemp("o") / "line8d.npz")
    export_line.main(ref, prim, out, centre=CENTRE)  # the Task 8d line, reproduced
    return np.load(out)["xy"]


@pytest.fixture(scope="module", params=["free", "straights"])
def line(request, tmp_path_factory, old_line):
    if not os.path.exists(mincurv_line.WP):
        pytest.skip("official waypoints not found: %s" % mincurv_line.WP)
    out = str(tmp_path_factory.mktemp("m") / "line.npz")
    mincurv_line.main(out, straights=old_line if request.param == "straights" else None)
    return np.load(out)["xy"]


def test_straights_mode_keeps_corners_and_smooths_straights(tmp_path, old_line):
    out = str(tmp_path / "s.npz")
    mincurv_line.main(out, straights=old_line)
    new = np.load(out)["xy"]
    w, tr, idx = mincurv_line.straight_weight(old_line)
    dc = offsets(old_line, np.load(CENTRE))[0]
    move = np.linalg.norm(new - old_line, axis=1)
    fixed = (w == 0) & (np.abs(dc) < mincurv_line.WALL_IN)
    assert fixed.mean() > 0.2 and move[fixed].max() < 1e-6  # corners exactly as raced
    assert move.max() <= mincurv_line.STRAIGHT_DELTA + 1e-6
    sgn = lambda k: int(np.sum(np.sign(k[1:]) != np.sign(k[:-1])))
    st = w > 0.99  # straight points, in line order (new[i] is old_line[i] moved along its normal)
    tn = Track(new)
    k_new = np.array([tn.kappa[tn.project(p)[0]] for p in new[st]])
    assert sgn(k_new) < 0.5 * sgn(tr.kappa[idx[st]])  # the curvature wiggle on the straights


def test_closed_loop(line):
    step = np.linalg.norm(np.diff(np.vstack([line, line[:1]]), axis=0), axis=1)
    assert step.max() < 2.5 and step.max() / np.median(step) < 1.5  # incl. the closing segment last -> first


def test_within_offset_cap(line):
    c = np.load(CENTRE)
    assert np.abs(offsets(line, c)[0]).max() <= 4.0
    assert np.abs(offsets(Track(line).xy, c)[0]).max() <= 4.0  # the resampled line the car follows


def test_lower_curvature_than_8d_line(line):
    tr = Track(line, ds=1.0)
    assert float((tr.kappa ** 2).sum() * tr.ds) < K2_8D


def test_no_kinks(line):
    assert np.abs(kink_metric(Track(line))).max() < KINK_THR


def test_remap_sections_follows_xy():
    """A section start keyed by s on one line maps to the s of the same place on another line."""
    t = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    old = np.c_[100 * np.cos(t), 100 * np.sin(t)]
    new = np.roll(np.c_[103 * np.cos(t), 103 * np.sin(t)], -100, axis=0)  # wider, origin a quarter turn later
    secs = [[50.0, 20.0], [300.0, 25.0], [500.0, 30.0]]
    out = mincurv_line.remap_sections(old, new, secs)
    to = Track(old)
    tn = Track(new)
    for (s0, a0), (s1, a1) in zip(sorted(secs, key=lambda r: mincurv_line.remap_s(to, tn, r[0])), out):
        assert a0 == a1
        p = to.xy[int(round(s0 / to.ds)) % len(to.s)]
        assert abs(tn.project(p)[1] - s1) < 1.0
    assert [r[0] for r in out] == sorted(r[0] for r in out)
