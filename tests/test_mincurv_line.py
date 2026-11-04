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
