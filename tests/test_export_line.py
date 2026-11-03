import os

import numpy as np
import pytest

import export_line

R = os.path.join(os.path.dirname(__file__), "..", "..", "ref", "roar2026", "competition_code")
REF = os.path.join(R, "WaypointLine.py")
PRIM = os.path.join(R, "waypoints", "waypointsPrimary.npz")
CENTRE = os.path.join(os.path.dirname(__file__), "..", "competition_code", "data", "centreline.npy")


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    if not (os.path.exists(REF) and os.path.exists(PRIM)):
        pytest.skip("reference files not found: %s, %s" % (REF, PRIM))
    out = str(tmp_path_factory.mktemp("l") / "line.npz")
    base = str(tmp_path_factory.mktemp("b") / "base.npz")
    export_line.main(REF, PRIM, out, centre=CENTRE if os.path.exists(CENTRE) else None)
    export_line.main(REF, PRIM, base)
    raw = export_line.parse_literal(REF)
    prim = export_line.clean_primary(export_line.load_primary(PRIM))[0]
    return np.load(out)["xy"], raw, export_line.widen(raw), prim, np.load(base)["xy"]


def test_shape_and_widening(data):
    xy, raw, wid, _, base = data
    assert xy.shape[1] == 2 and 5900 < len(xy) < 6000 and xy.dtype == np.float64
    assert np.array_equal(base[:5701], wid[:5701])
    still = np.all(xy[:5701] == base[:5701], axis=1)  # identical wherever the wall-margin shift is 0
    assert still.any() and np.array_equal(xy[:5701][still], wid[:5701][still])
    d = np.linalg.norm(base[:5701] - raw[:5701], axis=1)
    assert abs(d[1353] - 0.9) < 0.01
    assert d[1313] < 1e-9


def test_closed_loop(data):
    xy = data[0]
    s = np.linalg.norm(np.roll(xy, -1, axis=0) - xy, axis=1)
    assert s.min() >= 0.5 and s.max() <= 1.5


def test_bridge_follows_primary(data):
    xy, prim = data[0][5701:], data[3]
    seg = np.vstack([prim, prim[:1]])  # cleaned primary, closed
    a, b = seg[:-1], np.diff(seg, axis=0)
    for p in xy:
        t = np.clip(((p - a) * b).sum(1) / np.maximum((b * b).sum(1), 1e-12), 0, 1)
        assert np.min(np.linalg.norm(a + t[:, None] * b - p, axis=1)) < 0.75  # ruling said 0.6; spline deviates 0.726 m at the primary seam jog


def test_bridge_smooth(data):
    xy = data[0]
    q = np.vstack([xy[5700 - 30:], xy[:31]])
    a, b, c = q[:-2], q[1:-1], q[2:]
    area2 = np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
    k = 2 * area2 / (np.linalg.norm(b - a, axis=1) * np.linalg.norm(c - b, axis=1) * np.linalg.norm(c - a, axis=1))
    assert k.max() < 0.08
    d = np.diff(q, axis=0)
    h = np.degrees(np.arctan2(d[:, 1], d[:, 0]))
    assert np.abs((np.diff(h) + 180) % 360 - 180).max() < 6


def test_wall_margin(data):
    if not os.path.exists(CENTRE):
        pytest.skip("centreline.npy absent")
    d, _ = export_line.offsets(data[0], np.load(CENTRE))
    assert np.abs(d).max() <= 4.0 + 1e-6


def test_margin_only_moves_near_violations(data):
    if not os.path.exists(CENTRE):
        pytest.skip("centreline.npy absent")
    xy, base = data[0], data[4]
    d0, _ = export_line.offsets(base, np.load(CENTRE))
    step = np.linalg.norm(np.roll(base, -1, axis=0) - base, axis=1)
    arc = np.cumsum(step) - step
    L = step.sum()
    gap = np.abs(arc[:, None] - arc[np.abs(d0) > 4.0][None, :])
    far = np.minimum(gap, L - gap).min(axis=1) > 60
    sel = far & (np.abs(d0) < 3.0)
    assert sel.any()
    assert np.linalg.norm(xy - base, axis=1)[sel].max() < 0.01
