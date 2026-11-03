import os

import numpy as np
import pytest

import export_line

R = os.path.join(os.path.dirname(__file__), "..", "..", "ref", "roar2026", "competition_code")
REF = os.path.join(R, "WaypointLine.py")
PRIM = os.path.join(R, "waypoints", "waypointsPrimary.npz")


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    if not (os.path.exists(REF) and os.path.exists(PRIM)):
        pytest.skip("reference files not found: %s, %s" % (REF, PRIM))
    out = str(tmp_path_factory.mktemp("l") / "line.npz")
    export_line.main(REF, PRIM, out)
    raw = export_line.parse_literal(REF)
    prim = export_line.load_primary(PRIM)
    return np.load(out)["xy"], raw, export_line.widen(raw), prim


def test_shape_and_widening(data):
    xy, raw, wid, _ = data
    assert xy.shape[1] == 2 and 5900 < len(xy) < 6000 and xy.dtype == np.float64
    assert np.array_equal(xy[5700], wid[5700])
    d = np.linalg.norm(xy[:5701] - raw[:5701], axis=1)
    assert abs(d[1353] - 0.9) < 0.01
    assert d[1313] < 1e-9


def test_closed_loop(data):
    xy = data[0]
    s = np.linalg.norm(np.roll(xy, -1, axis=0) - xy, axis=1)
    assert s.min() >= 0.5 and s.max() <= 1.5


def test_bridge_follows_primary(data):
    xy, prim = data[0][5701:], data[3]
    seg = prim
    a, b = seg[:-1], np.diff(seg, axis=0)
    for p in xy:
        t = np.clip(((p - a) * b).sum(1) / np.maximum((b * b).sum(1), 1e-12), 0, 1)
        assert np.min(np.linalg.norm(a + t[:, None] * b - p, axis=1)) < 0.3
