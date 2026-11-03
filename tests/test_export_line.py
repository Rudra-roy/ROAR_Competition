import os

import numpy as np
import pytest

import export_line

REF = os.path.join(os.path.dirname(__file__), "..", "..", "ref", "roar2026",
                   "competition_code", "WaypointLine.py")


def test_shape_and_widening(tmp_path):
    if not os.path.exists(REF):
        pytest.skip("reference WaypointLine.py not found at %s" % REF)
    raw = export_line.parse_literal(REF)
    out = tmp_path / "line.npz"
    export_line.main(REF, str(out))
    xy = np.load(out)["xy"]
    assert xy.shape == (5865, 2) and xy.dtype == np.float64
    assert tuple(xy[-1]) == (-281.61450497288973, 393.17536918350305)
    d = np.linalg.norm(xy - raw, axis=1)
    assert abs(d[1353] - 0.9) < 0.01
    assert d[1313] < 1e-9
