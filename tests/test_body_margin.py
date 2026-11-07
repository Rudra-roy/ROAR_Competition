import glob
import os

import numpy as np
import pytest

import body_margin as bm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = sorted(glob.glob(os.path.join(ROOT, "logs", "ref", "*.run.csv")))


def test_corner_offsets_of_a_straight_box():
    c = np.c_[np.linspace(0, 1000, 1001), np.zeros(1001)]  # centreline along +x, left = +y
    s, lo, hi = bm.corner_offsets(np.array([[500.0, 1.0]]), np.array([0.0]), c)
    assert s[0] == pytest.approx(500.0) and lo[0] == pytest.approx(1 - bm.HALF_W) and hi[0] == pytest.approx(1 + bm.HALF_W)
    _, lo, hi = bm.corner_offsets(np.array([[500.0, 0.0]]), np.array([np.pi / 2]), c)  # sideways: length spans y
    assert lo[0] == pytest.approx(-bm.HALF_L) and hi[0] == pytest.approx(bm.HALF_L)


@pytest.mark.skipif(not REF, reason="needs reference race logs in logs/ref/")
def test_reference_is_within_itself_and_a_shift_shows(tmp_path):
    m = bm.BodyMargin(REF[:1])
    per, _ = m.excess(REF[0])
    assert per.max() < 1e-9
    d = np.genfromtxt(REF[0], delimiter=",", names=True)
    d["e_y"] += 0.3
    f = tmp_path / "shift.run.csv"
    np.savetxt(f, d, delimiter=",", header=",".join(d.dtype.names), comments="", fmt="%.6g")
    per, _ = m.excess(str(f))
    assert 0.25 < per.max() < 0.35
