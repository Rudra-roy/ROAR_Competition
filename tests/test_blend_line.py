import os
import sys

import numpy as np

sys.path[:0] = [os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), d) for d in ("tools", "competition_code")]
import blend_line  # noqa: E402
from track import Track  # noqa: E402


def test_keep_weight_window_and_wrap():
    L = 1000.0
    s = np.arange(0, L, 1.0)
    w = blend_line.keep_weight(s, L, [(950.0, 30.0)], 20.0)  # window wraps the start
    assert w[960] == 1 and w[10] == 1 and w[30] == 1
    assert w[500] == 0 and w[60] == 0 and w[920] == 0
    assert 0 < w[40] < 1 and 0 < w[940] < 1


def test_blend_keeps_ours_in_window_and_follows_donor_outside():
    t = np.linspace(0, 2 * np.pi, 2000, endpoint=False)
    ours = Track(np.c_[300 * np.cos(t), 300 * np.sin(t)])
    donor = np.c_[302 * np.cos(t), 302 * np.sin(t)]  # 2 m outside everywhere
    xy, tck, w, E = blend_line.blend(ours, donor, [(100.0, 300.0)], 60.0, sigma=0.015)
    r = np.hypot(*xy.T)
    tr = Track(xy, tck=tck)
    s = np.array([ours.project(p)[1] for p in xy])
    assert np.abs(r[(s > 120) & (s < 280)] - 300).max() < 0.05
    assert np.abs(r[(s > 500) & (s < 1500)] - 302).max() < 0.05
    assert np.isfinite(tr.kappa).all()
