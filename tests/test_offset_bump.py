import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import offset_bump as ob  # noqa: E402


def test_bump_is_full_inside_ramps_outside_and_zero_beyond():
    s = np.arange(0.0, 1000.0, 1.0)
    d = ob.bump(s, np.zeros_like(s), ["400:500:-0.2:20"])
    assert np.allclose(d[400:501], -0.2) and d[300] == 0.0 and d[600] == 0.0
    assert -0.2 < d[390] < 0.0  # on the cosine ramp
    assert np.allclose(ob.bump(s, d, []), d)  # no spec = unchanged
    assert np.allclose(ob.bump(s, np.ones_like(s), ["400:500:0.1"])[450], 1.1)  # adds to the existing offset
