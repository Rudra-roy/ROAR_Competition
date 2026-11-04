import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import fit_scrub  # noqa: E402


def test_fit_recovers_quadratic_then_linear():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 33, 20000)
    y = fit_scrub.model(x, 0.005, 0.7) + rng.normal(0, 0.3, len(x))
    k2, slope, _, _ = fit_scrub.fit(x, y)
    assert abs(k2 - 0.005) < 0.0005
    assert abs(slope - 0.7) < 0.15
    s = fit_scrub.model(np.array(fit_scrub.BREAKS), k2, slope)
    assert np.all(np.diff(np.diff(s) / np.diff(fit_scrub.BREAKS)) >= -1e-9)  # convex breakpoints
