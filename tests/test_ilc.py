import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import ilc  # noqa: E402


def test_smooth_keeps_slow_kills_fast():
    n, ds = 4000, 1.0
    s = np.arange(n) * ds
    slow = np.sin(2 * np.pi * s / 400)
    fast = 0.5 * np.sin(2 * np.pi * s / 4)
    y = ilc.smooth_periodic(slow + fast, ds, 8.0)
    assert np.abs(y - slow).max() < 0.05
    assert abs(ilc.smooth_periodic(np.ones(n), ds, 8.0) - 1).max() < 1e-9  # unit gain, periodic


def test_update_moves_against_error_and_caps():
    n, ds = 4000, 1.0
    s = np.arange(n) * ds
    err = 2.0 * np.sin(2 * np.pi * s / 1000)
    d = ilc.update(np.zeros(n), err, ds, gamma=0.5, sigma=8.0, cap=2.5)
    assert np.abs(d + 0.5 * err).max() < 0.02  # half a step against the error
    assert np.allclose(ilc.update(d, np.zeros(n), ds, sigma=8.0), d, atol=0.02)  # no error -> stays put
    big = ilc.update(np.zeros(n), 20 * err, ds, gamma=0.5, sigma=8.0, cap=2.5)
    assert np.abs(big).max() <= 2.5 + 1e-9


def test_lap_error_uses_flying_laps(tmp_path):
    L, n = 1000.0, 3000
    s_total = np.linspace(0, 3 * L - 1, n)
    line_s = s_total % L
    lap = (s_total // L).astype(int) + 1
    e_y = np.where(lap == 1, 5.0, np.sin(2 * np.pi * line_s / L))  # lap 1 (standing start) differs, ignored
    f = tmp_path / "r.run.csv"
    np.savetxt(f, np.c_[s_total, line_s, e_y], delimiter=",", header="s_total,line_s,e_y", comments="")
    g = np.arange(0, L, 1.0)
    e = ilc.lap_error(str(f), g, L)
    assert np.abs(e - np.sin(2 * np.pi * g / L)).max() < 0.02


def test_target_pulls_in_only_beyond_cap():
    o = np.array([0.0, 3.0, 3.8, 3.95, -3.9, -4.2])
    t = ilc.target(o, cap=3.8)
    assert np.allclose(t, [0, 0, 0, -0.15, 0.1, 0.4])
    assert np.all(np.abs(o + t) <= 3.8 + 1e-12)


def test_window_is_one_inside_zero_outside_and_ramps():
    s = np.arange(0.0, 1000.0, 1.0)
    w = ilc.window(s, 400.0, 600.0)
    assert np.all(w[(s >= 400) & (s <= 600)] == 1.0) and np.all(w[(s <= 360) | (s >= 640)] == 0.0)
    assert w[380] == pytest.approx(0.5)
