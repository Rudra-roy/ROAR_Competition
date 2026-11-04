import numpy as np
import pytest

from corner_compare import corner_metrics, path_a_lat


def test_path_a_lat_on_a_circle():
    """Constant 20 m/s on a radius-50 arc: v^2 kappa = 8 m/s^2 (sign: + left / CCW)."""
    dt, v, r = 0.05, 20.0, 50.0
    th = np.arange(400) * v * dt / r
    xy = r * np.c_[np.cos(th), np.sin(th)]
    a = path_a_lat(xy, np.full(400, v))
    assert np.median(a[20:-20]) == pytest.approx(8.0, rel=0.02)


def test_corner_metrics_brake_min_exit():
    """Straight 60 m/s, full brake from s=100 at 15 m/s^2 down to 30 m/s, part throttle, full throttle from s=300."""
    dt = 0.05
    s, v, t = [0.0], [60.0], [0.0]
    thr, brk = [], []
    while s[-1] < 500:
        if s[-1] < 100:
            a, th, b = 0.0, 1.0, 0.0
        elif v[-1] > 30.0 and s[-1] < 300:
            a, th, b = -15.0, 0.0, 1.0
        elif s[-1] < 300:
            a, th, b = 0.0, 0.3, 0.0
        else:
            a, th, b = 3.0, 1.0, 0.0
        thr.append(th), brk.append(b)
        v.append(max(v[-1] + a * dt, 0.0)), s.append(s[-1] + v[-2] * dt), t.append(t[-1] + dt)
    n = len(thr)
    D = dict(s=np.array(s[:n]), v=np.array(v[:n]), t=np.array(t[:n]), thr=np.array(thr), brk=np.array(brk),
             a_lat=np.zeros(n), plan=np.full(n, np.nan))
    m = corner_metrics(D, 0.0, 500.0)
    assert m["lift_s"] == pytest.approx(100.0, abs=3.0)
    assert m["decel_max"] == pytest.approx(15.0, abs=0.5)
    assert m["v_min"] == pytest.approx(30.0, abs=0.5)
    assert m["exit_thr_s"] == pytest.approx(300.0, abs=3.0)
    assert m["time"] == pytest.approx(D["t"][-1], abs=0.1)


def test_section_grip_percentile():
    """Per a_lat section (line_s ranges mapped to centreline metres): p-th percentile of |a_lat| on the flying laps
    (v > 15 m/s). Sections wrap (last section runs into the first)."""
    from corner_compare import section_grip
    L = 1000.0
    s = np.arange(0.0, 3 * L, 1.0)
    a = np.where((s % L) < 400, 10.0, 20.0)
    a[(s % L) > 900] = 30.0
    D = dict(s=s, a_lat=a, v=np.full_like(s, 30.0), L=L)
    g = section_grip(D, [0.0, 400.0, 900.0], p=95)
    assert g == pytest.approx([10.0, 20.0, 30.0])
