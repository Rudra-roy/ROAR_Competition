import os
from types import SimpleNamespace

import numpy as np
import pytest

from build_profile import build_profile, lap_times
from profile_oracle import oracle_profile
from track import Track
from vehicle import Vehicle

D = os.path.join(os.path.dirname(__file__), "..", "competition_code", "data")
V = Vehicle.from_json(os.path.join(D, "vehicle_params.json"))
BOX = Vehicle({k: x for k, x in V.p.items() if k != "brake_ellipse_c"})  # the oracle knows only the box


def test_brake_ellipse_bounds_combined_grip():
    """Braking while cornering: (a_long/A_brake)^2 + (a_lat/(c*A_lat))^2 <= 1 (polygon tangent to the ellipse)."""
    c = 1.0
    veh = Vehicle(dict(BOX.p, brake_ellipse_c=c))
    L = 2000.0
    s0 = np.arange(0.0, L, 1.0)
    kap = np.where((s0 > 900) & (s0 < 1100), 0.02, 0.0)  # straight -> 200 m arc -> straight
    tr = SimpleNamespace(length=L, s=s0, kappa=kap)
    s, v = build_profile(tr, veh, 0.0, laps=1)
    _, vb = build_profile(tr, BOX, 0.0, laps=1)
    a_long = np.diff(v ** 2) / (2 * np.diff(s))
    vm = 0.5 * (v[1:] + v[:-1])
    k = np.interp(0.5 * (s[1:] + s[:-1]), s0, kap)
    al = np.array([veh.a_lat(x) for x in s[1:]])
    brk = a_long < -0.5
    use = (a_long / (veh.a_brake * veh.brake_scale)) ** 2 + (vm ** 2 * k / (c * al)) ** 2
    assert brk.any() and use[brk].max() <= 1.05
    arc = (s > 950) & (s < 1050)
    assert (v <= vb + 1e-6).all() and v[arc].min() > 0.95 * vb[arc].min()


def test_brake_ellipse_c_below_one_rejected():
    """c < 1 is not a pure braking knob (the 75 deg facet also caps pure lateral grip): reject it."""
    tr = SimpleNamespace(length=200.0, s=np.arange(0.0, 200.0, 1.0), kappa=np.zeros(200))
    with pytest.raises(ValueError):
        build_profile(tr, Vehicle(dict(BOX.p, brake_ellipse_c=0.9)), 0.0, laps=1)


def _flat(length, kappa):
    s = np.arange(0.0, length, 1.0)
    return SimpleNamespace(length=float(length), s=s, kappa=np.full_like(s, kappa))


def test_straight_hits_vmax():
    s, v = build_profile(_flat(3000, 0.0), BOX, 0.0, laps=1)
    vmax = V.p["v_max"]
    assert abs(v.max() - vmax) <= 0.01 * vmax and v.max() <= vmax * 1.001
    dv2 = np.diff(v ** 2) / np.diff(s)
    assert (dv2 <= 2 * np.array([V.a_drive(x) for x in v[:-1]]) + 0.05).all()


def test_circle_lat_limit():
    s, v = build_profile(_flat(2 * np.pi * 50, 0.02), BOX, 0.0, laps=3)
    lim = np.sqrt(np.array([V.a_lat(x % (2 * np.pi * 50)) for x in s]) * 50)
    assert (v <= lim * 1.001).all() and v.max() > 0.95 * lim.min()


@pytest.fixture(scope="module")
def real():
    tr = Track(np.load(os.path.join(D, "racing_line.npz"))["xy"])
    start = tr.project(np.load(os.path.join(D, "centreline.npy"))[0])[1]
    s, v = build_profile(tr, BOX, start)
    return tr, start, s, v


def test_matches_oracle(real):
    tr, start, s, v = real
    L = tr.length
    so, vo = oracle_profile(
        lambda q: np.interp((start + q) % L, tr.s, tr.kappa, period=L),
        lambda q: V.a_lat((start + q) % L),
        V.a_drive, V.a_brake * V.brake_scale, V.p["v_max"], 3 * L, s[1] - s[0])
    assert len(so) == len(s)
    rel = np.abs(v[1:] - vo[1:]) / vo[1:]
    assert np.mean(rel <= 0.02) >= 0.95


def test_predicted_flying_lap_plausible(real):
    tr, _, s, v = real
    t = lap_times(s, v, tr.length)
    assert len(t) == 3 and 95 <= t[1] <= 125
