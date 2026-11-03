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


def _flat(length, kappa):
    s = np.arange(0.0, length, 1.0)
    return SimpleNamespace(length=float(length), s=s, kappa=np.full_like(s, kappa))


def test_straight_hits_vmax():
    s, v = build_profile(_flat(3000, 0.0), V, 0.0, laps=1)
    vmax = V.p["v_max"]
    assert abs(v.max() - vmax) <= 0.01 * vmax and v.max() <= vmax * 1.001
    dv2 = np.diff(v ** 2) / np.diff(s)
    assert (dv2 <= 2 * np.array([V.a_drive(x) for x in v[:-1]]) + 0.05).all()


def test_circle_lat_limit():
    s, v = build_profile(_flat(2 * np.pi * 50, 0.02), V, 0.0, laps=3)
    lim = np.sqrt(np.array([V.a_lat(x % (2 * np.pi * 50)) for x in s]) * 50)
    assert (v <= lim * 1.001).all() and v.max() > 0.95 * lim.min()


@pytest.fixture(scope="module")
def real():
    tr = Track(np.load(os.path.join(D, "racing_line.npz"))["xy"])
    start = tr.project(np.load(os.path.join(D, "centreline.npy"))[0])[1]
    s, v = build_profile(tr, V, start)
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
