import os
import numpy as np
from vehicle import Vehicle

P = os.path.join(os.path.dirname(__file__), "..", "competition_code", "data", "vehicle_params.json")
V = Vehicle.from_json(P)


def test_left_curvature_gives_negative_steer():
    assert V.steer_for(0.01, 30) < 0


def test_pedals_exclusive_and_saturate():
    assert V.pedals_for(100, 30) == (1.0, 0.0)
    assert V.pedals_for(-100, 30) == (0.0, 1.0)
    assert V.pedals_for(0.96 * V.a_drive(30), 30)[0] == 1.0


def test_a_lat_section_lookup():
    s = V.p["a_lat_sections"]
    assert V.a_lat(s[0][0]) == s[0][1]
    assert V.a_lat(s[1][0] - 0.1) == s[0][1]
    assert V.a_lat(s[1][0]) == s[1][1]
    assert V.a_lat(s[0][0] - 1) == s[-1][1]  # wrap
    assert V.a_lat(0.0) == s[-1][1]


def test_k_continuous_and_positive():
    vs = np.arange(1, 80, 0.01)
    k = np.array([V.K(v) for v in vs])
    assert (k > 0).all() and np.abs(np.diff(k)).max() < 0.01


def test_a_drive_matches_json():
    assert abs(V.a_drive(20) - 3.64) < 0.05
    assert abs(V.a_drive(V.p["v_max"])) < 0.05
