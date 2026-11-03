import os
import numpy as np
from vehicle import Vehicle

P = os.path.join(os.path.dirname(__file__), "..", "competition_code", "data", "vehicle_params.json")
V = Vehicle.from_json(P)


def test_left_curvature_gives_negative_steer():
    assert V.steer_for(0.01, 30) < 0


LEGACY = Vehicle({k: x for k, x in V.p.items() if k != "pedal_map"})
PM = V.p["pedal_map"]


def a_table(thr, brk, v):
    """Forward pedal map: bilinear on the json tables (independent of Vehicle's inverse)."""
    if brk > 0:
        return float(np.interp(brk, PM["brk"], [np.interp(v, PM["v"], c) for c in np.array(PM["a_brk"]).T]))
    return float(np.interp(thr, PM["thr"], [np.interp(v, PM["v"], c) for c in np.array(PM["a_thr"]).T]))


def test_pedals_exclusive_and_saturate():
    for veh in (V, LEGACY):
        assert veh.pedals_for(100, 30) == (1.0, 0.0)
        assert veh.pedals_for(-100, 30) == (0.0, 1.0)
        assert veh.pedals_for(veh.a_drive(30), 30)[0] == 1.0
        for a in np.linspace(-30, 10, 81):
            th, br = veh.pedals_for(a, 45.0)
            assert th * br == 0 and 0 <= th <= 1 and 0 <= br <= 1
    assert LEGACY.pedals_for(0.96 * LEGACY.a_drive(30), 30)[0] == 1.0


def test_legacy_pedals_unchanged():
    """No "pedal_map" in the json: the old linear maps, bit for bit."""
    for v in (0.5, 10.0, 30.0, 55.0, 70.0):
        ad = LEGACY.a_drive(v)
        for a in np.linspace(-20, 5, 101):
            if a >= 0:
                th = a / ad
                exp = (1.0 if th >= 0.95 else float(np.clip(th, 0, 1))), 0.0
            else:
                exp = 0.0, float(np.clip(-a / (LEGACY.a_brake * LEGACY.brake_scale), 0, 1))
            assert LEGACY.pedals_for(a, v) == exp


def test_pedal_map_round_trip():
    """Inside the achievable range the tabulated accel of the chosen pedals is a_cmd within 0.3 m/s^2."""
    for v in (20.0, 35.0, 45.0, 55.0, 62.0, 68.0):
        lo, hi = a_table(0, 1, v), a_table(1, 0, v)
        for a in np.linspace(lo, hi, 60):
            th, br = V.pedals_for(a, v)
            assert abs(a_table(th, br, v) - a) < 0.3, (v, a, th, br)


def test_mild_decel_at_speed_lifts_not_brakes():
    """The old linear map turned a_cmd -1 at 55 m/s into brake (real ~-13.5 m/s^2); the measured one uses throttle."""
    th, br = V.pedals_for(-1.0, 55.0)
    assert br == 0 and 0.6 < th < 1.0
    assert LEGACY.pedals_for(-1.0, 55.0)[1] > 0


def test_former_gap_averages_to_a_cmd():
    """a_cmd inside the old 'gap' (-11 ... -0.4 m/s^2 at 55 m/s): 20 ticks of commands average to a_cmd."""
    for a in np.linspace(-11, -0.4, 12):
        acc = [a_table(*V.pedals_for(a, 55.0), 55.0) for _ in range(20)]
        assert abs(np.mean(acc) - a) < 0.5


def test_pedals_monotone_in_a_cmd():
    for v in (5.0, 25.0, 45.0, 55.0, 65.0, 75.0):
        u = [th - br for th, br in (V.pedals_for(a, v) for a in np.linspace(-30, 6, 400))]
        assert (np.diff(u) >= 0).all()


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
    assert abs(V.a_drive(20) - 3.65) < 0.05  # fit_pedals refit, steady full throttle
    assert abs(V.a_drive(V.p["v_max"])) < 0.05


def test_load_profile_interp_scale_clamp(tmp_path):
    from vehicle import load_profile
    f = tmp_path / "p.npz"
    np.savez(f, s=np.array([0.0, 10.0, 20.0]), v=np.array([0.0, 20.0, 30.0]), start_s=5.0)
    vr = load_profile(str(f), speed_scale=0.5)
    assert abs(vr(5.0) - 5.0) < 1e-9
    assert abs(vr(15.0) - 12.5) < 1e-9
    assert vr(1e6) == 15.0
    assert load_profile(str(f))(10.0) == 20.0
