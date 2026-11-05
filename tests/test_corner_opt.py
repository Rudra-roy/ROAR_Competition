import json
import os

import numpy as np

import corner_opt as co

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def params():
    return json.load(open(os.path.join(ROOT, "competition_code", "data", "vehicle_params.json")))


def test_knobs_round_trip_and_bounds():
    p = params()
    X = co.knobs_of(p)
    assert X.shape == (len(p["a_lat_sections"]), 3)
    X2 = X.copy()
    X2[3] = [1.7, 0.2, 50.0]  # out of bounds
    q = co.apply(p, X2)
    assert q["corner_throttle_floor"]["thr_sec"][3] == 1.0
    assert q["brake_sec_scale"][3] == 0.8
    assert q["a_lat_sections"][3][1] == 31.0
    assert np.allclose(co.knobs_of(co.apply(p, X)), X)


def test_race_result_on_logged_runs():
    import pytest
    for f in ("gate_t_1.run.csv", "bs125_abort.run.csv"):
        if not os.path.exists(os.path.join(ROOT, "logs", f)):
            pytest.skip("race logs not present (logs/ is not in git)")
    p = params()
    clean = co.race_result(os.path.join(ROOT, "logs", "gate_t_1.run.csv"), p)
    assert clean is not None
    t_sec, coll, ok, total = clean
    assert ok and coll == -1 and abs(t_sec.sum() - total) < 0.5 and 330 < total < 335
    ab = co.race_result(os.path.join(ROOT, "logs", "bs125_abort.run.csv"), p)
    assert ab is not None and not ab[2] and ab[1] == 8  # spun at c3835 = section 8


def test_corner_mask_marks_corner_sections_only():
    m = co.corner_mask(params())
    assert m[0] and m[10] and m[7]  # chicane, hairpin, bottom-curve entry
    assert not m[9]  # the main straight


def test_grid_overrides():
    import grid_race as gr
    p = params()
    q = gr.apply_overrides(p, "brake_scale=1.2 sec3=25 brk7=1.3 ctf4=0.9 mpc.q_v=3")
    assert q["brake_scale"] == 1.2 and q["a_lat_sections"][3][1] == 25.0
    assert q["brake_sec_scale"][7] == 1.3 and q["corner_throttle_floor"]["thr_sec"][4] == 0.9
    assert q["mpc"]["q_v"] == 3.0
    assert gr.apply_overrides(p, "") == p
