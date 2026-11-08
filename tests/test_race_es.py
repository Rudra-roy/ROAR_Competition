import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "rl"))
import race_es  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "competition_code", "data")


def test_flat_unflat_round_trip_touches_only_trained_nets():
    res = dict(np.load(os.path.join(DATA, "residual.npz")))
    nets = [0, 2]
    th = race_es.flat(res, nets)
    back = race_es.unflat(th + 1.0, res, nets)
    for k in race_es.KEYS:
        assert np.allclose(back[k][0], np.asarray(res[k][0]) + 1.0)
        assert np.array_equal(back[k][1], res[k][1])  # untrained net unchanged
    assert np.array_equal(back["cuts"], res["cuts"])


def test_segment_times_and_ranks(tmp_path):
    cuts = np.array([0.0, 100.0, 300.0])
    s = np.arange(0, 1200, 1.0)  # 2 laps of a 600 m loop, 1 m per tick
    f = tmp_path / "r.run.csv"
    np.savetxt(f, np.c_[s, s % 600], delimiter=",", header="s_total,line_s", comments="")
    seg, clean = race_es.segment_times(str(f), cuts, laps_m=1200.0)
    assert clean and np.allclose(seg, np.array([100, 200, 300]) * 2 * race_es.DT)
    seg, clean = race_es.segment_times(str(f), cuts, laps_m=1800.0)  # stopped before the race distance
    assert not clean
    assert list(race_es.centred_ranks([3.0, 1.0, 2.0])) == [-0.5, 0.5, 0.0]  # lowest cost ranks best
