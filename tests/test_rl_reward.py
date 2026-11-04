import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "rl"))
import train_es as es  # noqa: E402


def test_reward_grades_crashes_by_progress_and_severity():
    R = es.Reward([10.0, 20.0], pen=5.0)
    clean = {"times": [10.0, 20.0], "coll": [False, False], "impulse": [0, 0], "frac": [1, 1]}
    assert R(clean) == -30.0
    late = {"times": [10.0, 18.0], "coll": [False, True], "impulse": [0, 500], "frac": [1, 0.9]}
    early = {"times": [10.0, 4.0], "coll": [False, True], "impulse": [0, 500], "frac": [1, 0.2]}
    hard = {"times": [10.0, 18.0], "coll": [False, True], "impulse": [0, 4000], "frac": [1, 0.9]}
    assert R(late) > R(early)  # crashing later is better
    assert R(late) > R(hard)  # a scrape costs less than a hard hit
    assert R(clean) > R(late)


def test_penalty_adapts_to_crash_rate_and_maturity():
    R = es.Reward([10.0], pen=5.0)
    t0 = R.target()
    R.adapt(0.9, improving=False)
    assert R.pen > 5.0 and R.gen == 0
    p = R.pen
    R.adapt(0.0, improving=True)
    assert R.pen < p and R.gen == 1
    for _ in range(100):
        R.adapt(0.0, improving=True)
    assert R.target() < t0 and R.pen >= es.COLL_PEN_MIN
    for _ in range(100):
        R.adapt(1.0, improving=False)
    assert R.pen == es.COLL_PEN_MAX


def test_per_segment_reward_sums_to_total():
    R = es.Reward([10.0, 20.0], pen=5.0)
    r = {"times": [10.0, 18.0], "coll": [False, True], "impulse": [0, 500], "frac": [1, 0.9]}
    import numpy as np
    assert np.isclose(R.per_segment(r).sum(), R(r)) and R.per_segment(r)[0] == -10.0
