import numpy as np

from envelope import envelope, outside


def _circle(r, d, n=3000, laps=1.0, phase=0.0):
    th = phase + np.linspace(0, 2 * np.pi * laps, int(n * laps), endpoint=False)
    rr = r + np.broadcast_to(d(th % (2 * np.pi)) if callable(d) else d, th.shape)
    return np.c_[rr * np.cos(th), rr * np.sin(th)]


R = 200.0
CENTRE = _circle(R, 0.0, n=600)  # CCW: left (+d) = inward = smaller radius, so d = R - r


def test_envelope_bounds():
    """Our line at d +1 all round. S25 (3 laps, CCW, different sampling): d -3 on th < pi (it is the extreme on the
    right there), d +2.5 on th >= pi except lap 2 at +3.2 near th = 1.5 pi, and +5 on a short arc (cap)."""
    ours = _circle(R, -1.0)  # r = R - d
    lap_d = lambda extra: (lambda th: -np.where(th < np.pi, -3.0, np.where(np.abs(th - 1.5 * np.pi) < 0.1,
                                                                         2.5 + extra, 2.5)))
    laps = [_circle(R, lambda th: np.where(np.abs(th - 1.0) < 0.05, -5.0, lap_d(0.0)(th)), n=2300, phase=0.3),
            _circle(R, lap_d(0.7), n=2500), _circle(R, lap_d(0.0), n=2100, phase=-0.2)]
    s25 = np.vstack(laps)
    st, lo, hi, lo_raw, hi_raw = envelope(ours, s25, CENTRE, shrink=0.3, cap=4.0)
    th = st / (2 * np.pi * R) * 2 * np.pi
    first = (th > 0.2) & (th < np.pi - 0.2) & (np.abs(th - 1.0) > 0.1)
    assert np.allclose(lo_raw[first], -3.0, atol=0.05) and np.allclose(lo[first], -2.7, atol=0.05)
    assert np.allclose(hi[first], 1.0, atol=0.05)  # our line is the extreme there: not shrunk
    second = (th > np.pi + 0.2) & (th < 2 * np.pi - 0.2) & (np.abs(th - 1.5 * np.pi) > 0.15)
    assert np.allclose(lo[second], 1.0, atol=0.05) and np.allclose(hi[second], 2.2, atol=0.05)
    peak = np.abs(th - 1.5 * np.pi) < 0.05  # lap 2 alone went to 3.2: max over laps
    assert np.allclose(hi_raw[peak], 3.2, atol=0.05) and np.allclose(hi[peak], 2.9, atol=0.05)
    arc = np.abs(th - 1.0) < 0.02
    assert np.all(hi[arc] <= 4.0 + 1e-9) and np.all(hi_raw[arc] > 4.5)  # |offset| <= 4 everywhere
    assert np.all(lo <= hi)


def test_envelope_keeps_our_line_inside():
    """S25 just 0.1 m outside our line: shrinking S25's side by 0.3 must not cut our own line out."""
    ours = _circle(R, -1.0)
    s25 = _circle(R, -1.1, n=2000, laps=1.1)
    _, lo, hi, _, _ = envelope(ours, s25, CENTRE)
    assert np.all(lo <= 1.0 + 0.02) and np.all(hi >= 1.0 - 0.02) and np.all(hi < 1.05)


def test_envelope_s25_on_the_centre_side_is_not_moved():
    """Our line at d +3, S25 at +1 (between us and the centreline): S25's bound stays at +1 (moving it toward the
    centreline would widen it); S25 at -0.1 moves only to 0 (not past the centreline)."""
    ours = _circle(R, -3.0)
    _, lo, hi, _, _ = envelope(ours, _circle(R, -1.0, n=2000, laps=1.1), CENTRE)
    assert np.allclose(lo, 1.0, atol=0.03) and np.allclose(hi, 3.0, atol=0.03)
    _, lo, _, _, _ = envelope(ours, _circle(R, 0.1, n=2000, laps=1.1), CENTRE)
    assert np.allclose(lo, 0.0, atol=0.03)


def test_outside_reports_car_beyond_the_raw_envelope():
    ours = _circle(R, -1.0)
    s25 = _circle(R, 2.0, n=2000, laps=1.1)  # d -2
    st, lo, hi, lo_raw, hi_raw = envelope(ours, s25, CENTRE)
    car = _circle(R, lambda th: np.where(np.abs(th - 2.0) < 0.02, -1.5, 1.5), n=2500)  # d -1.5, +1.5 on a short arc
    ex = outside(car, st, lo_raw, hi_raw, CENTRE)
    assert abs(ex.max() - 0.5) < 0.05  # 0.5 m beyond our line (the left extreme) on that arc
    assert np.mean(ex > 0.01) < 0.02


def test_check_bounds_union_of_driven_paths_widened_to_the_centre_side():
    """Car rule: never further out than our line, S25 or our e115 car went; the side facing the centreline gets 0.3 m
    (not past the centreline). Ours +1, S25 +2.5, car +1.5 -> [0.7, 2.5]; ours -3, S25 -2, car -2.5 -> [-3, -1.7];
    ours +0.1, S25 +1 -> [0, 1]."""
    from envelope import check_bounds
    for d_ours, d_s25, d_car, want in ((1.0, 2.5, 1.5, (0.7, 2.5)), (-3.0, -2.0, -2.5, (-3.0, -1.7)),
                                       (0.1, 1.0, 0.5, (0.0, 1.0))):
        st, lo, hi = check_bounds(_circle(R, -d_ours), [_circle(R, -d_s25, n=2000, laps=1.1),
                                                        _circle(R, -d_car, n=2500, laps=1.1)], CENTRE)
        assert np.allclose(lo, want[0], atol=0.03) and np.allclose(hi, want[1], atol=0.03)
