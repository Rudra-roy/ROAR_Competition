import os

import numpy as np
import pytest

import export_line
import mincurv_line
from export_line import KINK_THR, kink_metric, offsets
from track import Track

CENTRE = os.path.join(os.path.dirname(__file__), "..", "competition_code", "data", "centreline.npy")
K2_8D = 0.2189  # integral of kappa^2 ds of the Task 8d line (Track(xy, ds=1), analysis/path_study summary)
R = os.path.join(os.path.dirname(__file__), "..", "..", "ref", "roar2026", "competition_code")


@pytest.fixture(scope="module")
def old_line(tmp_path_factory):
    ref, prim = os.path.join(R, "WaypointLine.py"), os.path.join(R, "waypoints", "waypointsPrimary.npz")
    if not (os.path.exists(ref) and os.path.exists(prim)):
        pytest.skip("reference files not found: %s" % R)
    out = str(tmp_path_factory.mktemp("o") / "line8d.npz")
    export_line.main(ref, prim, out, centre=CENTRE)  # the Task 8d line, reproduced
    return np.load(out)["xy"]


@pytest.fixture(scope="module", params=["free", "straights"])
def line(request, tmp_path_factory, old_line):
    if not os.path.exists(mincurv_line.WP):
        pytest.skip("official waypoints not found: %s" % mincurv_line.WP)
    out = str(tmp_path_factory.mktemp("m") / "line.npz")
    mincurv_line.main(out, straights=old_line if request.param == "straights" else None)
    return np.load(out)["xy"]


def test_straights_mode_keeps_corners_and_smooths_straights(tmp_path, old_line):
    out = str(tmp_path / "s.npz")
    mincurv_line.main(out, straights=old_line)
    new = np.load(out)["xy"]
    w, tr, idx = mincurv_line.straight_weight(old_line)
    dc = offsets(old_line, np.load(CENTRE))[0]
    move = np.linalg.norm(new - old_line, axis=1)
    fixed = (w == 0) & (np.abs(dc) < mincurv_line.WALL_IN)
    assert fixed.mean() > 0.2 and move[fixed].max() < 1e-6  # corners exactly as raced
    assert move.max() <= mincurv_line.STRAIGHT_DELTA + 1e-6
    sgn = lambda k: int(np.sum(np.sign(k[1:]) != np.sign(k[:-1])))
    st = w > 0.99  # straight points, in line order (new[i] is old_line[i] moved along its normal)
    tn = Track(new)
    k_new = np.array([tn.kappa[tn.project(p)[0]] for p in new[st]])
    assert sgn(k_new) < 0.5 * sgn(tr.kappa[idx[st]])  # the curvature wiggle on the straights


def test_closed_loop(line):
    step = np.linalg.norm(np.diff(np.vstack([line, line[:1]]), axis=0), axis=1)
    assert step.max() < 2.5 and step.max() / np.median(step) < 1.5  # incl. the closing segment last -> first


def test_within_offset_cap(line):
    c = np.load(CENTRE)
    assert np.abs(offsets(line, c)[0]).max() <= 4.0
    assert np.abs(offsets(Track(line).xy, c)[0]).max() <= 4.0  # the resampled line the car follows


def test_lower_curvature_than_8d_line(line):
    tr = Track(line, ds=1.0)
    assert float((tr.kappa ** 2).sum() * tr.ds) < K2_8D


def test_no_kinks(line):
    assert np.abs(kink_metric(Track(line))).max() < KINK_THR


def test_remap_sections_follows_xy():
    """A section start keyed by s on one line maps to the s of the same place on another line."""
    t = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    old = np.c_[100 * np.cos(t), 100 * np.sin(t)]
    new = np.roll(np.c_[103 * np.cos(t), 103 * np.sin(t)], -100, axis=0)  # wider, origin a quarter turn later
    secs = [[50.0, 20.0], [300.0, 25.0], [500.0, 30.0]]
    out = mincurv_line.remap_sections(old, new, secs)
    to = Track(old)
    tn = Track(new)
    for (s0, a0), (s1, a1) in zip(sorted(secs, key=lambda r: mincurv_line.remap_s(to, tn, r[0])), out):
        assert a0 == a1
        p = to.xy[int(round(s0 / to.ds)) % len(to.s)]
        assert abs(tn.project(p)[1] - s1) < 1.0
    assert [r[0] for r in out] == sorted(r[0] for r in out)


@pytest.fixture(scope="module")
def widened(old_line, tmp_path_factory):
    """(8e line, Track of the corner-widened line (its own spline), its points). The 8e line = straights mode on the
    8d line (byte-identical to the shipped 8e/e115 racing_line.npz)."""
    import envelope
    if not (os.path.exists(envelope.S25_HUD) and os.path.exists(mincurv_line.WP)):
        pytest.skip("S25 HUD log or waypoints not found")
    from track import load_track
    d = tmp_path_factory.mktemp("w")
    mincurv_line.main(str(d / "e8.npz"), straights=old_line)
    e8 = np.load(d / "e8.npz")["xy"]
    mincurv_line.main(str(d / "w.npz"), widen=e8, factors=mincurv_line.TEST_FACTORS)
    return e8, load_track(str(d / "w.npz")), np.load(d / "w.npz")["xy"]


def _on(tn, to, sel):
    """New-line kappa and |offset| at the old line's samples sel (same geometric place, interpolated in s)."""
    pr = [tn.project(p) for p in to.xy[sel]]
    s = np.array([q[1] for q in pr])
    return np.interp(s, tn.s, tn.kappa, period=tn.length), np.abs([q[2] for q in pr])


def _active():
    return [wi for wi, f in zip(mincurv_line.WINDOWS, mincurv_line.TEST_FACTORS) if f > 0]


def test_widen_pins_the_line_outside_the_windows(widened):
    """Outside the windows (+ blend) the resampled line the car follows is the old one to 1 cm, and no corner's
    kappa moves by more than 5 % (Track's global spline coupling, Task 8e: up to 80 % with a plain refit)."""
    e8, tn, _ = widened
    to = Track(e8)
    out = mincurv_line.window_weight(to.s, to.length, windows=_active())[0] == 0
    assert 0.5 < out.mean() < 0.95
    k, d = _on(tn, to, out)
    assert d.max() <= 0.01
    k0 = to.kappa[out]
    corner = np.abs(k0) >= 0.005  # R <= 200 m
    assert corner.sum() > 500
    assert np.all(np.abs(k - k0)[corner] <= 0.05 * np.abs(k0[corner]))
    assert np.abs(k - k0)[~corner].max() <= 5e-4


def test_widen_raises_min_radius_in_every_window(widened):
    e8, tn, _ = widened
    to = Track(e8)
    s_new_on_old = np.array([to.project(p)[1] for p in tn.xy])
    for a, b in _active():
        r_old = 1 / np.abs(to.kappa[(to.s >= a) & (to.s <= b)]).max()
        r_new = 1 / np.abs(tn.kappa[(s_new_on_old >= a) & (s_new_on_old <= b)]).max()
        assert r_new > r_old + 1.0, (a, b, r_old, r_new)


def test_widen_respects_the_envelope_and_cap(widened):
    import envelope
    e8, tn, xy = widened
    c = np.load(CENTRE)
    st, lo, hi, _, _ = envelope.envelope(e8, envelope.load_xy(envelope.S25_HUD), c)
    L = np.linalg.norm(np.roll(c, -1, 0) - c, axis=1).sum()
    so, do = envelope.station(Track(e8).xy, c)  # the old line as the car follows it (its raw points carry +-6 cm noise)
    o = np.argsort(so)
    d_old = np.interp(st, so[o], do[o], period=L)
    lo, hi = np.minimum(lo, d_old), np.maximum(hi, d_old)
    s, d = envelope.station(tn.xy, c)
    tol = 0.03  # linear interpolation between the 2 m stations in the tightest corners
    assert np.all(d >= np.interp(s, st, lo, period=L) - tol) and np.all(d <= np.interp(s, st, hi, period=L) + tol)
    assert np.abs(offsets(tn.xy, c)[0]).max() <= 4.0 and np.abs(offsets(xy, c)[0]).max() <= 4.0


def test_widen_closed_and_no_kinks(widened):
    _, tn, _ = widened
    assert np.linalg.norm(tn.xy[0] - tn.xy[-1]) < 1.5 * tn.ds
    assert np.abs(kink_metric(tn)).max() < KINK_THR


def test_shipped_line_is_the_tool_output(tmp_path):
    """competition_code/data/racing_line.npz is two tool steps away from the e115 line:
    1. tools/line_w5.npz == mincurv_line --widen tools/line_e115.npz --factors SHIP_FACTORS (the w5 line);
    2. shipped == mincurv_line --widen tools/line_w5.npz --windows SHIP_EXIT_WINDOWS --exit-weight SHIP_EXIT_WEIGHT
       --winner (the kink exit straightened, its stored spline kept)."""
    import envelope
    for f in (envelope.S25_HUD, envelope.W26_HUD):
        if not os.path.exists(f):
            pytest.skip("%s not found" % f)
    ml = mincurv_line
    same = lambda a, b: sorted(a.files) == sorted(b.files) and all(np.array_equal(a[k], b[k]) for k in a.files)  # noqa
    saved = ml.MAX_OFFSET, ml.ENV_SHRINK
    ml.MAX_OFFSET, ml.ENV_SHRINK = ml.SHIP_MAX_OFFSET, ml.SHIP_SHRINK
    try:
        w5 = str(tmp_path / "w5.npz")
        ml.main(w5, widen=np.load(envelope.LINE_8E)["xy"], factors=ml.SHIP_FACTORS)
        assert same(np.load(w5), np.load(os.path.join(ml.ROOT, "tools", "line_w5.npz")))
        d = np.load(w5)
        out = str(tmp_path / "s.npz")
        wins = [(a, b) for a, _, b in ml.SHIP_EXIT_WINDOWS]
        ml.main(out, widen=d["xy"], factors=[1.0] * len(wins), windows=wins, widen_tck=(d["tck_t"], list(d["tck_c"]), 3),
                exit_w=ml.SHIP_EXIT_WEIGHT, apexes=[p for _, p, _ in ml.SHIP_EXIT_WINDOWS], winner=True)
    finally:
        ml.MAX_OFFSET, ml.ENV_SHRINK = saved
    assert same(np.load(out), np.load(os.path.join(os.path.dirname(CENTRE), "racing_line.npz")))


def test_exit_ramp_weights_rise_from_apex_to_exit():
    L = 1000.0
    ls = np.arange(0.0, L, 1.0)
    kw = mincurv_line.exit_ramp(ls, L, [(100.0, 300.0)], [4.0], apexes=[200.0], blend=50.0)
    assert np.all(kw[(ls >= 100) & (ls <= 200)] == 1.0)  # entry to apex: plain min-curvature
    assert kw[250] == pytest.approx(3.0) and kw[300] == pytest.approx(5.0)  # linear to 1 + alpha at the exit
    assert np.all(kw[(ls > 300) & (ls <= 350)] == 5.0)  # held over the blend after the window
    assert np.all(kw[(ls > 350) | (ls < 100)] == 1.0)


def test_full_width_widening_respects_caps_and_moves_only_windows():
    import pytest
    from envelope import S25_HUD as _hud
    if not os.path.exists(_hud):
        pytest.skip("S25 HUD log not found")
    import mincurv_line as ml
    from envelope import S25_HUD, load_xy
    from export_line import offsets
    centre = np.load(os.path.join(ml.DATA, "centreline.npy"), allow_pickle=False)
    old = np.load(os.path.join(ml.ROOT, "tools", "line_e115.npz"))["xy"]
    caps = [4.4, 4.0, 4.6, 4.2, 4.6, 4.6, 4.6]
    xy, tck = ml.widen_corners(old, centre, load_xy(S25_HUD), [1.0] * 7, ml.WINDOWS_ALL, True, caps)
    tr = ml.Track(xy, tck=tck)
    d = offsets(tr.xy, centre)[0]
    w, k = ml.window_weight(tr.s, tr.length, ml.WINDOWS_ALL, blend=ml.BLEND + ml.MARGIN)
    for j, c in enumerate(caps):
        inside = (k == j) & (w > 0.99)
        if inside.any():
            assert np.abs(d[inside]).max() <= c + 0.05, (j, np.abs(d[inside]).max())
    told = ml.Track(old)
    far = w == 0
    moved = np.array([abs(told.project(p)[2]) for p in tr.xy[far][::25]])
    assert moved.max() < 0.02  # untouched outside the windows
