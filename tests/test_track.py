import os
import numpy as np
import pytest
from track import Track, Progress

NPZ = os.path.join(os.path.dirname(__file__), "..", "competition_code", "data", "racing_line.npz")


@pytest.fixture(scope="module")
def real():
    xy = np.load(NPZ)["xy"]
    return xy, Track(xy)


def test_circle_curvature():
    t = np.linspace(0, 2 * np.pi, 2000, endpoint=False)
    tr = Track(np.c_[50 * np.cos(t), 50 * np.sin(t)])
    assert np.allclose(tr.kappa, 0.02, rtol=0.03)
    assert abs(tr.length - 2 * np.pi * 50) < 0.005 * 2 * np.pi * 50


def test_curvature_smooth_and_line_preserved(real):
    xy, tr = real
    from scipy.spatial import cKDTree
    f = np.linspace(0, 1, 21)[:-1]
    nx = np.roll(tr.xy, -1, 0)
    dense = (tr.xy[:, None] * (1 - f[None, :, None]) + nx[:, None] * f[None, :, None]).reshape(-1, 2)
    dd = cKDTree(dense).query(xy)[0]
    assert dd.max() < 0.15
    assert np.abs(tr.kappa).max() < 0.08
    assert np.abs(np.diff(np.r_[tr.kappa, tr.kappa[0]])).max() < 0.014


def test_project_on_line(real):
    xy, tr = real
    rng = np.random.RandomState(0)
    for i in rng.randint(0, len(tr.s), 50):
        j = (i + 1) % len(tr.s)
        mid = (tr.xy[i] + tr.xy[j]) / 2  # not a resampled sample
        s_mid = tr.s[i] + tr.ds / 2
        for hint in (None, int(i)):
            idx, s, ey = tr.project(mid, hint)
            ds = abs((s - s_mid + tr.length / 2) % tr.length - tr.length / 2)
            assert ds < 0.5 and abs(ey) < 0.05
        n = np.array([-np.sin(tr.psi[i]), np.cos(tr.psi[i])])
        assert tr.project(mid + 2 * n, int(i))[2] == pytest.approx(2.0, abs=0.05)
        assert tr.project(mid - 1 * n, None)[2] == pytest.approx(-1.0, abs=0.05)


def test_progress_unwraps_across_seam(real):
    xy, tr = real
    pg = Progress(tr)
    n = len(tr.s)
    st = []
    for k in range(int(1.5 * n)):
        i = k % n
        st.append(pg.update(tr.xy[i], tr.psi[i])[0])
    st = np.array(st)
    assert np.all(np.diff(st) > -0.01)
    assert abs(st[-1] - 1.5 * tr.length) < 2 + tr.length / n
    assert pg.start_s == pytest.approx(tr.s[0], abs=1.0)


def test_progress_fast_weave_never_backwards(real):
    xy, tr = real
    pg = Progress(tr)
    st = []
    for k in range(int(1.2 * tr.length / 3.6)):
        s = (k * 3.6) % tr.length
        i = int(round(s / tr.ds)) % len(tr.s)
        nrm = np.array([-np.sin(tr.psi[i]), np.cos(tr.psi[i])])
        p = tr.xy[i] + 1.5 * np.sin(k / 5.0) * nrm
        st.append(pg.update(p, tr.psi[i])[0])
    st = np.array(st)
    assert np.all(np.diff(st) > 0)
    assert abs(st[-1] - (len(st) - 1) * 3.6) < 3


def test_heading_error_wrap_boundary(real):
    xy, tr = real
    assert tr.heading_error(0, tr.psi[0] + np.pi) == pytest.approx(np.pi, abs=1e-9)
    assert tr.heading_error(0, tr.psi[0] + np.pi) > 0


def test_relocalize_near_spawn(real):
    xy, tr = real
    pg = Progress(tr)
    m = len(tr.s)
    for i in range(m - 14):
        end = pg.update(tr.xy[i], tr.psi[i])[0]
    assert abs(end - (tr.length - 15)) < 3
    s_tot, _, _, idx = pg.update(tr.xy[0], tr.psi[0])
    assert abs(s_tot) < 2 and idx in (0, 1, m - 1)
    assert pg.start_s == pytest.approx(0.0, abs=1.0)


def test_relocalize_after_teleport(real):
    xy, tr = real
    pg = Progress(tr)
    start = 100
    for k in range(start, start + 300):
        pg.update(tr.xy[k], tr.psi[k])
    s_tot, ey, epsi, idx = pg.update(tr.xy[start], tr.psi[start])
    assert abs(s_tot) < 2
    assert abs(idx - start) <= 2
    assert pg.start_s == pytest.approx(tr.s[start], abs=1.0)
    s2 = pg.update(tr.xy[start + 1], tr.psi[start + 1])[0]
    assert 0 < s2 < 3
