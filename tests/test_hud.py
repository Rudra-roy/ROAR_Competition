import os
import statistics
import time
from types import SimpleNamespace as NS

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import numpy as np
import pygame
import pytest
from PIL import Image

import hud

D = os.path.join(os.path.dirname(__file__), "..", "competition_code", "data")
CL = np.load(os.path.join(D, "centreline.npy"))


class FakeActor:
    """CARLA-frame state: x east, y south (ROAR y = -CARLA y), yaw deg clockwise from above."""

    def __init__(self, x=0.0, y=0.0, yaw=0.0, vx=0.0, vy=0.0, thr=0.0, brk=0.0, steer=0.0, role="hero"):
        self.set(x, y, yaw, vx, vy, thr, brk, steer)
        self.attributes = {"role_name": role}

    def set(self, x, y, yaw=0.0, vx=0.0, vy=0.0, thr=0.0, brk=0.0, steer=0.0):
        self.p = (x, y, yaw, vx, vy, thr, brk, steer)

    def get_velocity(self):
        return NS(x=self.p[3], y=self.p[4], z=0.0)

    def get_transform(self):
        return NS(location=NS(x=self.p[0], y=self.p[1], z=0.0), rotation=NS(yaw=self.p[2], pitch=0.0, roll=0.0))

    def get_control(self):
        return NS(throttle=self.p[5], brake=self.p[6], steer=self.p[7])


class FakeClient:
    def __init__(self, actor, t=[0.0]):
        self.actor, self.t = actor, t

    def set_timeout(self, s):
        self.timeout = s

    def get_world(self):
        a, t = self.actor, self.t
        return NS(
            get_actors=lambda: NS(filter=lambda pat: [a]),
            get_snapshot=lambda: NS(timestamp=NS(elapsed_seconds=t[0])),
        )


def make(actor=None, **kw):
    actor = actor or FakeActor()
    return hud.Hud(client_factory=lambda: FakeClient(actor), **kw), actor


def test_polyline_laps_and_progress():
    xy = CL * [1, 1]
    pl = hud.Polyline(xy)
    lt = hud.LapTracker(pl)
    N = len(xy)
    steps = int(3.2 * N)
    prog, lap_at = [], {}
    for i in range(steps + 1):
        total, lap, pct = lt.update(xy[i % N])
        prog.append(pct)
        lap_at.setdefault(lap, i)
    assert lap_at == {0: 0, 1: N, 2: 2 * N, 3: 3 * N}
    assert all(b >= a - 1e-9 for a, b in zip(prog, prog[1:]))
    assert prog[3 * N] == pytest.approx(100.0, abs=0.01) and prog[-1] == 100.0
    assert prog[N // 2] == pytest.approx(100 / 6, abs=0.5)


def test_carla_to_roar_flip():
    assert hud.carla_to_roar(3.0, 4.0) == (3.0, -4.0)
    a = FakeActor()
    h, _ = make(a)
    i = 100
    x, ry = CL[i]
    a.set(x, -ry)  # CARLA y = -ROAR y
    s = h.sample()
    assert s["idx"] == i


def test_beta_sign_and_zero():
    assert hud.sideslip_deg(10, 0, 0.0) == pytest.approx(0, abs=1e-9)
    assert hud.sideslip_deg(0, 10, 90.0) == pytest.approx(0, abs=1e-9)  # yaw 90 = +y direction
    v, ang = 20.0, np.radians(10)
    # velocity 10 deg to the LEFT of the nose (counter-clockwise from above = negative CARLA angle)
    assert hud.sideslip_deg(v * np.cos(-ang), v * np.sin(-ang), 0.0) == pytest.approx(-10, abs=1e-6)
    assert hud.sideslip_deg(v * np.cos(ang), v * np.sin(ang), 0.0) == pytest.approx(10, abs=1e-6)
    assert hud.sideslip_deg(0.0, 0.0, 33.0) == 0.0


def test_flags_sliding_and_below_plan():
    a = FakeActor(vx=50.0, vy=50 * np.tan(np.radians(6)))
    h, _ = make(a)
    a.set(*CL[500] * [1, -1], yaw=0.0, vx=50.0, vy=50 * np.tan(np.radians(6)))
    s = h.sample()
    assert s["sliding"]
    a.set(*CL[500] * [1, -1], yaw=0.0, vx=50.0, vy=0.0)
    assert not h.sample()["sliding"]
    if h.plan is not None:  # on-line, far below plan, coasting
        s = h.sample()
        if s["plan"] is not None and s["plan"] * 3.6 > s["v"] * 3.6 + hud.PLAN_TOL_KMH:
            assert s["below"]


def test_hides_plan_when_unmappable(tmp_path):
    a = FakeActor()
    np.save(tmp_path / "centreline.npy", CL)  # only the centreline: no profile / racing line
    h = hud.Hud(client_factory=lambda: FakeClient(a), data_dir=str(tmp_path))
    a.set(*CL[10] * [1, -1], vx=30.0)
    assert h.sample()["plan"] is None
    h2, a2 = make()
    a2.set(5000.0, 5000.0, vx=30.0)  # nowhere near the racing line
    assert h2.sample()["plan"] is None


def _surface():
    pygame.init()
    return pygame.display.set_mode((1024, 600))


def test_render_time_and_no_exception(tmp_path):
    h, a = make()
    scr = _surface()
    scr.fill((90, 90, 90))
    n = len(CL)
    times = []
    for k in range(200):
        i = (k * 3) % n
        a.set(*CL[i] * [1, -1], yaw=10.0, vx=40.0, vy=3.0, thr=0.5, steer=0.2)
        t0 = time.perf_counter()
        h.frame(scr)
        times.append(time.perf_counter() - t0)
    assert not h.disabled
    assert statistics.median(times) < 0.005, statistics.median(times)


def test_failure_isolation(capsys):
    def boom():
        raise RuntimeError("no server")

    h = hud.Hud(client_factory=boom)
    scr = _surface()
    h.frame(scr)  # must not raise
    h.frame(scr)
    assert h.disabled
    assert capsys.readouterr().err.count("HUD disabled") == 1


def test_install_preserves_return(monkeypatch):
    import infrastructure

    orig = infrastructure.ManualControlViewer.render
    monkeypatch.setattr(infrastructure.ManualControlViewer, "render", orig)  # restores after the test
    boom = hud.Hud(client_factory=lambda: (_ for _ in ()).throw(RuntimeError("x")))
    hud.install(boom)
    assert infrastructure.ManualControlViewer.render is not orig
    img = NS(get_image=lambda: Image.new("RGB", (320, 200), (90, 90, 90)))
    flips = []
    real_flip = pygame.display.flip
    v1, v2 = infrastructure.ManualControlViewer(), infrastructure.ManualControlViewer()
    r_new = v1.render(img)
    assert pygame.display.flip is real_flip  # restored after the call
    pygame.quit()
    hud.uninstall()
    assert infrastructure.ManualControlViewer.render is orig
    r_old = v2.render(img)
    pygame.quit()
    assert r_new == r_old


def _walk_racing_line(h, a, frac, n=120, stride=8):
    """Drive the racing line from the start line at frac * the plan speed; returns the last sample."""
    rl = h.line_tr.line
    pr = h.plan
    i0 = int(np.argmin(np.abs(rl.s - pr[2])))
    for k in range(n):
        x, y = rl.xy[(i0 + k * stride) % len(rl.xy)]
        v = max(frac * float(np.interp(k * stride * 0.94, pr[0], pr[1])), 20.0)
        a.set(x, -y, vx=v, thr=0.3)
        s = h.sample()
    return s


def test_plan_mapping_and_below_flag():
    h, a = make()
    s = _walk_racing_line(h, a, 0.6)
    assert s["plan"] is not None and s["below"] and not s["over"]
    h2, a2 = make()
    s2 = _walk_racing_line(h2, a2, 1.0)
    assert not s2["below"]
    s3 = _walk_racing_line(*make(), 1.5)
    assert s3["over"]
