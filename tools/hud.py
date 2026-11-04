"""Dev-only live race HUD for the viewer window (never imported by competition_code/).

    import hud; hud.install()        # before the first viewer.render()

Wraps infrastructure.ManualControlViewer.render and draws onto the viewer's screen just before the original
pygame.display.flip(). Reads the car through a SECOND, read-only carla client, so it works for any controller.
Any HUD exception prints one line to stderr and disables the HUD; the race goes on.
"""
import csv
import json
import math
import os
import sys

import numpy as np
import pygame

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "competition_code", "data")

# --- thresholds / knobs --------------------------------------------------------------------------------------------
SLIDE_BETA_DEG = 4.0      # |beta| above this ...
SLIDE_MIN_V = 15.0        # ... at more than this speed (m/s) => SLIDING
PLAN_TOL_KMH = 8.0        # BELOW / OVER PLAN margin
BELOW_MAX_THROTTLE = 0.9  # BELOW PLAN only if the driver is not already flat out
FLAG_MIN_V = 10.0         # no plan flags below this speed (m/s): the profile starts from rest
PLAN_MAX_OFFLINE_M = 15.0  # farther than this from the racing line => plan not mappable, hidden
SPEED_MAX = 75.0          # m/s, full scale of the speed bar / trail colours
LAPS = 3
START_V = 0.5             # m/s: the race clock starts the first frame the car moves
TELEPORT_M = 30.0         # per-frame jump treated as a respawn (ignored by the unwrapped progress)
BETA_FULL_DEG = 15.0      # sideslip gauge full scale
MINIMAP_PX = 220


def carla_to_roar(x, y):
    """ROAR frame is y-flipped relative to CARLA."""
    return x, -y


def sideslip_deg(vx, vy, yaw_deg):
    """Angle from the nose to the velocity vector, CARLA frame (yaw clockwise from above, y to the right).
    beta > 0: velocity points to the RIGHT of the nose; beta < 0: to the left. 0 if (nearly) stationary."""
    if math.hypot(vx, vy) < 1.0:
        return 0.0
    d = math.degrees(math.atan2(vy, vx)) - yaw_deg
    return (d + 180.0) % 360.0 - 180.0


class Polyline:
    """Closed polyline with arc-length s and nearest-point projection (windowed around a hint, full as fallback)."""

    WIN = 150  # vertices either side of the hint

    def __init__(self, xy):
        self.xy = np.asarray(xy, dtype=float)[:, :2]
        nxt = np.roll(self.xy, -1, axis=0)
        seg = nxt - self.xy
        el = np.hypot(seg[:, 0], seg[:, 1])
        self.s = np.r_[0.0, np.cumsum(el)[:-1]]
        self.length = float(el.sum())
        self.tan = seg / np.maximum(el, 1e-9)[:, None]
        self.el = el

    def project(self, p, hint=None):
        """-> (idx, s, distance)"""
        n = len(self.xy)
        p = np.asarray(p, dtype=float)[:2]
        if hint is not None:
            cand = (hint + np.arange(-self.WIN, self.WIN + 1)) % n
            d = self.xy[cand] - p
            k = int(np.argmin(np.einsum("ij,ij->i", d, d)))
            if math.sqrt(float(d[k] @ d[k])) < 10.0:
                return self._refine(int(cand[k]), p)
        d = self.xy - p
        return self._refine(int(np.argmin(np.einsum("ij,ij->i", d, d))), p)

    def _refine(self, i, p):
        r = p - self.xy[i]
        a = float(np.clip(r @ self.tan[i], -self.el[i - 1], self.el[i]))
        q = self.xy[i] + a * self.tan[i]
        return i, (self.s[i] + a) % self.length, float(np.hypot(*(p - q)))


class LapTracker:
    """Unwrapped distance along a closed polyline since the first sample; laps = floor(total / L)."""

    def __init__(self, line):
        self.line, self.total, self.idx, self.s_prev = line, 0.0, None, None
        self.dist = 0.0

    def update(self, p_roar):
        L = self.line.length
        self.idx, s, self.dist = self.line.project(p_roar, self.idx)
        if self.s_prev is not None:
            d = (s - self.s_prev + L / 2) % L - L / 2
            if abs(d) < TELEPORT_M:
                self.total += d
        self.s_prev = s
        lap = max(0, int((self.total + 1e-6) // L))
        return self.total, min(lap, LAPS), min(100.0, max(0.0, 100.0 * self.total / (LAPS * L)))

    @property
    def s(self):
        return self.s_prev


def _rg(f):  # 0 -> red, 0.5 -> yellow, 1 -> green
    f = min(1.0, max(0.0, f))
    return (255, int(510 * f), 40) if f < 0.5 else (int(510 * (1 - f)), 255, 40)


def _fmt_t(t):
    return "%d:%04.1f" % (int(t // 60), t % 60)


class NotReady(Exception):
    """Vehicle not found yet; the HUD skips this frame instead of disabling itself."""


class Hud:
    def __init__(self, client_factory=None, data_dir=DATA, log=None):
        self.client_factory = client_factory or self._default_client
        self.disabled = False
        self.actor = self.world = None
        self.cl = Polyline(np.load(os.path.join(data_dir, "centreline.npy")))
        self.laps = LapTracker(self.cl)
        self.plan = self.line_tr = None
        self._load_plan(data_dir)
        self.font = self.big = None
        self.t0 = self.lap_t0 = None
        self.lap_times, self.lap_prev = [], 0
        self.cur = None
        self.base = None
        self._mini = None
        self.trail = [None, None]  # previous, current lap layers
        self.csv = None
        self.log = os.environ.get("ROAR_HUD_LOG") == "1" if log is None else log

    # -- data ---------------------------------------------------------------------------------------------------------
    @staticmethod
    def _default_client():
        import carla  # lazy: only the dev viewer needs it

        return carla.Client("127.0.0.1", 2000)

    def _load_plan(self, data_dir):
        """Speed plan keyed by distance since start; mapped via the racing line. None => plan line hidden."""
        try:
            pr = np.load(os.path.join(data_dir, "speed_profile.npz"))
            line = Polyline(np.load(os.path.join(data_dir, "racing_line.npz"))["xy"])
            scale = 1.0
            try:
                scale = float(json.load(open(os.path.join(data_dir, "vehicle_params.json")))["speed_scale"])
            except Exception:
                pass
            self.plan = (pr["s"], pr["v"] * scale, float(pr["start_s"]))
            self.line_tr = LapTracker(line)
        except Exception:
            self.plan = self.line_tr = None

    def _connect(self):
        c = self.client_factory()
        c.set_timeout(120.0)
        self.world = c.get_world()
        actors = list(self.world.get_actors().filter("vehicle.*"))
        if not actors:  # a second client's actor list lags in synchronous mode: retry next frame, don't disable
            raise NotReady("no vehicle in the world yet")
        hero = [a for a in actors if a.attributes.get("role_name") == "hero"]
        self.actor = (hero or actors)[0]

    def sample(self):
        if self.actor is None:
            self._connect()
        a = self.actor
        v3, tf, ctl = a.get_velocity(), a.get_transform(), a.get_control()
        t = float(self.world.get_snapshot().timestamp.elapsed_seconds)
        v = math.hypot(v3.x, v3.y)
        x, y = carla_to_roar(tf.location.x, tf.location.y)
        total, lap, pct = self.laps.update((x, y))
        beta = sideslip_deg(v3.x, v3.y, tf.rotation.yaw)
        if self.t0 is None and v > START_V:
            self.t0 = self.lap_t0 = t
        if lap > self.lap_prev and self.t0 is not None:
            self.lap_times.append(t - self.lap_t0)
            self.lap_t0, self.lap_prev = t, lap
            self.trail = [self.trail[1], None]
        plan = None
        if self.plan is not None:
            lt = self.line_tr
            lt.update((x, y))
            if lt.dist < PLAN_MAX_OFFLINE_M:
                ps, pv, start_s = self.plan
                if self.base is None:
                    d0 = (lt.s - start_s + lt.line.length / 2) % lt.line.length - lt.line.length / 2
                    self.base = d0 if abs(d0) < 50 else d0 % lt.line.length
                plan = float(np.interp(self.base + lt.total, ps, pv))
        sliding = abs(beta) > SLIDE_BETA_DEG and v > SLIDE_MIN_V
        below = over = False
        if plan is not None and v > FLAG_MIN_V:
            below = v < plan - PLAN_TOL_KMH / 3.6 and ctl.brake == 0 and ctl.throttle < BELOW_MAX_THROTTLE
            over = v > plan + PLAN_TOL_KMH / 3.6
        s = dict(t=t, el=0.0 if self.t0 is None else t - self.t0, lap_t=0.0 if self.lap_t0 is None else t - self.lap_t0,
                 total=total, lap=lap, pct=pct, v=v, beta=beta, plan=plan, sliding=sliding, below=below, over=over,
                 thr=float(ctl.throttle), brk=float(ctl.brake), steer=float(ctl.steer), pos=(x, y), idx=self.laps.idx)
        self.cur = s
        return s

    # -- drawing ------------------------------------------------------------------------------------------------------
    def _init_draw(self):
        self.font = pygame.font.Font(None, 22)
        self.big = pygame.font.Font(None, 64)
        xy = self.cl.xy
        lo, hi = xy.min(0), xy.max(0)
        m = 8
        self._k = (MINIMAP_PX - 2 * m) / float((hi - lo).max())
        self._lo, self._m = lo, m
        base = pygame.Surface((MINIMAP_PX, MINIMAP_PX), pygame.SRCALPHA)
        base.fill((0, 0, 0, 120))
        pygame.draw.lines(base, (200, 200, 200), True, [self._mp(p) for p in xy], 2)
        self._mini = base

    def _mp(self, p):  # ROAR frame, plain axes; only the screen y is flipped
        return (int(self._m + (p[0] - self._lo[0]) * self._k), int(MINIMAP_PX - self._m - (p[1] - self._lo[1]) * self._k))

    def _text(self, scr, s, txt, x, y, col=(255, 255, 255), big=False):
        scr.blit((self.big if big else self.font).render(txt, True, col), (x, y))

    @staticmethod
    def _bar(scr, x, y, w, h, f, col, centre=False):
        pygame.draw.rect(scr, (60, 60, 60), (x, y, w, h))
        if centre:
            x0 = x + w // 2
            pygame.draw.rect(scr, col, (min(x0, x0 + int(f * w / 2)), y, abs(int(f * w / 2)) or 1, h))
        else:
            pygame.draw.rect(scr, col, (x, y, int(min(1.0, max(0.0, f)) * w), h))

    def draw(self, scr, s):
        if self._mini is None:
            self._init_draw()
        W, X = 300, 10
        panel = pygame.Surface((W, 310), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 150))
        scr.blit(panel, (X, 10))
        v = s["v"]
        self._text(scr, s, "%3d" % round(v * 3.6), X + 10, 16, _rg(v / SPEED_MAX), big=True)
        self._text(scr, s, "km/h   %.1f m/s" % v, X + 105, 40)
        self._bar(scr, X + 10, 72, W - 20, 8, v / SPEED_MAX, _rg(v / SPEED_MAX))
        self._text(scr, s, "LAP %d / %d   %5.1f %%" % (min(s["lap"] + 1, LAPS), LAPS, s["pct"]), X + 10, 88)
        self._text(scr, s, "T %s   lap %s" % (_fmt_t(s["el"]), _fmt_t(s["lap_t"])), X + 10, 108)
        if self.lap_times:
            self._text(scr, s, "  ".join("L%d %s" % (i + 1, _fmt_t(t)) for i, t in enumerate(self.lap_times[-3:])),
                       X + 10, 128, (190, 190, 190))
        b = s["beta"]
        self._text(scr, s, "slip %+5.1f deg" % b, X + 10, 152)
        self._bar(scr, X + 150, 156, 140, 8, b / BETA_FULL_DEG, (230, 200, 60), centre=True)
        if s["sliding"]:
            self._text(scr, s, "SLIDING", X + 10, 172, (255, 60, 60))
        if s["plan"] is not None:
            d = (v - s["plan"]) * 3.6
            col = (255, 190, 40) if s["below"] else (255, 120, 60) if s["over"] else (190, 190, 190)
            self._text(scr, s, "plan %3d km/h  %+d" % (round(s["plan"] * 3.6), round(d)), X + 10, 194, col)
            if s["below"]:
                self._text(scr, s, "BELOW PLAN %+d km/h" % round(d), X + 10, 214, (255, 190, 40))
            elif s["over"]:
                self._text(scr, s, "OVER PLAN %+d km/h" % round(d), X + 10, 214, (255, 120, 60))
        for i, (n, f, c, ce) in enumerate((("thr", s["thr"], (60, 200, 60), False), ("brk", s["brk"], (230, 60, 60), False),
                                           ("str", s["steer"], (90, 160, 255), True))):
            y = 244 + 22 * i
            self._text(scr, s, n, X + 10, y - 3)
            self._bar(scr, X + 55, y, W - 70, 10, f, c, centre=ce)
        self._draw_minimap(scr, s)

    def _draw_minimap(self, scr, s):
        cur = self.trail[1]
        if cur is None:
            cur = self.trail[1] = pygame.Surface((MINIMAP_PX, MINIMAP_PX), pygame.SRCALPHA)
        pt = self._mp(s["pos"])
        pygame.draw.circle(cur, _rg(s["v"] / SPEED_MAX), pt, 2)
        if s["sliding"]:
            pygame.draw.circle(cur, (255, 0, 0), pt, 4)
        elif s["below"]:
            pygame.draw.circle(cur, (255, 170, 0), pt, 4)
        mini = self._mini.copy()
        for lay in self.trail:
            if lay is not None:
                mini.blit(lay, (0, 0))
        pygame.draw.circle(mini, (255, 255, 255), pt, 5, 2)
        scr.blit(mini, (10, max(330, scr.get_height() - MINIMAP_PX - 10)))

    # -- per frame ----------------------------------------------------------------------------------------------------
    def frame(self, scr):
        if self.disabled:
            return
        try:
            s = self.sample()
            self.draw(scr, s)
            if self.log:
                self._log(s)
        except NotReady:
            pass
        except Exception as e:  # never break the race
            self.disabled = True
            print("HUD disabled: %s: %s" % (type(e).__name__, e), file=sys.stderr)

    def _log(self, s):
        if self.csv is None:
            os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
            f = open(os.path.join(ROOT, "logs", "hud.csv"), "w", buffering=1, newline="")
            self.csv = csv.writer(f)
            self.csv.writerow(["t", "el", "pct", "lap", "v", "beta", "plan", "thr", "brk", "steer", "sliding", "below", "over"])
        self.csv.writerow([round(s[k], 3) if isinstance(s[k], float) else s[k] for k in
                           ("t", "el", "pct", "lap", "v", "beta", "plan", "thr", "brk", "steer", "sliding", "below", "over")])


_orig = None


def install(hud=None):
    """Patch ManualControlViewer.render so the HUD is drawn right before its (single) display flip."""
    global _orig
    import infrastructure

    if _orig is not None:
        return
    hud = hud or Hud()
    _orig = orig = infrastructure.ManualControlViewer.render

    def render(self, *a, **k):
        real = pygame.display.flip

        def flip():
            if self.screen is not None:
                hud.frame(self.screen)
            real()

        pygame.display.flip = flip
        try:
            return orig(self, *a, **k)
        finally:
            pygame.display.flip = real

    infrastructure.ManualControlViewer.render = render

    # Grab the race car in-process when the runner wraps it (no second client needed); fallback = _connect().
    wrap_init = infrastructure.RoarCompetitionAgentWrapper.__init__

    def init(self, wrapped, *a, **k):
        wrap_init(self, wrapped, *a, **k)
        try:
            hud.actor = wrapped._base_actor
            hud.world = hud.actor.get_world()
        except Exception:
            pass

    infrastructure.RoarCompetitionAgentWrapper.__init__ = init


def uninstall():
    global _orig
    import infrastructure

    if _orig is not None:
        infrastructure.ManualControlViewer.render = _orig
        _orig = None
