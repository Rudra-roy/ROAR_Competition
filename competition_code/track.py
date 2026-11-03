"""Racing-line geometry: smoothed resampling, curvature, projection, progress."""
from typing import Optional, Tuple

import numpy as np
from scipy.interpolate import splev, splprep

SMOOTH_SIGMA = 0.015  # m, assumed per-point noise std at ~0.94 m point spacing (scaled with spacing)


TELEPORT_JUMP_M = 6.0  # max legit motion is ~71 m/s * 0.05 s = 3.6 m per tick


def _wrap(a):
    return np.pi - (np.pi - a) % (2 * np.pi)  # (-pi, pi]


class Track:
    def __init__(self, xy: np.ndarray, ds: float = 1.0, sigma: float = SMOOTH_SIGMA):
        xy = np.asarray(xy, dtype=float)[:, :2]
        n = len(xy)
        pts = np.vstack([xy, xy[:1]])
        spacing = np.linalg.norm(np.diff(pts, axis=0), axis=1).mean()  # close the loop for splprep(per=1)
        tck, _ = splprep([pts[:, 0], pts[:, 1]], s=n * (sigma * spacing / 0.94) ** 2, per=1)
        u = np.linspace(0, 1, 20 * n + 1)
        x, y = splev(u, tck)
        seg = np.hypot(np.diff(x), np.diff(y))
        cum = np.r_[0.0, np.cumsum(seg)]
        self.length = float(cum[-1])
        m = int(round(self.length / ds))
        self.ds = self.length / m
        self.s = np.arange(m) * self.ds
        uu = np.interp(self.s, cum, u)
        x, y = splev(uu, tck)
        dx, dy = splev(uu, tck, der=1)
        ddx, ddy = splev(uu, tck, der=2)
        self.xy = np.c_[x, y]
        self.psi = np.arctan2(dy, dx)
        self.kappa = (dx * ddy - dy * ddx) / (dx * dx + dy * dy) ** 1.5
        self._tan = np.c_[np.cos(self.psi), np.sin(self.psi)]

    def project(self, p: np.ndarray, hint_idx: Optional[int] = None) -> Tuple[int, float, float]:
        p = np.asarray(p, dtype=float)[:2]
        m = len(self.s)
        if hint_idx is None:
            cand = np.arange(m)
        else:
            w = int(np.ceil(40.0 / self.ds))
            cand = (hint_idx + np.arange(-w, w + 1)) % m
        d = self.xy[cand] - p
        i = int(cand[np.argmin(np.einsum("ij,ij->i", d, d))])
        # refine along the tangent at the nearest sample (clamped to +-ds)
        t = self._tan[i]
        r = p - self.xy[i]
        a = float(np.clip(r @ t, -self.ds, self.ds))
        s = (self.s[i] + a) % self.length
        e_y = float(t[0] * r[1] - t[1] * r[0])
        return i, s, e_y

    def heading_error(self, idx: int, yaw: float) -> float:
        return float(_wrap(yaw - self.psi[idx]))


class Progress:
    """Unwrapped distance since (re)start, with teleport relocalization."""

    def __init__(self, track: Track):
        self.track = track
        self._last_p = None
        self._idx = None
        self._s_prev = 0.0
        self.start_s = 0.0
        self._total = 0.0

    def update(self, p, yaw):
        tr = self.track
        p = np.asarray(p, dtype=float)[:2]
        fresh = self._last_p is None or np.hypot(*(p - self._last_p)) > TELEPORT_JUMP_M
        idx, s, e_y = tr.project(p, None if fresh else self._idx)
        if fresh:
            self.start_s, self._s_prev, self._total = s, s, 0.0
        else:
            ds = _wrap((s - self._s_prev) * 2 * np.pi / tr.length) * tr.length / (2 * np.pi)
            self._total += ds
            self._s_prev = s
        self._last_p, self._idx = p, idx
        return self._total, e_y, tr.heading_error(idx, yaw), idx
