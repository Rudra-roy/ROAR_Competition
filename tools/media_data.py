"""Telemetry -> plot-ready arrays for the README media (tools/make_media.py). Training-time tool.

A race's run.csv (written with ROAR_LOG=1) holds one row per 0.05 s tick: line_s (position on the racing line), e_y
(lateral offset from it), v (m/s), v_ref (planned speed). Positions are rebuilt on the committed racing line, so the
figures show exactly the path the controller drove, in the ROAR frame (x, y).
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "competition_code")]
from track import load_track  # noqa: E402

DATA = os.path.join(ROOT, "competition_code", "data")
DT = 0.05  # s per tick

# Corners named in the README, as line_s (m) of the apex window on the racing line (see "Real walls").
CORNERS = {
    "First chicane": (975.0, 990.0),
    "c1329": (1337.0, 1341.0),
    "Lesmo 1": (1686.0, 1698.0),
    "Kink": (2695.0, 2702.0),
    "Final hairpin": (5405.0, 5411.0),
}


class Race:
    """One logged race: time t (s), position xy (m), speed v (m/s), planned speed v_ref (m/s), distance s (m), lap (1..3)."""

    def __init__(self, run_csv, line=os.path.join(DATA, "racing_line.npz")):
        d = np.genfromtxt(run_csv, delimiter=",", names=True)
        tr = load_track(line)
        L = tr.length
        ls = d["line_s"] % L
        p = np.c_[np.interp(ls, tr.s, tr.xy[:, 0], period=L), np.interp(ls, tr.s, tr.xy[:, 1], period=L)]
        psi = np.interp(ls, tr.s, np.unwrap(tr.psi), period=L)
        self.xy = p + d["e_y"][:, None] * np.c_[-np.sin(psi), np.cos(psi)]
        self.t = d["tick"] * DT
        self.v, self.v_ref, self.line_s = d["v"], d["v_ref"], ls
        self.s = d["s_total"]  # metres driven since the start line
        self.L = L
        self.track = tr
        # line_s has an arbitrary origin, 87 m past the start grid: segments between its wraps are whole passes of the
        # line ('seg'), good for plotting one flying lap, but they are NOT the runner's laps ('lap', from distance driven).
        wraps = np.where(np.diff(ls) < -L / 2)[0] + 1
        self.seg_ticks = [0] + list(wraps) + [len(self.t)]
        self.seg = np.zeros(len(self.t), int)
        for k in range(len(self.seg_ticks) - 1):
            self.seg[self.seg_ticks[k]:self.seg_ticks[k + 1]] = k + 1
        self.lap = np.clip(np.floor(self.s / L).astype(int) + 1, 1, 3)

    @property
    def total_time(self):
        return float(self.t[-1])

    def corner_ticks(self, name, seg=2):
        a, b = CORNERS[name]
        m = (self.seg == seg) & (self.line_s >= a) & (self.line_s <= b)
        return np.where(m)[0]
