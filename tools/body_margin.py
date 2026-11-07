"""How far a race's car BODY went beyond the widest body positions of reference races, per tuning section.

The car centre alone hides slides: at 20 deg of sideslip the nose and tail sweep ~0.8 m wider than the centre. A
sec2 grip of 41 m/s^2 hit the Lesmo-1 apex with its centre 0.46 m off the line, body 0.35-0.4 m beyond the shipped
car's body; 37-39 m/s^2 raced clean with 0.33-0.41 m. Body = 4 corners of a HALF_L x HALF_W box around the centre,
nose heading = line heading + e_psi (run.csv). Offsets are signed lateral distances from the centreline, binned per
0.5 m of centreline after 5x linear upsampling (+-0.5 m neighbours merged), reference = union over the reference races.

  python tools/body_margin.py "logs/ref/*.run.csv" RUN.run.csv [RUN ...]   -> per-section max metres beyond

logs/ref/ (not committed; git-ignored like all logs) held the five 462bc0b gate races (gk4w_1..5.run.csv, 322.50 s).
Any set of clean races of the config you trust works as the reference.
"""
import glob
import json
import os
import sys

import numpy as np
from scipy.ndimage import maximum_filter1d, minimum_filter1d

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "competition_code"), os.path.join(ROOT, "tools")]
from envelope import station  # noqa: E402
from track import load_track  # noqa: E402

DATA = os.path.join(ROOT, "competition_code", "data")
HALF_L, HALF_W = 2.4, 1.05
BIN = 0.5
UPSAMPLE = 5  # linear interpolation between 20 Hz samples (2-3 m apart at speed) before binning
SKIP_S = 300.0  # the standing start is identical for every config


def corner_offsets(xy, yaw, centre):
    """-> (centreline station of the centre, min and max signed offset of the 4 body corners)."""
    xy, yaw = np.asarray(xy, float), np.unwrap(np.asarray(yaw, float))
    if UPSAMPLE > 1 and len(xy) > 1:
        u = np.arange((len(xy) - 1) * UPSAMPLE + 1) / UPSAMPLE
        i = np.arange(len(xy))
        xy = np.c_[np.interp(u, i, xy[:, 0]), np.interp(u, i, xy[:, 1])]
        yaw = np.interp(u, i, yaw)
    s0, _ = station(xy, centre)
    ds = []
    for a, b in ((HALF_L, HALF_W), (HALF_L, -HALF_W), (-HALF_L, HALF_W), (-HALF_L, -HALF_W)):
        q = xy + np.c_[a * np.cos(yaw) - b * np.sin(yaw), a * np.sin(yaw) + b * np.cos(yaw)]
        ds.append(station(q, centre)[1])
    ds = np.array(ds)
    return s0, ds.min(0), ds.max(0)


def run_body(run_csv, track, centre):
    """(station, body min, body max, line_s) of a run.csv, reconstructed on its racing line. A HUD csv with x, y, yaw
    columns (e.g. another entry's telemetry) is used as is; its line_s comes from projecting onto `track`."""
    d = np.genfromtxt(run_csv, delimiter=",", names=True)
    if "yaw" in d.dtype.names and "x" in d.dtype.names:
        xy = np.c_[d["x"], d["y"]]
        r = corner_offsets(xy, d["yaw"], centre)
        return r + (np.zeros(len(r[0])),)  # line_s unknown; only used as a reference
    m = d["s_total"] > SKIP_S
    s = d["line_s"][m] % track.length
    p = np.c_[np.interp(s, track.s, track.xy[:, 0], period=track.length),
              np.interp(s, track.s, track.xy[:, 1], period=track.length)]
    psi = np.interp(s, track.s, np.unwrap(track.psi), period=track.length)
    xy = p + d["e_y"][m][:, None] * np.c_[-np.sin(psi), np.cos(psi)]
    r = corner_offsets(xy, psi + d["e_psi"][m], centre)
    su = np.unwrap(s * 2 * np.pi / track.length) * track.length / (2 * np.pi)
    su = np.interp(np.arange(len(r[0])) / UPSAMPLE, np.arange(len(s)), su) % track.length
    return r + (su,)


class BodyMargin:
    def __init__(self, ref_runs, line=os.path.join(DATA, "racing_line.npz"),
                 centre=os.path.join(DATA, "centreline.npy"), params=os.path.join(DATA, "vehicle_params.json")):
        """ref_runs: run.csv paths, all driven on `line`."""
        self.track = load_track(line)
        self.centre = np.load(centre)
        L = np.linalg.norm(np.roll(self.centre, -1, 0) - self.centre, axis=1).sum()
        self.nb = int(L / BIN) + 1
        self.starts = np.array([s for s, _ in json.load(open(params))["a_lat_sections"]])
        lo, hi = np.full(self.nb, np.inf), np.full(self.nb, -np.inf)
        if not ref_runs:
            raise ValueError("no reference runs")
        for f in ref_runs:
            s0, a, b, _ = run_body(f, self.track, self.centre)
            k = (s0 / BIN).astype(int) % self.nb
            np.minimum.at(lo, k, a)
            np.maximum.at(hi, k, b)
        self.lo = minimum_filter1d(lo, 3, mode="wrap")
        self.hi = maximum_filter1d(hi, 3, mode="wrap")

    def excess(self, run_csv, line=None):
        """-> (per-section max metres beyond the reference bodies (-inf where not driven), line_s of the worst tick)."""
        tr = self.track if line is None else load_track(line)
        s0, a, b, ls = run_body(run_csv, tr, self.centre)
        k = (s0 / BIN).astype(int) % self.nb
        out = np.maximum(b - self.hi[k], self.lo[k] - a)
        n = len(self.starts)
        i = np.searchsorted(self.starts, ls, side="right") - 1
        i = np.where(i < 0, n - 1, i)
        per = np.full(n, -np.inf)
        np.maximum.at(per, i, out)
        return per, float(ls[np.argmax(out)])


def main(argv):
    bm = BodyMargin(sorted(glob.glob(argv[0])))
    for f in argv[1:]:
        per, at = bm.excess(f)
        print("%-28s max %5.2f m @ line_s %5.0f | " % (os.path.basename(f)[:28], per.max(), at)
              + " ".join("%5.2f" % x for x in per))


if __name__ == "__main__":
    main(sys.argv[1:])
