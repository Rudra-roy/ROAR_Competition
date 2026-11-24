"""tools/media_data.py: the runner's laps come from distance driven, not from line_s wrapping (line_s has an arbitrary
origin 87 m past the start grid). A first draft numbered the run-in to the line as lap 1 and ended on 'lap 4 of 3'."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from media_data import DATA, Race  # noqa: E402
from track import load_track  # noqa: E402


def _csv(tmp_path):
    L = load_track(os.path.join(DATA, "racing_line.npz")).length
    s_total = np.concatenate([[-2.9e-5], np.linspace(0, 3 * L - 10, 3000)])  # tiny negative first tick, ends 10 m short
    line_s = (5501.0 + s_total) % L  # start grid 87 m before the line_s origin
    n = len(s_total)
    cols = dict(tick=np.arange(n), s_total=s_total, line_s=line_s, e_y=np.zeros(n), v=np.full(n, 50.0),
                v_ref=np.full(n, 52.0))
    p = tmp_path / "run.csv"
    p.write_text(",".join(cols) + "\n" + "\n".join(",".join(repr(float(c[i])) for c in cols.values()) for i in range(n)))
    return str(p), L


def test_laps_follow_distance_driven_and_stay_in_1_to_3(tmp_path):
    path, L = _csv(tmp_path)
    r = Race(path)
    assert r.lap[0] == 1 and r.lap[-1] == 3
    assert set(r.lap) == {1, 2, 3}
    assert np.all(np.diff(r.lap) >= 0)
    assert abs(r.s[np.argmax(r.lap == 2)] - L) < 3 * 50.0 * 0.05  # lap 2 starts one line-length after the grid


def test_segments_are_whole_passes_of_the_line(tmp_path):
    path, _ = _csv(tmp_path)
    r = Race(path)
    # three laps from a grid 87 m before the origin wrap line_s three times: run-in, two whole passes, the last stub
    assert r.seg.max() == 4
    assert len(r.corner_ticks("Lesmo 1", seg=2)) > 0
