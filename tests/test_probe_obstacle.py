import os

import numpy as np
import pytest

carla = pytest.importorskip("carla")  # probe_obstacle imports the CARLA client at module load
import probe_obstacle as po  # noqa: E402
from track import load_track  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = os.path.join(ROOT, "logs", "ref_ship", "gb4.run.csv")


@pytest.mark.skipif(not os.path.exists(REF), reason="needs a reference race log")
def test_poses_follow_the_reference_race_inside_the_window():
    tr = load_track(os.path.join(ROOT, "competition_code", "data", "racing_line.npz"))
    s, xy, nrm, yaw, kap = po.poses(REF, tr, 1680.0, 1700.0, lap=2)
    assert len(s) > 5 and s.min() >= 1680.0 and s.max() <= 1700.0
    assert np.all(np.diff(s) > 0)  # one pass, in order
    assert np.allclose(np.linalg.norm(nrm, axis=1), 1.0)
    assert np.all(kap < 0)  # Lesmo 1 is a right-hander: the inside is to the right
    d = np.genfromtxt(REF, delimiter=",", names=True)
    L = tr.length
    i = np.flatnonzero((d["s_total"] >= L - 300) & (d["s_total"] < 2 * L - 300) & (d["line_s"] >= 1680) & (d["line_s"] <= 1700))[0]
    assert s[0] == d["line_s"][i]
    assert abs((yaw[0] - d["yaw"][i] + np.pi) % (2 * np.pi) - np.pi) < 0.03  # = the car's logged heading
