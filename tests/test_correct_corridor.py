import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import correct_corridor as cc  # noqa: E402


def _map():
    n = 100
    return dict(station=np.arange(n, dtype=float), left=np.full(n, 6.0), right=np.full(n, 6.0))


def _probe(tmp_path, rows):
    f = tmp_path / "p.csv"
    np.savetxt(f, np.array(rows), delimiter=",", header="line_s,station,d_touch,impulse,body_in,body_in_touch", comments="")
    return str(f)


def test_probe_edges_move_the_right_wall_for_right_inside_and_left_wall_for_left_inside(tmp_path):
    f = _probe(tmp_path, [[0, 10.2, 0.30, 50, -5.4, -5.70],      # right-hander inside: wall on the right at 5.70
                          [0, 20.0, 0.20, 50, 5.6, 5.80],        # left-hander inside: wall on the left at 5.80
                          [0, 30.0, 0.00, 900, -5.0, -5.00],     # d_touch 0.00 = a credited-late event, skipped
                          [0, 40.0, np.nan, 0, -4.0, np.nan]])   # no touch within the shift range
    z, moved = cc.correct(_map(), [f])
    assert z["right"][10] == 5.70 and z["right"][9] == 5.70 and z["right"][11] == 5.70 and z["right"][12] == 6.0
    assert z["left"][20] == 5.80 and z["left"][10] == 6.0 and z["right"][20] == 6.0
    assert z["right"][30] == 6.0 and z["left"][40] == 6.0 and z["right"][40] == 6.0
    assert moved == 6 and bool(z["probe_corrected"])


def test_correction_only_moves_walls_nearer_and_is_idempotent(tmp_path):
    near = _probe(tmp_path, [[0, 50.0, 0.1, 9, -5.0, -5.20]])
    far = tmp_path / "far.csv"
    np.savetxt(far, np.array([[0, 50.0, 0.1, 9, -5.0, -5.90]]), delimiter=",",
               header="line_s,station,d_touch,impulse,body_in,body_in_touch", comments="")
    z, _ = cc.correct(_map(), [near])
    z2, _ = cc.correct(z, [str(far), near])  # a farther edge never moves a wall back out
    assert z2["right"][50] == 5.20
    z3, moved = cc.correct(z2, [near])
    assert moved == 0 and np.array_equal(z3["right"], z2["right"])
