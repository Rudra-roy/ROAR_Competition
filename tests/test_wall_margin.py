import numpy as np

from wall_margin import margins


def test_margins_on_a_circle(tmp_path):
    """Line = centreline = CCW circle (left = inward). The car runs 1 m right of the line, 3.5 m right near the wall
    point at angle 0 and 2.0 m left near angle pi: chicane right offset 3.5, lap max 3.5, corner window max 2.0."""
    r = 300.0
    th = np.linspace(0, 2 * np.pi, 2000, endpoint=False)
    circle = r * np.c_[np.cos(th), np.sin(th)]
    L = 2 * np.pi * r
    ls = np.arange(0.0, L, 0.5)
    e_y = np.full_like(ls, -1.0)
    e_y[(ls > L - 3) | (ls < 3)] = -3.5   # around the wall point (r, 0)
    e_y[np.abs(ls - L / 2) < 3] = 2.0     # left side, opposite
    run = tmp_path / "run.csv"
    np.savetxt(run, np.c_[ls, e_y], delimiter=",", header="line_s,e_y", comments="")
    m = margins(str(run), circle, circle, wall=np.array([[r, -1.0], [r, 1.0]]), corners=[(L / 2 - 50, L / 2 + 10)])
    assert abs(m["chicane_right"] - 3.5) < 0.02
    assert abs(m["lap_max"] - 3.5) < 0.02
    assert abs(m["corners"][0] - 2.0) < 0.02
