"""Add cosine-windowed lateral bumps to a learned line offset (tools/ilc.py output).

  python tools/offset_bump.py IN.npz OUT.npz A:B:AMP[:BLEND] [A:B:AMP[:BLEND] ...]

d(s) is metres LEFT of the racing line (the MPC tracks line + d). Each bump adds AMP over line_s A..B, cosine ramps of
BLEND m (default 25) each side. Negative AMP = move right: away from the inside wall of a left-hander.
"""
import sys

import numpy as np

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from ilc import window  # noqa: E402


def bump(s, d, specs):
    d = np.array(d, float, copy=True)
    for sp in specs:
        a, b, amp, *bl = (float(x) for x in sp.split(":"))
        d += amp * window(s, a, b, bl[0] if bl else 25.0)
    return d


if __name__ == "__main__":
    z = np.load(sys.argv[1])
    np.savez(sys.argv[2], s=z["s"], d=bump(z["s"], z["d"], sys.argv[3:]))
    print("wrote", sys.argv[2], "max |change| %.2f m" % np.abs(bump(z["s"], z["d"], sys.argv[3:]) - z["d"]).max())
