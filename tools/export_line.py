"""Export the widened reference racing line. Parses the ref file as data (ast); never imports it."""
import ast
import math
import sys

import numpy as np

SPECS = [(1313, 1393, 0.9), (1393, 1473, -0.9), (883, 963, -0.5), (963, 1043, 0.5),
         (4036, 4116, 0.5), (4116, 4196, -0.5), (1688, 1768, 0.3), (1768, 1848, -0.3)]


def parse_literal(path):
    with open(path) as f:
        tree = ast.parse(f.read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "WP_TO_FOLLOW" for t in node.targets):
            text = ast.literal_eval(node.value)
            break
    else:
        raise ValueError("WP_TO_FOLLOW not found")
    return np.array([[float(v) for v in ln.split(",")] for ln in text.strip().split("\n")])


def widen(line, specs=SPECS):
    n = len(line)
    out = line.copy()
    for start, end, amount in specs:
        span = end - start
        if span < 4:
            continue
        for k in range(span + 1):
            i = (start + k) % n
            tx, ty = line[(i + 1) % n] - line[(i - 1) % n]  # tangent from the unshifted line
            norm = math.hypot(tx, ty)
            if norm < 1e-6:
                continue
            w = 0.5 * (1 - math.cos(2 * math.pi * k / span))
            out[i] += np.array([-ty, tx]) / norm * amount * w
    return out


def load_primary(path, max_step=10.0):
    """x,y of the primary list, truncated at the first step > max_step (stray tail rows)."""
    xy = np.load(path, allow_pickle=False)["locations"][:, :2]
    jump = np.where(np.linalg.norm(np.diff(xy, axis=0), axis=1) > max_step)[0]
    return xy[:jump[0] + 1] if len(jump) else xy


def bridge(line, prim, cut=5700, step=1.0):
    """Close the loop: primary-path points from line[cut] back to line[0], resampled to ~step."""
    jo = int(np.argmin(np.linalg.norm(prim - line[cut], axis=1)))
    ji = int(np.argmin(np.linalg.norm(prim - line[0], axis=1)))
    n = len(prim)
    idx = [(jo + 1 + k) % n for k in range((ji - jo - 1) % n)]
    pts = np.vstack([line[cut], prim[idx], line[0]])
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    m = max(int(round(d[-1] / step)), 1)
    t = np.linspace(0, d[-1], m + 1)[1:-1]
    return np.c_[np.interp(t, d, pts[:, 0]), np.interp(t, d, pts[:, 1])]


def main(ref, prim_path, out, cut=5700):
    line = widen(parse_literal(ref))
    prim = load_primary(prim_path)
    np.savez(out, xy=np.vstack([line[:cut + 1], bridge(line, prim, cut)]))


if __name__ == "__main__":
    main(*sys.argv[1:4])
