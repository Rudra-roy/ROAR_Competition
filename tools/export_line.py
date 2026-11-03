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


def main(ref, out):
    np.savez(out, xy=widen(parse_literal(ref)))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
