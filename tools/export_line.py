"""Export the widened reference racing line. Parses the ref file as data (ast); never imports it."""
import ast
import math
import sys

import numpy as np
from scipy.interpolate import splev, splprep

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


def clean_primary(xy, max_dev=30.0):
    """Drop consecutive duplicates and isolated heading spikes (cyclic). Returns (points, dropped original rows)."""
    keep = [i for i in range(len(xy)) if np.linalg.norm(xy[i] - xy[i - 1]) > 1e-6 or i == 0]
    q = xy[keep]
    m = len(q)
    d = np.roll(q, -1, axis=0) - q  # step k -> k+1
    h = np.degrees(np.arctan2(d[:, 1], d[:, 0]))
    dev = lambda a, b: abs((a - b + 180) % 360 - 180)
    bad = [i for i in range(m)  # step into i is h[i-1]; compare with steps two before/after
           if dev(h[i - 1], h[(i - 3) % m]) > max_dev and dev(h[i - 1], h[(i + 1) % m]) > max_dev]
    dropped = [keep[i] for i in bad]
    print("dropped primary rows (spikes):", dropped)
    return np.delete(q, bad, axis=0), dropped


def _curv(tck, u):
    dx, dy = splev(u, tck, der=1)
    ddx, ddy = splev(u, tck, der=2)
    return (dx * ddy - dy * ddx) / np.hypot(dx, dy) ** 3


def bridge(line, prim, cut=5700, step=1.0, kmax=0.08, ctx=20):
    prim, _ = clean_primary(prim)
    jo = int(np.argmin(np.linalg.norm(prim - line[cut], axis=1)))
    ji = int(np.argmin(np.linalg.norm(prim - line[0], axis=1)))
    n = len(prim)
    mid = prim[[(jo + 1 + k) % n for k in range((ji - jo - 1) % n)]]
    pts = np.vstack([line[cut - ctx:cut + 1], mid, line[:ctx + 1]])
    w = np.r_[np.full(ctx + 1, 10.0), np.ones(len(mid)), np.full(ctx + 1, 10.0)]
    for s in np.geomspace(1e-3, 1e3, 200):  # smallest s whose bridge span is smooth enough
        tck, u = splprep(pts.T, w=w, k=3, s=s)
        ua, ub = u[ctx], u[-ctx - 1]
        dense = np.linspace(ua, ub, 4000)
        if np.abs(_curv(tck, dense)).max() < kmax:
            break
    else:
        raise RuntimeError("no smoothing factor reaches kmax")
    x, y = splev(dense, tck)
    d = np.r_[0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]
    t = np.linspace(0, d[-1], max(int(round(d[-1] / step)), 1) + 1)[1:-1]
    uu = np.interp(t, d, dense)
    print("bridge smoothing s=%.4g, max|k|=%.4f" % (s, np.abs(_curv(tck, dense)).max()))
    return np.column_stack(splev(uu, tck))


def offsets(xy, centre):
    """Signed lateral offset (left +) of each point from the closed centreline polyline.
    Returns (d, foot) with foot the nearest point on the polyline."""
    a, b = centre, np.roll(centre, -1, axis=0)
    e = b - a
    ee = np.maximum((e * e).sum(1), 1e-12)
    d = np.empty(len(xy))
    foot = np.empty_like(xy)
    for i, p in enumerate(xy):  # ponytail: O(N*M) scan (~17M ops), fine offline
        t = np.clip(((p - a) * e).sum(1) / ee, 0, 1)
        f = a + t[:, None] * e
        j = int(np.argmin(((f - p) ** 2).sum(1)))
        foot[i] = f[j]
        d[i] = np.sign(e[j, 0] * (p[1] - f[j, 1]) - e[j, 1] * (p[0] - f[j, 0])) * np.linalg.norm(p - f[j])
    return d, foot


def wall_margin(xy, centre, max_offset=4.0):
    """Shift points toward the centreline by a Gaussian-smoothed (sigma 15 m) excess so |d| <= max_offset."""
    d0, _ = offsets(xy, centre)
    excess = np.maximum(0, np.abs(d0) - max_offset)
    print("wall margin: %d violating points, max|d| before %.3f" % ((excess > 0).sum(), np.abs(d0).max()))
    if not excess.any():
        return xy
    ds = np.linalg.norm(np.roll(xy, -1, axis=0) - xy, axis=1)
    sig = 15.0 / ds.mean()
    h = int(4 * sig)
    k = np.exp(-0.5 * (np.arange(-h, h + 1) / sig) ** 2)
    k /= k.sum()
    sm = np.convolve(np.r_[excess[-h:], excess, excess[:h]], k, mode="valid")  # circular
    f = 1.0
    for _ in range(20):
        foot = offsets(xy, centre)[1]
        dirn = foot - xy
        dirn /= np.maximum(np.linalg.norm(dirn, axis=1, keepdims=True), 1e-9)
        new = xy + dirn * (f * sm)[:, None]
        dn, _ = offsets(new, centre)
        if np.abs(dn).max() <= max_offset + 1e-9:
            break
        f *= 1.1
    else:
        raise RuntimeError("wall margin did not converge")
    sh = np.linalg.norm(new - xy, axis=1)
    arc = np.r_[0, np.cumsum(ds)[:-1]]
    mv = np.where(sh > 1e-6)[0]
    runs = np.split(mv, np.where(np.diff(mv) > 1)[0] + 1)
    print("  max|d| after %.3f, max shift %.3f m (scale %.2f), moved s-ranges: %s" % (
        np.abs(dn).max(), sh.max(), f,
        ", ".join("%.0f-%.0f" % (arc[r[0]], arc[r[-1]]) for r in runs if len(r))))
    return new


def main(ref, prim_path, out, cut=5700, centre=None, max_offset=4.0):
    line = widen(parse_literal(ref))
    prim = load_primary(prim_path)
    xy = np.vstack([line[:cut + 1], bridge(line, prim, cut)])
    if centre is not None:
        xy = wall_margin(xy, np.load(centre, allow_pickle=False), max_offset)
    np.savez(out, xy=xy)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("primary")
    ap.add_argument("out")
    ap.add_argument("--centre")
    ap.add_argument("--max-offset", type=float, default=4.0)
    a = ap.parse_args()
    main(a.ref, a.primary, a.out, centre=a.centre, max_offset=a.max_offset)
