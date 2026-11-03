"""Offline TOPP-RA speed profile over the race -> competition_code/data/speed_profile.npz.

Dev-only (imports toppra). The runtime reads the npz via vehicle.load_profile.
Run: python tools/build_profile.py
"""
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "competition_code", "data")
# toppra's seidel LP returns NaN ("not controllable") with x = sd^2 ~ v^2 ~ 5e3 (ill-conditioned);
# path parameter r = s / PATH_SCALE keeps x ~ 0.5. q(r) = PATH_SCALE * r is still s. (1 and 0.1 fail; 10..1000 agree.)
PATH_SCALE = 100.0


def build_profile(track, veh, start_s, laps=3, grid=2.0):
    """Time-optimal v(s) from rest over `laps` laps; s = distance from start_s along the line."""
    import toppra
    import toppra.algorithm as algo
    import toppra.constraint as constraint

    L = track.length
    n = laps * int(round(L / grid))  # lap boundaries fall on grid points
    ss = np.linspace(0.0, laps * L, n + 1)
    line = lambda q: (start_s + q[0]) % L
    kappa = lambda q: np.interp(line(q), track.s, track.kappa, period=L)

    lines = veh.p["a_drive_lines"]
    a_brk = veh.a_brake * veh.brake_scale
    # Constraint vector [a_long, v^2, a_lat, |a_lat|]. Box: a_long >= -a_brk, |a_lat| <= a_lat(s). With
    # brake_ellipse_c also braking shares grip: (a_long/a_brk)^2 + (a_lat/(c a_lat(s)))^2 <= 1 for a_long < 0,
    # as tangent lines at 15..75 deg (outer polygon, <= 0.9 % loose). Absent = box only.
    c_el = veh.p.get("brake_ellipse_c")
    th = np.radians([15, 30, 45, 60, 75]) if c_el else np.zeros(0)
    F0 = np.array([[1.0, b, 0, 0] for _, b in lines] + [[-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, -1, 0], [0, 1, 0, 0]]
                  + [[-np.cos(t) / a_brk, 0, 0, 0] for t in th], float)
    g_fixed = [a for a, _ in lines] + [a_brk]
    vmax2 = veh.p["v_max"] ** 2

    def F(q):
        if not len(th):
            return F0
        Fq = F0.copy()
        Fq[-len(th):, 3] = np.sin(th) / (c_el * veh.a_lat(line(q)))
        return Fq

    def g(q):
        al = veh.a_lat(line(q))
        return np.array(g_fixed + [al, al, vmax2] + [1.0] * len(th))

    pc = constraint.SecondOrderConstraint(
        lambda q, qd, qdd: np.array([qdd[0], qd[0] ** 2, kappa(q) * qd[0] ** 2, abs(kappa(q)) * qd[0] ** 2]),
        F, g, dof=1)

    class FreeEnd(algo.TOPPRA):  # toppra pins the end speed to sd_end; free it to [0, sd_end]
        def compute_controllable_sets(self, sdmin, sdmax):
            return super().compute_controllable_sets(0.0, sdmax)

    r = ss / PATH_SCALE
    inst = FreeEnd([pc], toppra.SplineInterpolator(r, ss[:, None]), gridpoints=r,
                   solver_wrapper="seidel", parametrizer="ParametrizeConstAccel")
    _, sd, _ = inst.compute_parameterization(0.0, veh.p["v_max"] / PATH_SCALE)
    if sd is None or np.isnan(sd).any():
        raise RuntimeError("TOPP-RA failed: %s" % inst.problem_data.return_code)
    return ss, sd * PATH_SCALE  # v = dq/dt = PATH_SCALE * dr/dt


def lap_times(s, v, length):
    """Per-lap times with constant acceleration between grid points (finite from rest)."""
    t = np.r_[0.0, np.cumsum(2 * np.diff(s) / (v[:-1] + v[1:]))]
    tb = np.interp(np.arange(0, s[-1] / length + 0.5) * length, s, t)
    return np.diff(tb)


def slowest_corners(s, v, length, n=5, sep=100.0):
    """n lowest local minima of v, at least `sep` m apart along the closed line (s = line s)."""
    i = np.flatnonzero((v[1:-1] < v[:-2]) & (v[1:-1] <= v[2:])) + 1
    out = []
    for j in i[np.argsort(v[i])]:
        if all(min(abs(s[j] - s[k]), length - abs(s[j] - s[k])) > sep for k in out):
            out.append(j)
        if len(out) == n:
            break
    return out


def main():
    sys.path[:0] = [os.path.join(ROOT, "competition_code"), os.path.join(ROOT, "tests")]
    from profile_oracle import oracle_profile
    from track import Track
    from vehicle import Vehicle

    tr = Track(np.load(os.path.join(DATA, "racing_line.npz"))["xy"])
    veh = Vehicle.from_json(os.path.join(DATA, "vehicle_params.json"))
    L = tr.length
    start_s = tr.project(np.load(os.path.join(DATA, "centreline.npy"))[0])[1]

    t0 = time.time()
    s, v = build_profile(tr, veh, start_s)
    t_build = time.time() - t0
    np.savez(os.path.join(DATA, "speed_profile.npz"), s=s, v=v, start_s=start_s)

    line_s = (start_s + s) % L
    laps = lap_times(s, v, L)
    print("line length %.1f m, grid %.3f m, %d points" % (L, s[1] - s[0], len(s)))
    print("start_s (line) = %.2f m" % start_s)
    for k, t in enumerate(laps):
        print("lap %d: %.2f s" % (k + 1, t))
    print("total 3 laps: %.2f s" % laps.sum())
    j = int(np.argmax(v))
    print("top speed %.2f m/s (%.1f km/h) at line s %.1f (lap %d)" % (v[j], 3.6 * v[j], line_s[j], s[j] // L + 1))
    lap2 = (s >= L) & (s < 2 * L)
    print("5 slowest corners (lap 2):")
    for j in slowest_corners(line_s[lap2], v[lap2], L):
        print("  line s %7.1f  v %5.2f m/s" % (line_s[lap2][j], v[lap2][j]))

    _, vo = oracle_profile(lambda q: np.interp((start_s + q) % L, tr.s, tr.kappa, period=L),
                           lambda q: veh.a_lat((start_s + q) % L), veh.a_drive,
                           veh.a_brake * veh.brake_scale, veh.p["v_max"], 3 * L, s[1] - s[0])
    rel = np.abs(v[1:] - vo[1:]) / vo[1:]
    print("TOPPRA vs oracle: |dv|/v median %.4f, p95 %.4f, max %.4f, within 2%%: %.1f%%; oracle total %.2f s"
          % (np.median(rel), np.percentile(rel, 95), rel.max(), 100 * np.mean(rel <= 0.02),
             lap_times(s, vo, L).sum()))
    print("build time %.1f s" % t_build)


if __name__ == "__main__":
    main()
