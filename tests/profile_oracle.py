"""Pure-numpy forward-backward speed profile: the reference that TOPP-RA is checked against."""
import numpy as np


def oracle_profile(kappa_fn, a_lat_fn, a_drive_fn, a_brake, v_max, length, grid, v0=0.0):
    n = int(round(length / grid))
    s = np.linspace(0.0, length, n + 1)
    ds = s[1] - s[0]
    k = np.abs([kappa_fn(x) for x in s])
    a_lat = np.array([a_lat_fn(x) for x in s])
    cap = np.minimum(v_max, np.sqrt(a_lat / np.maximum(k, 1e-12)))
    v = cap.copy()
    v[0] = min(v0, cap[0])
    for i in range(n):  # forward: drive limit
        v[i + 1] = min(cap[i + 1], np.sqrt(max(v[i] ** 2 + 2 * a_drive_fn(v[i]) * ds, 0.0)))
    for i in range(n - 1, -1, -1):  # backward: brake limit
        v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2 * a_brake * ds))
    return s, v
