"""Fit a dynamic bicycle model with saturating tyres to HUD race logs (tools/hud.py, ROAR_HUD_LOG=1).

  python tools/fit_tyres.py logs/a.hud.csv [more .hud.csv] [--write] [--holdout N]

Model (per unit mass; ROAR frame: x forward, y left, yaw CCW; beta = course - yaw; delta = front steer angle, left +):
    alpha_f = delta - beta - a r / v          alpha_r = -beta + b r / v
    F_f = mu_f tanh(C_f alpha_f / mu_f)       F_r = mu_r tanh(C_r alpha_r / mu_r)        [m/s^2]
    beta' = (F_f + F_r) / v - r               r' = (a F_f - b F_r) / I                    [I = Iz/m, m^2]
    delta' = (g(v) * (-steer) - delta) / tau_d        g(v): steer gain, rad per unit, piecewise linear in v
CARLA's steer command is positive to the right, so delta = -g(v) steer. Fitted by multiple shooting: every window
of H ticks is simulated (forward Euler, dt 0.05, measured v and steer as inputs) from its measured (beta, r) and
compared with the measured beta and r. Measured: beta from the velocity vector, r = d(yaw)/dt.
--write stores vehicle_params["tyre"] = {Cf, Cr, mu_f, mu_r, a, b, I, tau_d, g_v, g_k, dt}.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARAMS = os.path.join(ROOT, "competition_code", "data", "vehicle_params.json")
DT = 0.05
H = 20  # window length, ticks (1 s)
STRIDE = 5
V_MIN = 10.0
G_V = [10.0, 25.0, 40.0, 55.0]  # steer-gain knots, m/s (clamped above: steer is too small up there to identify it)
WHEELBASE = 2.875  # m, Tesla Model 3 (CARLA vehicle.tesla.model3); a + b pinned, a within [0.8, L - 0.8]
W_BETA = 5.0  # beta residual weight vs r (beta ~0.05 rad typical, r ~0.3 rad/s)


def load_hud(path):
    """-> dict of arrays: v, steer, beta (rad, ROAR, left +), r (rad/s), yaw (rad), ok mask (no teleport, moving)."""
    h = np.genfromtxt(path, delimiter=",", names=True)
    yaw = np.unwrap(h["yaw"])
    r = np.gradient(yaw) / DT
    beta = -np.radians(h["beta"])  # HUD beta: CARLA sign (right +) -> ROAR (left +)
    jump = np.r_[False, np.hypot(np.diff(h["x"]), np.diff(h["y"])) > 6.0]
    bad = np.convolve(jump, np.ones(2 * H + 1), "same") > 0  # drop windows touching a teleport
    ok = (h["v"] > V_MIN) & ~bad & np.isfinite(beta) & np.isfinite(r)
    return dict(v=h["v"], steer=h["steer"], beta=beta, r=r, yaw=yaw, ok=ok)


def windows(d, h=H, stride=STRIDE):
    """Index arrays (n_win, h+1) of fully-valid windows."""
    n = len(d["v"])
    starts = np.arange(0, n - h - 1, stride)
    idx = starts[:, None] + np.arange(h + 1)[None, :]
    good = d["ok"][idx].all(1)
    return idx[good]


def unpack(theta):
    Cf, Cr, mu_f, mu_r, I, tau_d = np.exp(theta[:6])
    a = 0.8 + (WHEELBASE - 1.6) / (1.0 + np.exp(-theta[6]))
    g = np.exp(theta[7:7 + len(G_V)])
    return Cf, Cr, mu_f, mu_r, a, WHEELBASE - a, I, tau_d, g


def pack(p):
    fa = (p["a"] - 0.8) / (WHEELBASE - 1.6)
    return np.r_[np.log([p["Cf"], p["Cr"], p["mu_f"], p["mu_r"], p["I"], p["tau_d"]]), np.log(fa / (1 - fa)),
                 np.log(p["g_k"])]


def step(beta, r, delta, v, steer, p, dt=DT):
    """One Euler step of (beta, r, delta); all arrays broadcast. p = dict of parameters (g_v, g_k tables)."""
    Cf, Cr, mu_f, mu_r, a, b, I, tau_d = (p[k] for k in ("Cf", "Cr", "mu_f", "mu_r", "a", "b", "I", "tau_d"))
    vv = np.maximum(v, 1.0)
    af = delta - beta - a * r / vv
    ar = -beta + b * r / vv
    Ff = mu_f * np.tanh(Cf * af / mu_f)
    Fr = mu_r * np.tanh(Cr * ar / mu_r)
    beta1 = beta + dt * ((Ff + Fr) / vv - r)
    r1 = r + dt * (a * Ff - b * Fr) / I
    g = np.interp(v, p["g_v"], p["g_k"])
    delta1 = delta + min(dt / tau_d, 1.0) * (-g * steer - delta)
    return beta1, r1, delta1


def simulate(p, V, S, beta0, r0):
    """Windows: V, S (n, h+1) measured speed and steer; returns beta, r (n, h) predicted at ticks 1..h."""
    g0 = np.interp(V[:, 0], p["g_v"], p["g_k"])
    beta, r, delta = beta0.copy(), r0.copy(), -g0 * S[:, 0]  # actuator assumed settled at the window start
    B, R = [], []
    for k in range(V.shape[1] - 1):
        beta, r, delta = step(beta, r, delta, V[:, k], S[:, k], p)
        B.append(beta), R.append(r)
    return np.array(B).T, np.array(R).T


def params_from_theta(theta):
    Cf, Cr, mu_f, mu_r, a, b, I, tau_d, g = unpack(theta)
    return dict(Cf=Cf, Cr=Cr, mu_f=mu_f, mu_r=mu_r, a=a, b=b, I=I, tau_d=tau_d, g_v=list(G_V), g_k=list(g), dt=DT)


def residuals(theta, V, S, BETA, R):
    p = params_from_theta(theta)
    b, r = simulate(p, V, S, BETA[:, 0], R[:, 0])
    return np.r_[(W_BETA * (b - BETA[:, 1:])).ravel(), (r - R[:, 1:]).ravel()]


def fit(V, S, BETA, R, theta0=None, verbose=0):
    from scipy.optimize import least_squares
    if theta0 is None:
        theta0 = pack(dict(Cf=60.0, Cr=80.0, mu_f=16.0, mu_r=16.0, a=1.4, b=WHEELBASE - 1.4, I=2.5, tau_d=0.1,
                           g_k=[0.6, 0.5, 0.4, 0.3]))
    res = least_squares(residuals, theta0, args=(V, S, BETA, R), method="trf", verbose=verbose, max_nfev=200,
                        x_scale=0.3)
    return params_from_theta(res.x), res


def kinematic_baseline(V, S, R, veh):
    """1 s-ahead r prediction of the shipped kinematic model: k_act -> -steer K(v) with lag steer_tau_s, r = v k_act."""
    ck = min(DT / veh.p["steer_tau_s"], 1.0)
    k = R[:, 0] / np.maximum(V[:, 0], 1.0)
    out = []
    for j in range(V.shape[1] - 1):
        k = k + ck * (-S[:, j] * np.array([veh.K(x) for x in V[:, j]]) - k)
        out.append(V[:, j + 1] * k)
    return np.array(out).T


def main(argv):
    hold = int(argv[argv.index("--holdout") + 1]) if "--holdout" in argv else None
    files = [a for a in argv if a.endswith(".csv")]
    hold = max(1, len(files) // 5) if hold is None else hold
    data = [load_hud(f) for f in files]
    sets = []
    for d in data:
        idx = windows(d)
        sets.append(tuple(d[k][idx] for k in ("v", "steer", "beta", "r")))
    train = [np.concatenate(x) for x in zip(*sets[:-hold])] if hold < len(files) else [np.concatenate(x) for x in zip(*sets)]
    test = [np.concatenate(x) for x in zip(*sets[-hold:])]
    print("train windows %d, holdout windows %d (%d files)" % (len(train[0]), len(test[0]), hold))
    p, res = fit(*train, verbose=1)
    for k in ("Cf", "Cr", "mu_f", "mu_r", "a", "b", "I", "tau_d"):
        print("  %-6s %8.4f" % (k, p[k]))
    print("  g(v) at", G_V, "=", np.round(p["g_k"], 4), " (mu_f + mu_r = %.1f m/s^2)" % (p["mu_f"] + p["mu_r"]))
    V, S, BETA, R = test
    b, r = simulate(p, V, S, BETA[:, 0], R[:, 0])
    sys.path.insert(0, os.path.join(ROOT, "competition_code"))
    from vehicle import Vehicle
    with open(PARAMS) as f:
        pj = json.load(f)
    rk = kinematic_baseline(V, S, R, Vehicle(pj))
    for name, pred, meas in (("r tyre", r, R[:, 1:]), ("r kinematic", rk, R[:, 1:]), ("beta tyre", b, BETA[:, 1:])):
        e = pred - meas
        print("  holdout %-12s rms @0.5 s %.4f  @1 s %.4f   (meas std %.4f)" % (
            name, np.sqrt((e[:, 9] ** 2).mean()), np.sqrt((e[:, -1] ** 2).mean()), meas[:, -1].std()))
    if "--write" in argv:
        pj["tyre"] = {k: (round(float(v), 5) if not isinstance(v, list) else [round(float(x), 5) for x in v])
                      for k, v in p.items()}
        with open(PARAMS, "w") as f:
            json.dump(pj, f, indent=2)
        print("wrote vehicle_params tyre")
    return p


if __name__ == "__main__":
    main(sys.argv[1:])
