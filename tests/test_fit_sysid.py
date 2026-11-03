import numpy as np

from fit_sysid import a_drive, fit

DT = 0.05


def true_a_drive(v):
    return np.minimum(9.0, 12.0 - 0.0025 * v ** 2)


def true_k(v):
    return 0.12 / (1 + 0.0004 * v ** 2)


def synth_log(seed=0):
    """Time series like a sysid lap: launch, full-throttle run-ups, full brakes, steered holds.

    Noise sigma = 0.2 m/s^2 on the integrated acceleration and 0.02 rad/s on yaw rate.
    """
    rng = np.random.default_rng(seed)
    rows = []  # v, throttle, brake, steer, yaw_rate
    v = 0.0

    def run(n, thr, brk, steer_fn, stop=lambda v: False):
        nonlocal v
        for i in range(n):
            st = steer_fn(i, v)
            # brake builds up over 10 ticks like CARLA's (measured ~0.5 s ramp)
            a = true_a_drive(v) if thr == 1 else (-17.0 * min(1.0, i / 10) if brk == 1 else 0.0)
            yr = -true_k(v) * v * st + rng.normal(0, 0.02)
            rows.append((v, thr, brk, st, yr))
            v = max(0.0, v + (a + rng.normal(0, 0.2)) * DT)
            if stop(v):
                return

    dither = lambda i, v: (0.05 if (i * DT) % 1.0 < 0.5 else -0.05) if v < 40 else 0.0
    for _ in range(3):
        run(4000, 1, 0, dither)                                 # up to ~top speed
        run(400, 0, 1, lambda i, v: 0.0, stop=lambda v: v < 20)  # hard brake
    # steered constant-speed holds covering the speed range (K data up to ~72 m/s)
    for vh in np.arange(6.0, 73.0, 3.0):
        v = vh
        run(200, 0, 0, lambda i, v: 0.2 * np.sin(i * 0.3) + 0.03)
    r = np.array(rows)
    t = np.arange(len(r)) * DT
    return dict(t=t, v=r[:, 0], throttle=r[:, 1], brake=r[:, 2], steer=r[:, 3], yaw_rate=r[:, 4])


def test_fit_recovers_synthetic():
    log = synth_log()
    p = fit(log)
    for v in (10.0, 40.0, 70.0):
        want = float(true_a_drive(v))
        # true a_drive(70) = -0.25: 5 % of it (0.0125) is below the noise floor, so scale by max(|a|, 1)
        assert abs(a_drive(p, v) - want) <= 0.05 * max(abs(want), 1.0), (v, a_drive(p, v), want)
        k = float(np.interp(v, p["steer_gain_v"], p["steer_gain_k"]))
        assert k > 0
        assert abs(k - true_k(v)) <= 0.05 * true_k(v), (v, k, true_k(v))
    assert abs(p["a_brake"] - 17.0) <= 0.05 * 17.0
    assert len(p["a_drive_lines"]) <= 4
    assert p["v_max"] == log["v"].max()
