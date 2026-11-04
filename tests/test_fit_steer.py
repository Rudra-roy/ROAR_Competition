import numpy as np

from fit_steer import DT, fit_beta, fit_yaw, lag


def synth(seed, tau=0.35, k_of_v=lambda v: 0.28 + 0.0 * v, kb=0.01, tb=0.6):
    """A straight at slowly varying speed with a random smooth steer; yaw curvature and sideslip from lag models."""
    rng = np.random.default_rng(seed)
    n = 6000
    v = 50 + 18 * np.sin(np.arange(n) / 700.0)
    u = np.convolve(rng.standard_normal(n), np.ones(20) / 20, "same") * 0.01
    x_b = v * v * lag(0.28 * u, 0.2)
    return dict(u=u, v=v, ky=lag(k_of_v(v) * u, tau) + 2e-5 * rng.standard_normal(n), x_b=x_b,
                beta=kb * lag(x_b, tb) + 1e-4 * rng.standard_normal(n), ok=v >= 45)


def test_recovers_yaw_lag_and_band_gain():
    kv = lambda v: 0.36 - 0.0012 * v
    tau, bands = fit_yaw([synth(s, 0.35, kv) for s in range(3)])
    assert abs(tau - 0.35) < 0.026
    assert len(bands) >= 4
    for v, k, n, r2 in bands:
        assert abs(k - kv(v)) < 0.01 * kv(v), (v, k, kv(v))
        assert r2 > 0.95


def test_recovers_sideslip_lag_and_gain():
    kb, tb, r2 = fit_beta([synth(s, kb=0.0095, tb=0.65) for s in range(3)])
    assert abs(tb - 0.65) < 0.026 and abs(kb - 0.0095) < 0.03 * 0.0095 and r2 > 0.95


def test_lag_is_one_tick_late_first_order():
    x = np.r_[0.0, np.ones(50)]
    f = lag(x, 0.2)
    assert f[1] == 0.0 and abs(f[2] - DT / 0.2) < 1e-12
