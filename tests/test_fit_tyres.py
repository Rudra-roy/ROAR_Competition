import numpy as np

import fit_tyres as ft

TRUE = dict(Cf=70.0, Cr=90.0, mu_f=15.0, mu_r=17.0, a=1.3, b=ft.WHEELBASE - 1.3, I=2.4, tau_d=0.12,
            g_v=list(ft.G_V), g_k=[0.55, 0.45, 0.36, 0.3], dt=ft.DT)


def synth(n_win=3000, h=ft.H, seed=1):
    rng = np.random.default_rng(seed)
    V = np.clip(rng.uniform(12, 65, (n_win, 1)) + rng.normal(0, 0.3, (n_win, h + 1)).cumsum(1), 10, 70)
    amp = rng.uniform(0.0, 0.25, (n_win, 1))
    ph = rng.uniform(0, 2 * np.pi, (n_win, 1))
    S = amp * np.sin(np.linspace(0, 1.5 * np.pi, h + 1)[None, :] + ph) + rng.normal(0, 0.01, (n_win, h + 1))
    S = np.clip(S, -1, 1)
    beta0 = rng.normal(0, 0.03, n_win)
    r0 = rng.normal(0, 0.2, n_win)
    b, r = ft.simulate(TRUE, V, S, beta0, r0)
    BETA = np.c_[beta0, b] + rng.normal(0, 0.002, (n_win, h + 1))
    R = np.c_[r0, r] + rng.normal(0, 0.01, (n_win, h + 1))
    return V, S, BETA, R


def test_fit_recovers_synthetic_parameters():
    V, S, BETA, R = synth()
    p, res = ft.fit(V, S, BETA, R)
    for k in ("Cf", "Cr", "mu_f", "mu_r", "a", "b", "I"):
        assert abs(p[k] / TRUE[k] - 1) < 0.15, (k, p[k])
    assert np.allclose(p["g_k"], TRUE["g_k"], rtol=0.15)
    b, r = ft.simulate(p, V, S, BETA[:, 0], R[:, 0])
    assert np.sqrt(((r - R[:, 1:]) ** 2).mean()) < 0.03


def test_tyre_force_saturates():
    p = dict(TRUE)
    big = np.array([0.5])  # 28 deg front slip: force at the limit, not C * alpha
    beta, r, delta = ft.step(np.zeros(1), np.zeros(1), big, np.array([30.0]), np.zeros(1), p)
    beta_lin = ft.DT * (p["mu_f"] / 30.0)
    assert abs(beta[0] - beta_lin) < 0.1 * beta_lin
    assert beta[0] < ft.DT * p["Cf"] * 0.5 / 30.0 * 0.5
