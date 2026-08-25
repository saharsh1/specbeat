"""Tests for the bounded polynomial-baseline fit (``specbeat.amplitude``).

Column convention: ``construct_log_amp_model_matrix(zeta, trial_v=0,
n_cos_terms=None, n_baseline=2)`` gives ``[1, |ζ|, ζ²]``; default bounds are
``a0`` free, ``a1 ≤ 0``, ``a2 ≤ 0`` (all higher columns free).
"""

from __future__ import annotations

import numpy as np
import pytest

from specbeat import amplitude as amp
from specbeat.simulate import generate_simulated_spectrum, generate_wavelength_grid


@pytest.fixture(scope="module")
def small_design():
    rng = np.random.default_rng(20260509)
    n_pad = 256
    dv = 1.0
    zeta_full = np.fft.rfftfreq(n_pad, d=dv)[1:]
    zeta = zeta_full[zeta_full <= 0.05]
    M = amp.construct_log_amp_model_matrix(zeta, trial_v=0.0, n_cos_terms=None,
                                           n_baseline=2)
    weights = np.ones(M.shape[0])
    return {"zeta": zeta, "M": M, "weights": weights, "rng": rng, "dv": dv,
            "n_pad": n_pad}


def _truth_y(M, theta_true, sigma=1e-4, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    return M @ theta_true + sigma * rng.standard_normal(M.shape[0])


# 1. Backward-compat: unconstrained == bounds=None == inactive box ------------
def test_unconstrained_equivalence_interior(small_design):
    M, weights, rng = small_design["M"], small_design["weights"], small_design["rng"]
    theta_true = np.array([0.5, -2.0, -10.0])
    y = _truth_y(M, theta_true, sigma=1e-4, rng=rng)

    theta_u, cov_u, chi2_u, mask_u = amp.bounded_weighted_least_squares(
        M, y, weights, bounds=None, return_active=True)
    bounds = amp.make_baseline_bounds(n_baseline=2, n_cos_terms=None)
    theta_b, cov_b, chi2_b, mask_b = amp.bounded_weighted_least_squares(
        M, y, weights, bounds=bounds, return_active=True)

    np.testing.assert_allclose(theta_u, theta_b, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(chi2_u, chi2_b, rtol=1e-10)
    assert mask_u.dtype == bool and mask_b.dtype == bool
    assert not mask_u.any() and not mask_b.any()


# 2. a_2 constraint binds; KKT optimality -------------------------------------
def test_constraint_activation_a2(small_design):
    M, weights, rng = small_design["M"], small_design["weights"], small_design["rng"]
    theta_true = np.array([0.2, -20.0, +12.0])
    y = _truth_y(M, theta_true, sigma=1e-4, rng=rng)

    theta_u, _, chi2_u = amp.bounded_weighted_least_squares(M, y, weights, bounds=None)
    theta_c, _, chi2_c, mask = amp.bounded_weighted_least_squares(
        M, y, weights, bounds=amp.make_baseline_bounds(n_baseline=2, n_cos_terms=None),
        return_active=True)

    assert theta_u[2] > 0.0
    assert abs(theta_c[2]) < 1e-9
    assert bool(mask[2]) is True
    assert not mask[0] and not mask[1]
    assert chi2_c >= chi2_u + 1e-6

    resid = y - M @ theta_c
    free_cols = np.where(~mask)[0]
    grad = M[:, free_cols].T @ (weights * resid)
    np.testing.assert_allclose(grad, np.zeros_like(grad), atol=1e-7)


# 3. a_1 activation -----------------------------------------------------------
def test_constraint_activation_a1(small_design):
    M, weights, rng = small_design["M"], small_design["weights"], small_design["rng"]
    theta_true = np.array([0.1, +50.0, -8.0])
    y = _truth_y(M, theta_true, sigma=1e-4, rng=rng)

    theta_u, _, chi2_u = amp.bounded_weighted_least_squares(M, y, weights, bounds=None)
    theta_c, _, chi2_c, mask = amp.bounded_weighted_least_squares(
        M, y, weights, bounds=amp.make_baseline_bounds(n_baseline=2, n_cos_terms=None),
        return_active=True)

    assert theta_u[1] > 0.0
    assert abs(theta_c[1]) < 1e-9
    assert bool(mask[1]) is True
    assert not mask[0]
    assert chi2_c >= chi2_u + 1e-6

    resid = y - M @ theta_c
    free_cols = np.where(~mask)[0]
    grad = M[:, free_cols].T @ (weights * resid)
    np.testing.assert_allclose(grad, np.zeros_like(grad), atol=1e-7)


# 4. Both a_1 and a_2 active --------------------------------------------------
def test_both_constraints_active(small_design):
    M, weights, rng = small_design["M"], small_design["weights"], small_design["rng"]
    theta_true = np.array([0.0, +30.0, +50.0])
    y = _truth_y(M, theta_true, sigma=1e-4, rng=rng)

    theta_u, _, _ = amp.bounded_weighted_least_squares(M, y, weights, bounds=None)
    assert theta_u[1] > 0.0 and theta_u[2] > 0.0

    theta_c, _, _, mask = amp.bounded_weighted_least_squares(
        M, y, weights, bounds=amp.make_baseline_bounds(n_baseline=2, n_cos_terms=None),
        return_active=True)

    assert abs(theta_c[1]) < 1e-9 and abs(theta_c[2]) < 1e-9
    assert bool(mask[1]) is True and bool(mask[2]) is True
    assert not mask[0]

    M0 = M[:, :1]
    # .item() avoids the numpy-2 shape-(1,) -> scalar DeprecationWarning.
    a0_reduced = np.linalg.solve(M0.T @ (weights[:, None] * M0),
                                 M0.T @ (weights * y)).item()
    np.testing.assert_allclose(theta_c[0], a0_reduced, rtol=1e-8)


# 5. Tikhonov ridge -----------------------------------------------------------
def test_ridge_inactive_constraints(small_design):
    M, weights, rng = small_design["M"], small_design["weights"], small_design["rng"]
    theta_true = np.array([0.7, -3.0, -20.0])
    y = _truth_y(M, theta_true, sigma=1e-4, rng=rng)
    lam = 1e-3

    theta_b, _, _ = amp.bounded_weighted_least_squares(
        M, y, weights, bounds=None, lambda_ridge=lam)

    p = M.shape[1]
    MTW = M.T * weights
    A = MTW @ M + lam * np.eye(p)
    theta_ref = np.linalg.solve(A, MTW @ y)
    np.testing.assert_allclose(theta_b, theta_ref, rtol=1e-8, atol=1e-10)

    bounds = amp.make_baseline_bounds(n_baseline=2, n_cos_terms=None)
    theta_bc, _, _, mask = amp.bounded_weighted_least_squares(
        M, y, weights, bounds=bounds, lambda_ridge=lam, return_active=True)
    np.testing.assert_allclose(theta_bc, theta_ref, rtol=1e-8, atol=1e-10)
    assert not mask.any()


# 6. Covariance shape / PD / reduced-block match ------------------------------
def test_covariance_shape_and_dof(small_design):
    M, weights, rng = small_design["M"], small_design["weights"], small_design["rng"]
    theta_true = np.array([0.2, -0.5, +12.0])
    y = _truth_y(M, theta_true, sigma=1e-4, rng=rng)
    lam = 1e-6

    theta, cov, _, mask = amp.bounded_weighted_least_squares(
        M, y, weights, bounds=amp.make_baseline_bounds(n_baseline=2, n_cos_terms=None),
        lambda_ridge=lam, return_active=True)

    p = M.shape[1]
    assert cov.shape == (p, p)
    active = np.where(mask)[0]
    assert active.size >= 1
    for idx in active:
        np.testing.assert_array_equal(cov[idx, :], 0.0)
        np.testing.assert_array_equal(cov[:, idx], 0.0)
    np.testing.assert_allclose(cov, cov.T, atol=1e-12)
    free = np.where(~mask)[0]
    sub = cov[np.ix_(free, free)]
    assert np.linalg.eigvalsh(sub).min() > -1e-12
    M_free = M[:, free]
    A_free = M_free.T @ (weights[:, None] * M_free) + lam * np.eye(free.size)
    np.testing.assert_allclose(sub, np.linalg.inv(A_free), rtol=1e-6, atol=1e-9)


# 7. Synthetic Voigt + rotation: recover (a0, a1, a2) -------------------------
def test_voigt_recovery_synthetic():
    dv = 0.5
    N = 4096
    wavelength = generate_wavelength_grid(wavelength_0=5875.6, dv=dv, N=N)

    vG_kms = 6.0
    gamma_kms = 4.0
    spec, _, _ = generate_simulated_spectrum(
        wavelength=wavelength, dv=dv, N=N, wavelength_0=5875.6, line_depth=0.5,
        v=0.0, vG=vG_kms, gamma=gamma_kms, vsini=0.01, epsilon=0.0)

    from scipy.signal.windows import tukey
    win = tukey(N, alpha=0.1)
    line = spec * win
    amp_arr = np.abs(np.fft.rfft(line))
    zeta = np.fft.rfftfreq(N, d=dv)

    sigma_kms = vG_kms / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    zeta_max = 0.7 / (np.pi * sigma_kms)
    keep = (zeta > 0) & (zeta <= zeta_max)
    zeta_fit = zeta[keep]
    y_fit = np.log(amp_arr[keep])
    weights = np.ones_like(y_fit)

    M = amp.construct_log_amp_model_matrix(zeta_fit, trial_v=0.0, n_cos_terms=None,
                                           n_baseline=2)
    bounds = amp.make_baseline_bounds(n_baseline=2, n_cos_terms=None)
    theta, _, _ = amp.bounded_weighted_least_squares(M, y_fit, weights, bounds=bounds)

    a1_expected = -np.pi * gamma_kms
    a2_expected = -2.0 * (np.pi ** 2) * (sigma_kms ** 2)
    a0_expected = y_fit[0]

    np.testing.assert_allclose(theta[0], a0_expected, rtol=1e-3, atol=1e-2)
    np.testing.assert_allclose(theta[1], a1_expected, rtol=0.05)
    np.testing.assert_allclose(theta[2], a2_expected, rtol=0.05)
    assert theta[1] <= 1e-12 and theta[2] <= 1e-12


# 8. Null (noise-only) false-activation rate ----------------------------------
def test_noise_only_false_activation_rate():
    rng = np.random.default_rng(20260509)
    n_pad = 256
    dv = 1.0
    zeta_full = np.fft.rfftfreq(n_pad, d=dv)[1:]
    zeta = zeta_full[zeta_full <= 0.05]
    M = amp.construct_log_amp_model_matrix(zeta, trial_v=0.0, n_cos_terms=None,
                                           n_baseline=2)
    weights = np.ones(M.shape[0])
    bounds = amp.make_baseline_bounds(n_baseline=2, n_cos_terms=None)

    n_trials = 200
    p = M.shape[1]
    activation_count = np.zeros(p, dtype=int)
    for _ in range(n_trials):
        y = 0.5 * rng.standard_normal(M.shape[0])
        theta, _, chi2, mask = amp.bounded_weighted_least_squares(
            M, y, weights, bounds=bounds, return_active=True)
        assert np.all(np.isfinite(theta))
        assert np.isfinite(chi2)
        activation_count += mask.astype(int)

    rates = activation_count / n_trials
    assert np.all(rates <= 0.85), f"activation rates too high: {rates}"


# 9. amp_noise_sigma closed form ---------------------------------------------
def test_amp_noise_sigma_unwindowed_factor():
    # sigma_A = sigma_pix * sqrt(N/2) for an unwindowed rFFT (Parseval).
    np.testing.assert_allclose(
        amp.amp_noise_sigma(0.02, n=1024), 0.02 * np.sqrt(1024 / 2.0), rtol=1e-12)


def test_amp_noise_sigma_window_factor():
    rng = np.random.default_rng(0)
    window = 0.5 + rng.random(512)
    np.testing.assert_allclose(
        amp.amp_noise_sigma(0.03, window=window),
        0.03 * np.sqrt(np.sum(window ** 2) / 2.0), rtol=1e-12)


def test_amp_noise_sigma_requires_exactly_one_of_n_window():
    with pytest.raises(ValueError):
        amp.amp_noise_sigma(0.02)
    with pytest.raises(ValueError):
        amp.amp_noise_sigma(0.02, n=256, window=np.ones(256))
