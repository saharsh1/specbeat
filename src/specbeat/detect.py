"""Detection: velocity scan, single-star fit, and the detection statistic.

The detection statistic scans a companion velocity grid ``v_grid`` and forms
``T(v) = χ²₀ − χ²₁(v)``. The rule that matters:
**each ``T(v)`` is clipped to 0 whenever the fitted companion amplitude
``α̂(v) < 0``, BEFORE the max over ``v``**. :func:`blind_null_Tmax` owns this
rule; both :func:`detection_statistic` and :mod:`specbeat.mc` call it, so the
data path and every MC draw are bit-identical.
"""

from __future__ import annotations

import numpy as np

from .amplitude import (
    bounded_weighted_least_squares,
    construct_log_amp_model_matrix,
    fourier_log_amplitude,
    make_baseline_bounds,
    truncate_to_snr_cutoff,
)

__all__ = [
    "fit_velocity",
    "fit_single_star",
    "blind_null_Tmax",
    "detection_statistic",
    "delta_chi2_errors",
    "c0_index",
]


def _prepare_ft(spec, dv, sigma_ft, snr_cutoff, zeta_max):
    """Single-epoch FT + truncation + inverse-variance weights.

    Returns ``(zeta_fit, log_amp_fit, weights, sigma_ft)``. When both cuts are
    None the full positive-ζ band is kept (``truncate_to_snr_cutoff`` requires at
    least one cut, so it is bypassed). ``sigma_ft`` overrides the Rayleigh-tail
    estimate when given — this is what lets the data path and every MC draw share
    an identical noise scale.
    """
    spec = np.asarray(spec, dtype=float)
    amp, _, zeta, sigma_A = fourier_log_amplitude(
        spec, dv, normalize_to_snr=False)
    if sigma_ft is None:
        sigma_ft = float(sigma_A)

    if snr_cutoff is None and zeta_max is None:
        zeta_fit, amp_fit = zeta, amp
    else:
        amp_SNR = amp / sigma_ft
        zeta_fit, _, amp_fit = truncate_to_snr_cutoff(
            zeta, np.log(amp), amp, amp_SNR,
            threshold=snr_cutoff, zeta_max=zeta_max)

    log_amp_fit = np.log(amp_fit)
    weights = (amp_fit / sigma_ft) ** 2
    return zeta_fit, log_amp_fit, weights, float(sigma_ft)


def _wls_at_v(zeta_fit, log_amp_fit, weights, trial_v, n_cos_terms, n_baseline):
    """Bounded WLS of the eq.-12 design at one trial velocity."""
    M = construct_log_amp_model_matrix(
        zeta_fit, trial_v, n_cos_terms=n_cos_terms, n_baseline=n_baseline)
    # bounded_weighted_least_squares right-pads shorter bounds with ±inf.
    bounds = make_baseline_bounds(n_baseline, n_cos_terms=n_cos_terms)
    theta, cov, chi2 = bounded_weighted_least_squares(
        M, log_amp_fit, weights, bounds=bounds)
    return theta, cov, chi2


def _scan(spec, dv, v_grid, sigma_ft, n_cos_terms, n_baseline,
          snr_cutoff, zeta_max):
    """Shared velocity scan: the single source of the clip-before-max surface.

    Returns a dict with the per-v raw ``alpha_hat``, ``chi2_1``, ``sigma_alpha``,
    the H0 ``chi2_0``, and the CLIPPED ``T`` surface (``T=0`` where
    ``alpha_hat<0``). Both :func:`blind_null_Tmax` and
    :func:`detection_statistic` consume this, so the data path and every MC draw
    are bit-identical.
    """
    zeta_fit, log_amp_fit, weights, sigma_ft = _prepare_ft(
        spec, dv, sigma_ft, snr_cutoff, zeta_max)

    _, _, chi2_0 = _wls_at_v(
        zeta_fit, log_amp_fit, weights, 0.0, None, n_baseline)

    ci = c0_index(n_baseline, n_cos_terms)
    v_grid = np.asarray(v_grid, dtype=float)
    alpha = np.empty(v_grid.size)
    chi2_1 = np.empty(v_grid.size)
    sigma_alpha = np.empty(v_grid.size)
    for i, v in enumerate(v_grid):
        theta, cov, c2 = _wls_at_v(
            zeta_fit, log_amp_fit, weights, v, n_cos_terms, n_baseline)
        alpha[i] = theta[ci]
        chi2_1[i] = c2
        sigma_alpha[i] = np.sqrt(cov[ci, ci])

    T = chi2_0 - chi2_1
    T_clipped = np.where(alpha < 0.0, 0.0, T)
    return dict(v_grid=v_grid, alpha_hat=alpha, chi2_1=chi2_1,
                sigma_alpha=sigma_alpha, chi2_0=float(chi2_0),
                T_clipped=T_clipped, sigma_ft=sigma_ft)


def fit_single_star(spec, dv, *, n_baseline=2, sigma_ft=None, snr_cutoff=None,
                    zeta_max=None):
    """Fit the single-star (H0) baseline to the log-|FT| of a spectrum.

    Parameters
    ----------
    spec : array-like, shape (N,)
        Real pixel-space spectrum.
    dv : float
        Velocity step per pixel (km/s).
    n_baseline : int, optional
        Baseline polynomial order.
    sigma_ft : float or None
        FT-amplitude noise std; None → estimate from the Rayleigh tail.
    snr_cutoff, zeta_max : float or None
        Fit-band truncation (see :func:`specbeat.amplitude.truncate_to_snr_cutoff`).

    Returns
    -------
    dict
        ``theta`` (baseline coefficients), ``chi2`` (χ²₀), ``zeta``, ``sigma_ft``.
    """
    zeta_fit, log_amp_fit, weights, sigma_ft = _prepare_ft(
        spec, dv, sigma_ft, snr_cutoff, zeta_max)
    theta, _, chi2 = _wls_at_v(
        zeta_fit, log_amp_fit, weights, 0.0, None, n_baseline)
    return dict(theta=theta, chi2=float(chi2), zeta=zeta_fit, sigma_ft=sigma_ft)


def fit_velocity(spec, dv, v_grid, *, n_cos_terms=0, n_baseline=2, sigma_ft=None,
                 snr_cutoff=None, zeta_max=None):
    """Fit the companion amplitude ``α̂(v)`` and ``χ²₁(v)`` over ``v_grid``.

    For each trial ``v`` this is the bounded WLS of the paper-eq.-12 design
    (baseline bounded, companion cosine columns free). Returns the raw,
    possibly-negative ``α̂(v)`` — the ``α ≥ 0`` clip lives downstream in
    :func:`blind_null_Tmax`.

    Parameters
    ----------
    spec : array-like, shape (N,)
        Real pixel-space spectrum.
    dv : float
        Velocity step per pixel (km/s).
    v_grid : array-like
        Trial companion velocity separations (km/s).
    n_cos_terms : int, optional
        Companion cosine order (default 0).
    n_baseline : int, optional
        Baseline polynomial order.
    sigma_ft : float or None
        FT-amplitude noise std; None → estimate.
    snr_cutoff, zeta_max : float or None
        Fit-band truncation.

    Returns
    -------
    dict
        ``v_grid``, ``alpha_hat`` (per-v raw), ``chi2_1`` (per-v), ``chi2_0``.
    """
    scan = _scan(spec, dv, v_grid, sigma_ft, n_cos_terms, n_baseline,
                 snr_cutoff, zeta_max)
    return dict(v_grid=scan["v_grid"], alpha_hat=scan["alpha_hat"],
                chi2_1=scan["chi2_1"], chi2_0=scan["chi2_0"])


def c0_index(n_baseline=2, n_cos_terms=0):
    """Column index of the companion amplitude ``c0`` in the eq.-12 design.

    Equals the number of baseline columns ``3 + (n_baseline - 2)//2``; the first
    cosine column ``c0`` (the amplitude whose sign the clip rule tests) sits
    immediately after the baseline block. ``n_cos_terms`` is accepted for
    signature symmetry with the design-matrix helpers and does not affect the
    index.

    Parameters
    ----------
    n_baseline : int, optional
        Baseline polynomial order.
    n_cos_terms : int or None, optional
        Cosine order (only its presence matters for the index).

    Returns
    -------
    int
        Index of ``c0`` in the coefficient vector.
    """
    return 3 + max(0, (n_baseline - 2) // 2)


def blind_null_Tmax(spec, dv, v_grid, sigma_ft, *, n_cos_terms=0, n_baseline=2,
                    snr_cutoff=None, zeta_max=None):
    """The clip-before-max detection statistic ``T``.

    Computes ``T(v) = χ²₀ − χ²₁(v)`` over ``v_grid``, sets ``T(v) = 0`` wherever
    the fitted companion amplitude ``α̂(v) < 0`` (the clip), THEN returns
    ``max_v T(v)``. This ordering is the whole point: an unphysical negative
    ripple at any ``v`` contributes no evidence, and the max is taken over the
    clipped surface.

    This single function is shared bit-identically by
    :func:`detection_statistic` and :mod:`specbeat.mc`; a change here changes
    both the data path and the null.

    Parameters
    ----------
    spec : array-like, shape (N,)
        Real pixel-space spectrum.
    dv : float
        Velocity step per pixel (km/s).
    v_grid : array-like
        Trial companion velocity separations (km/s).
    sigma_ft : float
        FT-amplitude noise std (passed explicitly so data and MC agree exactly).
    n_cos_terms, n_baseline : int
        Model orders.
    snr_cutoff, zeta_max : float or None
        Fit-band truncation.

    Returns
    -------
    float
        The clipped ``max_v T(v)``.
    """
    scan = _scan(spec, dv, v_grid, sigma_ft, n_cos_terms, n_baseline,
                 snr_cutoff, zeta_max)
    return float(np.max(scan["T_clipped"]))


def detection_statistic(spec, dv, v_grid, sigma_ft=None, *, n_cos_terms=0,
                        n_baseline=2, snr_cutoff=None, zeta_max=None):
    """Full detection result: clipped ``T``, best ``v``, and the raw ``α̂``.

    Thin wrapper over :func:`blind_null_Tmax` that also reports the argmax ``v``
    and the (raw, possibly-negative) ``α̂`` at that ``v``. ``T`` is bit-identical
    to what :func:`blind_null_Tmax` (hence :mod:`specbeat.mc`) returns for the
    same ``spec`` and ``sigma_ft``.

    Parameters
    ----------
    spec : array-like, shape (N,)
        Real pixel-space spectrum.
    dv : float
        Velocity step per pixel (km/s).
    v_grid : array-like
        Trial companion velocity separations (km/s).
    sigma_ft : float or None
        FT-amplitude noise std; None → estimate from the Rayleigh tail.
    n_cos_terms, n_baseline : int
        Model orders.
    snr_cutoff, zeta_max : float or None
        Fit-band truncation.

    Returns
    -------
    dict
        ``T`` (clipped max), ``v_best``, ``alpha_hat`` (raw at v_best),
        ``sigma_alpha``, ``clip_active`` (bool at v_best), ``sigma_ft``.

    Notes
    -----
    ``sigma_alpha`` is the formal WLS error at ``v_best``, built from weights
    estimated from the Fourier amplitude itself. It is optimistic: on the
    notebooks' demo line a nominal 68% interval covers the true ``α`` about
    59% of the time. Treat it as a scale, and calibrate errors and limits by
    injection (:func:`specbeat.mc.run_injection_mc`) before quoting them.
    """
    scan = _scan(spec, dv, v_grid, sigma_ft, n_cos_terms, n_baseline,
                 snr_cutoff, zeta_max)
    T_clipped = scan["T_clipped"]
    idx = int(np.argmax(T_clipped))
    return dict(
        T=float(np.max(T_clipped)),
        v_best=float(scan["v_grid"][idx]),
        alpha_hat=float(scan["alpha_hat"][idx]),
        sigma_alpha=float(scan["sigma_alpha"][idx]),
        clip_active=bool(scan["alpha_hat"][idx] < 0.0),
        sigma_ft=scan["sigma_ft"],
    )


def delta_chi2_errors(v_grid, chi2_grid, v_best, chi2_best):
    """Δχ² = 1 velocity error bars from a χ²(v) scan.

    Parameters
    ----------
    v_grid : array-like
        Velocity grid (km/s).
    chi2_grid : array-like
        χ² at each grid velocity.
    v_best : float
        Best-fit velocity (km/s).
    chi2_best : float
        χ² at ``v_best``.

    Returns
    -------
    (err_low, err_high) : (float, float)
        Lower/upper 1σ velocity errors (km/s).
    """
    v_grid = np.asarray(v_grid, dtype=float)
    chi2_grid = np.asarray(chi2_grid, dtype=float)
    order = np.argsort(v_grid)
    v_grid = v_grid[order]
    chi2_grid = chi2_grid[order]
    target = chi2_best + 1.0

    def _cross(vs, cs):
        # First Δχ²=1 crossing of (vs, cs) walking away from v_best; linear
        # interpolation between the bracketing grid points. NaN if uncrossed.
        for i in range(len(vs) - 1):
            c0, c1 = cs[i], cs[i + 1]
            if (c0 - target) * (c1 - target) <= 0.0 and c1 != c0:
                frac = (target - c0) / (c1 - c0)
                return vs[i] + frac * (vs[i + 1] - vs[i])
        return np.nan

    left = v_grid <= v_best
    right = v_grid >= v_best
    v_lo = _cross(v_grid[left][::-1], chi2_grid[left][::-1])
    v_hi = _cross(v_grid[right], chi2_grid[right])
    err_low = float(v_best - v_lo) if np.isfinite(v_lo) else np.nan
    err_high = float(v_hi - v_best) if np.isfinite(v_hi) else np.nan
    return err_low, err_high
