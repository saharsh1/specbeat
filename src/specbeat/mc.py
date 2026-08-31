"""Monte-Carlo null / injection-recovery for the detection statistic.

The null and injection MCs both compute each draw's statistic through the SAME
:func:`specbeat.detect.blind_null_Tmax` the data path uses (MC = data).
Supports white and AR(1) red noise. The empirical (rank) p-value
:func:`p_emp` is the model-free cross-check.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

from . import nulls
from .detect import blind_null_Tmax
from .simulate import generate_simulated_spectrum, generate_wavelength_grid

__all__ = [
    "make_template",
    "run_null_mc",
    "run_injection_mc",
    "calibrate_null",
    "completeness",
    "p_emp",
]


def _draw_noise(rng, n_pix, sigma_pix, dv, noise):
    """One noise realization: white Gaussian, or AR(1) red noise.

    AR(1): ``x_i = ρ x_{i-1} + √(1−ρ²) ε_i`` with ``ρ = exp(-dv / l_corr)`` and
    ``ε_i ~ N(0, σ_pix)``. The ``√(1−ρ²)`` innovation scaling keeps the marginal
    per-pixel variance at ``σ_pix²`` regardless of ``ρ``.
    """
    if noise == "white":
        return rng.normal(0.0, sigma_pix, n_pix)
    kind, l_corr = noise
    if kind != "ar1":
        raise ValueError(f"unknown noise model {noise!r}")
    rho = np.exp(-dv / l_corr)
    eps = rng.normal(0.0, sigma_pix, n_pix)
    # x = rho*x_prev + sqrt(1-rho^2)*eps is a one-pole IIR filter; lfilter runs
    # the recursion in C. The first innovation enters unscaled so that x[0]
    # already has the stationary variance sigma_pix^2.
    eps[1:] *= np.sqrt(1.0 - rho ** 2)
    return lfilter([1.0], [1.0, -rho], eps)


def make_template(dv, n_pix, *, wavelength_0=5875.6, line_depth=0.5, vG=20.0,
                  gamma=5.0, vsini=0.0, epsilon=0.5):
    """Build the single-star (H0) template used for injection and fitting.

    Parameters
    ----------
    dv : float
        Velocity step per pixel (km/s).
    n_pix : int
        Spectrum length.
    wavelength_0, line_depth, vG, gamma, vsini, epsilon : float
        Primary line parameters (FWHM km/s where applicable).

    Returns
    -------
    ndarray, shape (n_pix,)
        The noiseless single-star template (continuum-subtracted).
    """
    dv = float(dv)
    wave = generate_wavelength_grid(wavelength_0, dv, n_pix)
    spec, _, _ = generate_simulated_spectrum(
        wave, dv, N=n_pix, wavelength_0=wavelength_0, line_depth=line_depth,
        v=0.0, vG=vG, gamma=gamma, vsini=vsini, epsilon=epsilon,
        second_component=False)
    return np.asarray(spec, dtype=float)


def run_null_mc(sigma_pix, n_pix, dv, v_grid, n_draws, *, seed=0, noise="white",
                sigma_ft=None, template=None, n_cos_terms=0, n_baseline=2,
                snr_cutoff=None, zeta_max=None, return_draws=False):
    """Monte-Carlo null: statistics of ``T`` under H0 (no companion).

    Each draw is a noise realization (optionally on top of the single-star
    ``template``), passed through :func:`specbeat.detect.blind_null_Tmax` with the
    SAME ``sigma_ft`` the data fit uses. This shared owner of the clip-before-max
    rule is what makes each draw bit-identical to
    :func:`specbeat.detect.detection_statistic` on the same spectrum.

    Parameters
    ----------
    sigma_pix : float
        Per-pixel noise std.
    n_pix : int
        Spectrum length.
    dv : float
        Velocity step per pixel (km/s).
    v_grid : array-like
        Trial companion velocity grid (km/s).
    n_draws : int
        Number of null draws.
    seed : int
        RNG seed.
    noise : {"white"} or ("ar1", l_corr_kms)
        White Gaussian noise, or AR(1) red noise with correlation length
        ``l_corr_kms`` (km/s).
    sigma_ft : float or None
        FT-amplitude noise std shared with the data fit; None → estimate.
    template : ndarray or None
        Single-star profile to inject under each draw (None → pure noise).
    n_cos_terms, n_baseline : int
        Model orders.
    snr_cutoff, zeta_max : float or None
        Fit-band truncation.
    return_draws : bool
        If True, also return the per-draw spectra (for bit-identity checks).

    Returns
    -------
    dict
        ``T`` (ndarray, shape (n_draws,)); ``draws`` (ndarray, shape
        (n_draws, n_pix)) only when ``return_draws``.
    """
    rng = np.random.default_rng(seed)
    base = np.zeros(n_pix) if template is None else np.asarray(template, float)
    T = np.empty(n_draws)
    draws = np.empty((n_draws, n_pix)) if return_draws else None
    for i in range(n_draws):
        spec = base + _draw_noise(rng, n_pix, sigma_pix, dv, noise)
        T[i] = blind_null_Tmax(
            spec, dv, v_grid, sigma_ft, n_cos_terms=n_cos_terms,
            n_baseline=n_baseline, snr_cutoff=snr_cutoff, zeta_max=zeta_max)
        if return_draws:
            draws[i] = spec
    out = dict(T=T)
    if return_draws:
        out["draws"] = draws
    return out


def run_injection_mc(alpha, delta_v, sigma_pix, n_pix, dv, v_grid, n_draws, *,
                     seed=0, noise="white", sigma_ft=None, template=None,
                     n_cos_terms=0, n_baseline=2, snr_cutoff=None, zeta_max=None,
                     fap=0.05, null_T=None):
    """Injection-recovery: statistics of ``T`` under an injected companion.

    ``T`` here is the velocity-*scanned* ``blind_null_Tmax`` (max over
    ``v_grid``). Its null is heavier-tailed than the fixed-``v`` df=1 Chernoff
    mixture (look-elsewhere; Davies 1977), so a detection decision must
    threshold against a **matching v-scanned null**, NOT the fixed-``v`` value
    ``detection_threshold(fap, df=1)`` (= 2.706), which would be anti-conservative
    and inflate the reported completeness. Accordingly:

    - if ``null_T`` (a null sample from :func:`run_null_mc` run with the *same*
      scan) is supplied, ``detected`` is thresholded at the Gamma-null level
      ``gamma_null_threshold(fap, …)`` calibrated from it, and the threshold is
      returned;
    - otherwise only ``T`` is returned, and the caller should threshold via
      :func:`calibrate_null` + :func:`specbeat.nulls.gamma_null_threshold`
      (see :func:`completeness`).

    Parameters
    ----------
    alpha : float
        Injected companion amplitude.
    delta_v : float
        Injected companion separation (km/s).
    sigma_pix, n_pix, dv, v_grid, n_draws, seed, noise, sigma_ft, template,
    n_cos_terms, n_baseline, snr_cutoff, zeta_max :
        As in :func:`run_null_mc`.
    fap : float
        False-alarm probability defining the detection threshold. Only used when
        ``null_T`` is supplied.
    null_T : ndarray, optional
        A matching v-scanned MC null sample (from :func:`run_null_mc`). When
        given, the returned dict includes ``detected`` and ``threshold``.

    Returns
    -------
    dict
        ``T`` (ndarray); plus ``detected`` (bool ndarray) and ``threshold``
        (float) when ``null_T`` is supplied.
    """
    rng = np.random.default_rng(seed)
    if template is None:
        raise ValueError("run_injection_mc requires a single-star template")
    template = np.asarray(template, float)
    if template.size != n_pix:
        raise ValueError(
            f"template length {template.size} does not match n_pix={n_pix}"
        )

    # Companion = the primary profile scaled by alpha and shifted by delta_v
    # (a pure Fourier phase shift), added under the H0 template.
    zeta = np.fft.rfftfreq(n_pix, d=dv)
    companion = np.fft.irfft(
        np.fft.rfft(template) * np.exp(-2j * np.pi * zeta * delta_v), n=n_pix)
    injected = template + alpha * companion

    T = np.empty(n_draws)
    for i in range(n_draws):
        spec = injected + _draw_noise(rng, n_pix, sigma_pix, dv, noise)
        T[i] = blind_null_Tmax(
            spec, dv, v_grid, sigma_ft, n_cos_terms=n_cos_terms,
            n_baseline=n_baseline, snr_cutoff=snr_cutoff, zeta_max=zeta_max)

    out = dict(T=T)
    if null_T is not None:
        fit = nulls.fit_gamma_null(np.asarray(null_T, float))
        c = nulls.gamma_null_threshold(fap, fit["a"], fit["theta"], fit["w0"])
        out["threshold"] = c
        out["detected"] = T >= c
    return out


def calibrate_null(mc_T):
    """Summarize an MC null sample into calibration parameters.

    Bundles :func:`specbeat.nulls.fit_gamma_null` (headline) and
    :func:`specbeat.nulls.fit_effective_chi2` (cross-check) plus the boundary
    atom mass on one draw array.

    Parameters
    ----------
    mc_T : ndarray
        MC draws of the constrained statistic (clipped draws at 0).

    Returns
    -------
    dict
        ``gamma`` (fit dict), ``effective_chi2`` (fit dict), ``w0``.
    """
    mc_T = np.asarray(mc_T, float)
    gamma_fit = nulls.fit_gamma_null(mc_T)
    eff_fit = nulls.fit_effective_chi2(mc_T)
    return dict(gamma=gamma_fit, effective_chi2=eff_fit, w0=eff_fit["w0"])


def completeness(inj_T, threshold):
    """Fraction of injection draws exceeding a detection threshold.

    Parameters
    ----------
    inj_T : ndarray
        Injection-recovery draws of ``T``.
    threshold : float
        Detection threshold ``c``.

    Returns
    -------
    float
        Completeness (empirical detection power) in [0, 1].
    """
    inj_T = np.asarray(inj_T, float)
    finite = inj_T[np.isfinite(inj_T)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite >= threshold))


def p_emp(T_obs, T_null):
    """Rank-based empirical p-value.

    ``p_emp = (1 + #{T_null ≥ T_obs}) / (n_finite + 1)`` using ``≥`` (not ``>``)
    and counting only FINITE null draws. The ``+1`` regularisation (North et al.
    2002; Davison & Hinkley 1997) makes it a valid, slightly-conservative p-value
    for any ``n``.

    Parameters
    ----------
    T_obs : float
        Observed statistic.
    T_null : ndarray
        MC null draws (NaNs are dropped from the denominator).

    Returns
    -------
    float
        The empirical p-value.
    """
    T_null = np.asarray(T_null, float)
    finite = T_null[np.isfinite(T_null)]
    n = finite.size
    k = int(np.sum(finite >= T_obs))
    return (1 + k) / (n + 1)
