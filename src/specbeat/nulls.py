"""Boundary-aware (one-sided) null for the faint-companion detection statistic.

Because the companion amplitude ``α ≥ 0``, the null ``α = 0`` sits on the
*boundary*, so the fixed-``v`` null of ``T = χ²₀ − χ²₁`` is the Chernoff /
Self & Liang (1987) mixture::

    df = 1 :  T ~ ½·δ(0) + ½·χ²₁
    df > 1 :  T ~ ½·χ²_{df−1} + ½·χ²_df

The multi-epoch, multi-window machinery of the paper's Sec. 4 is not included:
everything here is single-epoch and single-window.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import brentq
from scipy.special import polygamma
from scipy.stats import chi2, gamma, kstest, norm

__all__ = [
    "detection_threshold",
    "pvalue",
    "significance",
    "upper_limit",
    "constrain_nonneg",
    "fit_effective_chi2",
    "effective_chi2_pvalue",
    "effective_chi2_pvalue_error",
    "fit_gamma_null",
    "gamma_null_pvalue",
    "gamma_null_pvalue_error",
    "gamma_null_threshold",
]


def _chi2_sf(x, k):
    """Survival ``P(χ²_k >= x)`` with ``χ²₀`` a point mass at 0.

    ``scipy.stats.chi2.sf(x, 0)`` is NaN, but the Chernoff mixture needs the
    ``df-1 == 0`` term (the χ²₀ point mass): ``P(χ²₀ >= x) = 0`` for ``x > 0``
    and ``1`` for ``x <= 0``. Preserves the scalar-in → scalar-out contract.
    """
    if k == 0:
        out = np.where(np.asarray(x, float) <= 0.0, 1.0, 0.0)
        return out if np.ndim(x) else float(out)
    return chi2.sf(x, k)


def detection_threshold(fap=0.05, df=1):
    """Threshold ``c`` on ``T`` for false-alarm probability ``fap``.

    ``df = 1``: ``c = χ²₁.ppf(1 − 2·fap)`` = 2.706 at ``fap = 0.05`` (NOT 3.841).
    ``df > 1``: numeric root of the mixture tail.

    Parameters
    ----------
    fap : float
        False-alarm probability in (0, 0.5).
    df : int
        Number of companion parameters.

    Returns
    -------
    float
        The threshold ``c``.
    """
    if not 0.0 < fap < 0.5:
        raise ValueError("fap must be in (0, 0.5)")
    if df == 1:
        return float(chi2.ppf(1.0 - 2.0 * fap, df=1))

    def tail(c):
        return 0.5 * (_chi2_sf(c, df - 1) + _chi2_sf(c, df)) - fap

    upper = chi2.ppf(1.0 - fap, df)
    return float(brentq(tail, 0.0, upper))


def pvalue(T, df=1):
    """One-sided mixture-tail p-value under the boundary null.

    ``T > 0``: ``½·(χ²_{df−1}.sf(T) + χ²_df.sf(T))`` (``= ½·χ²₁.sf(T)`` at df=1,
    using the χ²₀ point-mass guard for ``chi2.sf(·, 0)``). ``T ≤ 0``: ``1.0`` (no
    evidence at all — the atom is inside the right tail there).

    Parameters
    ----------
    T : float or ndarray
        Detection statistic (χ² units).
    df : int
        Number of companion parameters.

    Returns
    -------
    float or ndarray
        One-sided p-value.
    """
    T = np.asarray(T, float)
    tail = 0.5 * (_chi2_sf(T, df - 1) + _chi2_sf(T, df))
    p = np.where(T > 0.0, tail, 1.0)
    return p if p.ndim else float(p)


def significance(T, df=1):
    """Gaussian-σ significance from the one-sided p-value.

    ``σ = Φ⁻¹(1 − p)``; for ``df = 1, T > 0`` this equals ``√T``. ``T ≤ 0 → 0``.

    Parameters
    ----------
    T : float or ndarray
        Detection statistic (χ² units).
    df : int
        Number of companion parameters.

    Returns
    -------
    float or ndarray
        Significance in Gaussian σ.
    """
    p = pvalue(T, df)
    sig = norm.isf(p)
    sig = np.asarray(sig, float)
    sig = np.where(np.asarray(T, float) > 0.0, sig, 0.0)
    return sig if sig.ndim else float(sig)


def upper_limit(alpha_hat, sigma_alpha, cl=0.95, power_constrained=False):
    """One-sided UL on ``α`` from the truncated-Gaussian posterior.

    ``cl``-th percentile of ``N(α̂, σ_α)`` truncated to ``α ≥ 0``. Consumes the
    RAW (possibly negative) ``α̂``. Implemented in the numerically stable
    ``UL = α̂ + σ_α·Φ̄⁻¹(s·(1−cl))``, ``s = Φ(α̂/σ_α)``, which stays finite far
    into ``α̂ < 0`` (the naive form rounds to ``Φ⁻¹(1) = ∞`` below ~−8σ).

    Parameters
    ----------
    alpha_hat : float
        Raw MLE of the companion amplitude.
    sigma_alpha : float
        1σ error on ``α̂``.
    cl : float
        Credible level (default 0.95).
    power_constrained : bool
        If True, floor at ``Φ⁻¹(cl)·σ_α`` (Cowan et al. 2011).

    Returns
    -------
    float
        The one-sided upper limit on ``α``.

    Notes
    -----
    ``sigma_alpha`` is usually the formal WLS error, built from weights that
    are themselves estimated from the Fourier amplitude. Those weights make
    the formal error optimistic: on the notebooks' demo line a nominal 68%
    interval covers the true ``α`` about 59% of the time. A limit meant to be
    quoted should be calibrated by injection (:func:`specbeat.mc.run_injection_mc`)
    rather than read off ``sigma_alpha`` alone.
    """
    s = norm.cdf(alpha_hat / sigma_alpha)
    ul = alpha_hat + sigma_alpha * norm.isf(s * (1.0 - cl))
    if power_constrained:
        ul = max(ul, norm.ppf(cl) * sigma_alpha)
    return float(ul)


def constrain_nonneg(alpha_hat, sigma_alpha, T_uncon, df):
    """Apply the post-hoc ``α ≥ 0`` constraint to a raw fit.

    Clamps ``α̂ → max(0, α̂)`` and zeroes ``T`` when ``α̂ < 0``; the p-value is the
    one-sided tail at the clamped ``T`` (``= 1`` when the constraint is active).
    The UL is built from the RAW ``α̂`` (not clamped).

    Parameters
    ----------
    alpha_hat : float
        Raw MLE of the companion amplitude.
    sigma_alpha : float
        1σ error on ``α̂``.
    T_uncon : float
        Raw unconstrained statistic.
    df : int
        Number of companion parameters.

    Returns
    -------
    dict
        ``alpha_hat`` (clamped), ``constraint_active``, ``T``, ``df``,
        ``p_value``, ``alpha_ul95``, ``alpha_uncon``, ``sigma_alpha``.
    """
    active = alpha_hat < 0.0
    T = 0.0 if active else float(T_uncon)
    return dict(
        alpha_hat=max(0.0, float(alpha_hat)),
        constraint_active=bool(active),
        T=T,
        df=int(df),
        p_value=pvalue(T, df),
        alpha_ul95=upper_limit(alpha_hat, sigma_alpha, cl=0.95),
        alpha_uncon=float(alpha_hat),
        sigma_alpha=float(sigma_alpha),
    )


def fit_effective_chi2(samples):
    """Mean-match the effective-χ² null to an MC Δχ² sample.

    ``T ~ w0·δ(0) + (1 − w0)·χ²_{ν_eff}``: ``w0 = mean(samples ≤ 0)``,
    ``ν_eff = mean(samples[samples > 0])`` (zeros excluded — including them biases
    ``ν_eff`` low), ``var_ratio = var(pos)/(2·ν_eff)`` (≈ 1 for a true χ²).

    Parameters
    ----------
    samples : ndarray
        MC draws of the constrained statistic (clipped draws at 0).

    Returns
    -------
    dict
        ``nu_eff``, ``w0``, ``var_ratio``.
    """
    samples = np.asarray(samples, float)
    w0 = float(np.mean(samples <= 0.0))
    pos = samples[samples > 0.0]
    if pos.size == 0:
        return dict(nu_eff=float("nan"), w0=w0, var_ratio=float("nan"))
    nu_eff = float(np.mean(pos))
    var_ratio = float(np.var(pos) / (2.0 * nu_eff))
    return dict(nu_eff=nu_eff, w0=w0, var_ratio=var_ratio)


def effective_chi2_pvalue(T, nu_eff, w0):
    """p-value under the MC-calibrated effective-χ² null.

    ``T > 0``: ``(1 − w0)·χ²_{ν_eff}.sf(T)``; ``T ≤ 0``: ``1.0``. Reduces exactly
    to :func:`pvalue` (df=1) at ``(ν_eff=1, w0=½)``.

    Parameters
    ----------
    T : float or ndarray
        Detection statistic (χ² units).
    nu_eff : float
        Effective dof of the continuum.
    w0 : float
        Boundary atom mass in [0, 1].

    Returns
    -------
    float or ndarray
        One-sided p-value.
    """
    T = np.asarray(T, float)
    tail = (1.0 - w0) * chi2.sf(T, nu_eff)
    p = np.where(T > 0.0, tail, 1.0)
    return p if p.ndim else float(p)


def effective_chi2_pvalue_error(T, samples):
    """Delta-method std of ``ln p_param`` from one MC sample.

    ``Var(ln p) ≈ (∂ln S/∂ν)²·Var(ν̂) + w0/((1−w0)·n_total)`` with
    ``Var(ν̂) = var(pos)/n_pos`` and ``∂ln S/∂ν`` by central difference.
    Degenerate samples → all-nan dict (no raise).

    Parameters
    ----------
    T : float
        Observed statistic (χ² units).
    samples : ndarray
        MC draws (clipped draws at 0).

    Returns
    -------
    dict
        ``sd_ln_p``, ``sd_log10_p``, ``dlnS_dnu``, ``var_nu``, ``var_w0_term``, ``fit``.
    """
    samples = np.asarray(samples, float)
    fit = fit_effective_chi2(samples)
    nu, w0 = fit["nu_eff"], fit["w0"]
    pos = samples[samples > 0.0]
    if pos.size == 0 or not np.isfinite(nu) or w0 >= 1.0:
        return dict(sd_ln_p=float("nan"), sd_log10_p=float("nan"),
                    dlnS_dnu=float("nan"), var_nu=float("nan"),
                    var_w0_term=float("nan"), fit=fit)

    T = float(T)
    eps = max(1e-4 * nu, 1e-6)
    s_hi = chi2.sf(T, nu + eps)
    s_lo = chi2.sf(T, nu - eps)
    if not (np.isfinite(s_hi) and np.isfinite(s_lo)) or s_hi <= 0.0 or s_lo <= 0.0:
        return dict(sd_ln_p=float("nan"), sd_log10_p=float("nan"),
                    dlnS_dnu=float("nan"), var_nu=float("nan"),
                    var_w0_term=float("nan"), fit=fit)
    dlns_dnu = (np.log(s_hi) - np.log(s_lo)) / (2.0 * eps)
    var_nu = float(np.var(pos)) / pos.size
    var_w0_term = w0 / ((1.0 - w0) * samples.size)
    sd_ln_p = float(np.sqrt(dlns_dnu ** 2 * var_nu + var_w0_term))
    return dict(sd_ln_p=sd_ln_p, sd_log10_p=sd_ln_p / np.log(10.0),
                dlnS_dnu=float(dlns_dnu), var_nu=var_nu,
                var_w0_term=float(var_w0_term), fit=fit)


def fit_gamma_null(samples, min_pos=30):
    """ML-fit the Gamma-continuum null to an MC Δχ² sample.

    ``T ~ w0·δ(0) + (1 − w0)·Gamma(a, θ)`` with ``loc`` PINNED at 0
    (``gamma.fit(pos, floc=0)``). Returns the KS ``D`` statistic only (no KS
    p-value — the estimated-parameter/Lilliefors p-value is invalid). Fewer than
    ``min_pos`` positive draws → nan fit (no raise).

    Parameters
    ----------
    samples : ndarray
        MC draws of the constrained statistic (clipped draws at 0).
    min_pos : int
        Minimum positive draws to attempt the ML fit.

    Returns
    -------
    dict
        ``w0``, ``a``, ``theta``, ``ks_stat``, ``n_pos``, ``mean_pos``, ``var_pos``.

    Warns
    -----
    RuntimeWarning
        When ``a < 1`` (density diverges at the origin; far-tail fragile).
    """
    samples = np.asarray(samples, float)
    w0 = float(np.mean(samples <= 0.0))
    pos = samples[samples > 0.0]
    if pos.size < min_pos or np.ptp(pos) == 0.0:
        return dict(w0=w0, a=float("nan"), theta=float("nan"),
                    ks_stat=float("nan"), n_pos=int(pos.size),
                    mean_pos=float(np.mean(pos)) if pos.size else float("nan"),
                    var_pos=float(np.var(pos)) if pos.size else float("nan"))
    try:
        a, _loc, theta = gamma.fit(pos, floc=0)
    except (ValueError, RuntimeError):
        return dict(w0=w0, a=float("nan"), theta=float("nan"),
                    ks_stat=float("nan"), n_pos=int(pos.size),
                    mean_pos=float(np.mean(pos)), var_pos=float(np.var(pos)))
    ks_stat = float(kstest(pos, "gamma", args=(a, 0.0, theta)).statistic)
    if a < 1.0:
        warnings.warn(
            "Gamma shape a < 1: density diverges at the origin, far-tail "
            "extrapolation is fragile.",
            RuntimeWarning,
        )
    return dict(w0=w0, a=float(a), theta=float(theta), ks_stat=ks_stat,
                n_pos=int(pos.size), mean_pos=float(np.mean(pos)),
                var_pos=float(np.var(pos)))


def gamma_null_pvalue(T, a, theta, w0):
    """p-value under the Gamma-continuum boundary null.

    ``T > 0``: ``(1 − w0)·Gamma(a, θ).sf(T)``; ``T ≤ 0``: ``1.0``. Nests the
    effective-χ² null at ``(a=ν/2, θ=2)`` and the df=1 Chernoff at
    ``(a=½, θ=2, w0=½)``.

    Parameters
    ----------
    T : float or ndarray
        Detection statistic (χ² units).
    a : float
        Gamma shape.
    theta : float
        Gamma scale (χ² units).
    w0 : float
        Boundary atom mass in [0, 1].

    Returns
    -------
    float or ndarray
        One-sided p-value.
    """
    T = np.asarray(T, float)
    tail = (1.0 - w0) * gamma.sf(T, a, scale=theta)
    p = np.where(T > 0.0, tail, 1.0)
    return p if p.ndim else float(p)


def gamma_null_pvalue_error(T, samples, cov="fisher"):
    """Delta-method std of ``ln p`` under the Gamma null.

    ``Var(ln p) ≈ g·Cov(â, θ̂)·gᵀ + w0/((1−w0)·n_total)`` with the inverse
    observed-Fisher block (trigamma ``ψ₁``, ``D = a·ψ₁ − 1``) and
    ``g = [∂ln S/∂a, ∂ln S/∂θ]`` by central difference. Degenerate → all-nan.

    Parameters
    ----------
    T : float
        Observed statistic (χ² units).
    samples : ndarray
        MC draws (clipped draws at 0).
    cov : str
        Only ``"fisher"`` is implemented.

    Returns
    -------
    dict
        ``sd_ln_p``, ``sd_log10_p``, ``ga``, ``gth``, ``var_a``, ``var_theta``,
        ``cov_at``, ``fit``.
    """
    if cov != "fisher":
        raise NotImplementedError(
            "only the 'fisher' (inverse observed Fisher) covariance is "
            "implemented"
        )
    samples = np.asarray(samples, float)
    fit = fit_gamma_null(samples)
    a, theta, w0, n_pos = fit["a"], fit["theta"], fit["w0"], fit["n_pos"]
    pos = samples[samples > 0.0]

    def _nan_result():
        return dict(sd_ln_p=float("nan"), sd_log10_p=float("nan"),
                    ga=float("nan"), gth=float("nan"), var_a=float("nan"),
                    var_theta=float("nan"), cov_at=float("nan"), fit=fit)

    if (pos.size == 0 or not np.isfinite(a) or not np.isfinite(theta)
            or w0 >= 1.0):
        return _nan_result()

    psi1 = polygamma(1, a)
    D = a * psi1 - 1.0
    var_a = a / (D * n_pos)
    var_theta = theta ** 2 * psi1 / (D * n_pos)
    cov_at = -theta / (D * n_pos)

    T = float(T)
    eps_a = max(1e-4 * a, 1e-6)
    s_a_hi = gamma.sf(T, a + eps_a, scale=theta)
    s_a_lo = gamma.sf(T, a - eps_a, scale=theta)
    eps_th = max(1e-4 * theta, 1e-6)
    s_th_hi = gamma.sf(T, a, scale=theta + eps_th)
    s_th_lo = gamma.sf(T, a, scale=theta - eps_th)
    if not all(np.isfinite([s_a_hi, s_a_lo, s_th_hi, s_th_lo])) or \
            min(s_a_hi, s_a_lo, s_th_hi, s_th_lo) <= 0.0:
        return _nan_result()

    ga = (np.log(s_a_hi) - np.log(s_a_lo)) / (2.0 * eps_a)
    gth = (np.log(s_th_hi) - np.log(s_th_lo)) / (2.0 * eps_th)
    var_shape = (ga ** 2 * var_a + 2.0 * ga * gth * cov_at
                 + gth ** 2 * var_theta)
    var_w0 = w0 / ((1.0 - w0) * samples.size)
    sd_ln_p = float(np.sqrt(var_shape + var_w0))
    return dict(sd_ln_p=sd_ln_p, sd_log10_p=sd_ln_p / np.log(10.0),
                ga=float(ga), gth=float(gth), var_a=float(var_a),
                var_theta=float(var_theta), cov_at=float(cov_at), fit=fit)


def gamma_null_threshold(fap, a, theta, w0):
    """Invert the Gamma null: the threshold ``c`` for a target FAP.

    Solves ``(1 − w0)·Gamma(a, θ).sf(c) = fap`` for ``c``. Since the continuum
    carries only mass ``1 − w0``, a FAP not reachable by the continuum returns
    the boundary::

        c = Gamma(a, θ).isf(fap / (1 − w0))    if fap < 1 − w0
        c = 0.0                                if fap >= 1 − w0

    (When ``fap >= 1 − w0`` the requested tail probability is at or below the
    atom mass, so the threshold collapses to the boundary at ``T = 0``.)

    Parameters
    ----------
    fap : float
        Target false-alarm probability.
    a : float
        Gamma shape.
    theta : float
        Gamma scale (χ² units).
    w0 : float
        Boundary atom mass in [0, 1].

    Returns
    -------
    float
        The threshold ``c`` on ``T``.
    """
    if fap >= 1.0 - w0:
        return 0.0
    return float(gamma.isf(fap / (1.0 - w0), a, scale=theta))
