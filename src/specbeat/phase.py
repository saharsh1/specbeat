"""Phase channel: the ``arg S̃`` sibling of the amplitude fit (paper eq. 14).

With the primary RV known, shifting each epoch to the primary rest frame puts a
faint companion into the *phase* of the FFT::

    z(ζ) = arg S̃₀(ζ) ≈ -α sin(2π ζ Δv)    (paper eq. 14)

a single template with **no nuisance baseline**. The default
``n_sin_terms = 0`` is the LSF-limited single-amplitude case.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2 as _chi2

from . import nulls

__all__ = ["phase_observable", "phase_template", "fit_phase_alpha"]


def phase_observable(fft, zeta, rest_shift_kms, sigma_ft):
    """Phase observable ``z = arg S̃₀`` and inverse-variance weights (one epoch).

    ``S̃₀ = -fft · exp(+2πi ζ rest_shift)``: the exp shifts to the primary rest
    frame; the leading minus cancels the absorption-dip sign so ``S̃₀`` lands on
    the positive real axis under H0 (off the ``np.angle`` branch cut).

    Parameters
    ----------
    fft : ndarray, complex
        rFFT on the fitted ζ band (DC dropped; caller owns window/pad/mask).
    zeta : ndarray
        Fourier conjugate axis (s/km).
    rest_shift_kms : float
        Shift to the primary rest frame (km/s).
    sigma_ft : float
        Per-epoch FT-amplitude noise std.

    Returns
    -------
    z, w : ndarray, ndarray
        ``arg S̃₀`` and weights ``|S̃₀|² / σ_ft²``.
    """
    S0 = -np.asarray(fft) * np.exp(2j * np.pi * zeta * rest_shift_kms)
    return np.angle(S0), (np.abs(S0) / sigma_ft) ** 2


def phase_template(zeta, dv_rel, n_sin_terms=0):
    """Per-epoch phase design block: columns ``-sin(2π ζ Δv) ζ^p`` (paper eq. 14).

    ``p = 0`` gives ``α`` at the LSF fixed point; higher columns absorb the
    kernel-ratio expansion, which this version does not use. Signed ``dv_rel`` carries the sign
    so opposite-quadrature epochs flip the template while ``α`` stays positive.

    Parameters
    ----------
    zeta : ndarray
        Fourier conjugate axis (s/km).
    dv_rel : float
        Signed companion separation ``Δv = v_b - v_a`` (km/s).
    n_sin_terms : int, optional
        Kernel-ratio polynomial order (default 0 → one column).

    Returns
    -------
    ndarray, shape (len(zeta), n_sin_terms + 1)
    """
    s_base = -np.sin(2 * np.pi * zeta * dv_rel)
    return np.column_stack([s_base * zeta ** p for p in range(n_sin_terms + 1)])


def fit_phase_alpha(fft, zeta, rest_shift_kms, dv_rel, sigma_ft, n_sin_terms=0):
    """Closed-form phase-only LRT for ``α`` at fixed ``Δv`` (paper eq. 14).

    Weighted least squares with no nuisance baseline (``H0: z ≡ 0``). Returns the
    raw (possibly negative) ``α̂`` plus the boundary-aware one-sided statistics
    from :mod:`specbeat.nulls`.

    Parameters
    ----------
    fft : ndarray, complex, shape (n_keep,)
        rFFT on the fitted ζ band. The fit is single-epoch: a 2-D ``(K, n_keep)``
        with ``K > 1`` raises ``NotImplementedError``.
    zeta : ndarray, shape (n_keep,)
        Fourier conjugate axis (s/km).
    rest_shift_kms, dv_rel, sigma_ft : float
        Primary-rest-frame shift, signed ``Δv``, FT-noise std (km/s / km/s / --).
    n_sin_terms : int, optional
        Kernel-ratio order; ``df = n_sin_terms + 1``.

    Returns
    -------
    dict
        ``alpha_hat, sigma_alpha, theta, cov, T, df, p_value, n_sin_terms,
        alpha_constrained, p_value_onesided, significance, alpha_ul95``.
    """
    fft = np.asarray(fft)
    if fft.ndim > 1 and fft.shape[0] > 1:
        raise NotImplementedError(
            "specbeat is single-epoch: fitting a K>1 stack of epochs jointly "
            "is not implemented."
        )
    fft = np.reshape(fft, (-1,))

    z, w = phase_observable(fft, zeta, rest_shift_kms, sigma_ft)
    M = phase_template(zeta, dv_rel, n_sin_terms)

    F = (M.T * w) @ M
    theta = np.linalg.solve(F, (M.T * w) @ z)
    cov = np.linalg.inv(F)

    chi2_H0 = float((w * z * z).sum())               # H0: z ≡ 0 (no companion)
    chi2_H1 = float((w * (z - M @ theta) ** 2).sum())
    T = chi2_H0 - chi2_H1
    df = n_sin_terms + 1

    alpha_hat = float(theta[0])
    sigma_alpha = float(np.sqrt(cov[0, 0]))
    return dict(
        alpha_hat=alpha_hat,
        sigma_alpha=sigma_alpha,
        theta=theta.tolist(),
        cov=cov.tolist(),
        T=float(T),
        df=int(df),
        p_value=float(_chi2.sf(T, df)),
        n_sin_terms=int(n_sin_terms),
        alpha_constrained=max(0.0, alpha_hat),
        p_value_onesided=nulls.pvalue(float(T), df),
        significance=nulls.significance(float(T), df),
        alpha_ul95=nulls.upper_limit(alpha_hat, sigma_alpha, cl=0.95),
    )
