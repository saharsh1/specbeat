"""Amplitude channel: log-|FT| observable, design matrix, (bounded) WLS.

The observable is ``log|S̃(ζ)|`` on the Fourier grid ``np.fft.rfftfreq(N, d=dv)[1:]``
(DC dropped, positive freqs only). The single-star baseline is the polynomial
``[1, |ζ|, ζ², …]``; a companion adds cosine columns (paper eq. 12). The default
``n_cos_terms = 0`` is the LSF-limited single-amplitude case.
"""

from __future__ import annotations

import numpy as np
from numpy.fft import rfft, rfftfreq

__all__ = [
    "fourier_log_amplitude",
    "truncate_to_snr_cutoff",
    "construct_log_amp_model_matrix",
    "make_baseline_bounds",
    "weighted_least_squares",
    "bounded_weighted_least_squares",
    "amp_noise_sigma",
]

#: Rayleigh distribution: median = sigma_R * sqrt(2 * ln 2) ~ 1.1774.
RAYLEIGH_MEDIAN_TO_SIGMA = np.sqrt(2 * np.log(2))


def fourier_log_amplitude(spec, sampling_kms, normalize_to_snr=True,
                          tail_fraction=0.3):
    """Fourier log-amplitude of a spectrum, DC dropped.

    Returns ``|FT|``, ``log|FT|`` (optionally SNR-normalized), the ζ grid
    (``rfftfreq[1:]``, so ``zeta[0] > 0``), and the Rayleigh-tail noise scale.

    Parameters
    ----------
    spec : array-like, shape (N,)
        Real pixel-space spectrum.
    sampling_kms : float
        Velocity spacing per pixel (km/s).
    normalize_to_snr : bool, optional
        If True, divide ``log|FT|`` by its per-frequency uncertainty.
    tail_fraction : float, optional
        Fraction of high-ζ points used for the Rayleigh noise floor.

    Returns
    -------
    amp, log_amp, zeta, sigma_A : ndarray, ndarray, ndarray, float
    """
    spec = np.asarray(spec, dtype=float)
    if spec.ndim != 1:
        raise NotImplementedError(
            "specbeat is single-epoch: pass spectra one at a time "
            f"(got shape {spec.shape})")
    ft = rfft(spec)
    amp = np.abs(ft)
    zeta = rfftfreq(len(spec), d=sampling_kms)

    # Drop DC (zeta == 0): |FT[0]| = |sum(spec)| tracks the continuum mean, not
    # the line shape, and log|sum(spec)| is meaningless here (positive freqs
    # only — the project Fourier convention).
    amp = amp[1:]
    zeta = zeta[1:]

    n_tail = max(int(len(amp) * tail_fraction), 10)
    sigma_A = np.median(amp[-n_tail:]) / RAYLEIGH_MEDIAN_TO_SIGMA

    # A noiseless synthetic spectrum can have exactly-zero Fourier amplitudes;
    # log(0) = -inf is a legitimate value there, so silence the warning.
    with np.errstate(divide="ignore", invalid="ignore"):
        log_amp_raw = np.log(amp)

        if normalize_to_snr:
            # sigma_log = sigma_A / amp  =>  log_amp / sigma_log = log_amp * amp / sigma_A
            log_amp = log_amp_raw * amp / sigma_A
        else:
            log_amp = log_amp_raw

    return amp, log_amp, zeta, sigma_A


def truncate_to_snr_cutoff(zeta, log_amp, amp, amp_SNR, threshold=None,
                           zeta_max=None, exclude_low_zeta=0):
    """Keep the contiguous low-ζ region passing the SNR and/or ζ_max cut.

    Scans from low to high ζ and truncates at the first index that either falls
    below ``threshold`` (SNR) or exceeds ``zeta_max``. Removing the low-SNR
    high-ζ tail suppresses the log-of-noisy-amplitude (Rice) upward bias.

    Parameters
    ----------
    zeta, log_amp, amp, amp_SNR : array-like
        Fourier data (same length).
    threshold : float or None
        Minimum amplitude SNR to keep.
    zeta_max : float or None
        Maximum ζ (s/km) to keep.
    exclude_low_zeta : int, optional
        Extra near-DC indices to skip (DC already dropped upstream).

    Returns
    -------
    zeta_fit, log_amp_fit, amp_fit : ndarray, ndarray, ndarray
    """
    if threshold is None and zeta_max is None:
        raise ValueError("Must provide at least one of threshold or zeta_max")

    start_idx = 0 if exclude_low_zeta is None else int(exclude_low_zeta)
    n = len(zeta)

    if start_idx >= n:
        raise RuntimeError("exclude_low_zeta excludes all Fourier points")

    first_bad = n

    if threshold is not None:
        snr_below = amp_SNR[start_idx:] <= threshold
        if snr_below[0]:
            raise RuntimeError(
                "First included Fourier point is below SNR threshold"
            )
        if np.any(snr_below):
            first_bad = min(first_bad, start_idx + int(np.argmax(snr_below)))

    if zeta_max is not None:
        zeta_above = zeta[start_idx:] > zeta_max
        if zeta_above[0]:
            raise RuntimeError(
                "First included Fourier point exceeds zeta_max"
            )
        if np.any(zeta_above):
            first_bad = min(first_bad, start_idx + int(np.argmax(zeta_above)))

    return (
        zeta[start_idx:first_bad],
        log_amp[start_idx:first_bad],
        amp[start_idx:first_bad],
    )


def construct_log_amp_model_matrix(zeta, trial_v, n_cos_terms=0, n_baseline=2):
    """Design matrix for the log-amplitude model (paper eq. 12 columns).

    Columns: ``[1, |ζ|, ζ², ζ⁴, …, ζ^n_baseline]`` then, if ``n_cos_terms`` is
    not None, cosine columns ``ζ^p cos(2π ζ trial_v)`` for ``p = 0..n_cos_terms``.
    The default ``n_cos_terms = 0`` gives a single companion amplitude.

    Parameters
    ----------
    zeta : array-like
        Fourier conjugate axis (s/km).
    trial_v : float
        Trial companion velocity separation (km/s).
    n_cos_terms : int or None, optional
        Cosine polynomial order; None → baseline only. Default 0.
    n_baseline : int, optional
        Baseline polynomial order (>= 2). Default 2 → ``[1, |ζ|, ζ²]``.

    Returns
    -------
    ndarray, shape (len(zeta), p)
    """
    # Baseline [1, |ζ|, ζ²] always present; additional terms are strictly even
    # powers ζ⁴, ζ⁶, ..., ζ^n_baseline (n_baseline must be even when > 2).
    abs_zeta = np.abs(zeta)
    columns = [np.ones_like(zeta), abs_zeta, zeta ** 2]
    for p in range(4, n_baseline + 1, 2):
        columns.append(zeta ** p)

    if n_cos_terms is not None:
        cos_term = np.cos(2 * np.pi * zeta * trial_v)
        for power in range(n_cos_terms + 1):
            columns.append(zeta ** power * cos_term)

    return np.vstack(columns).T


def make_baseline_bounds(n_baseline, n_cos_terms=None, signs=(0, -1, -1)):
    """Per-column box bounds for :func:`construct_log_amp_model_matrix`.

    ``signs[i]``: 0 → unbounded, -1 → ``θ_i ≤ 0``, +1 → ``θ_i ≥ 0``. Default
    ``(0, -1, -1)`` bounds only the baseline ``a1, a2`` (a physically decaying
    Fourier amplitude); the companion cosine columns stay free.

    The default covers the three baseline columns of ``n_baseline <= 3``. For
    ``n_baseline > 3`` the extra even-order columns are left unbounded unless
    ``signs`` is given explicitly, one code per baseline column.

    Parameters
    ----------
    n_baseline : int
        Baseline polynomial order.
    n_cos_terms : int or None
        Cosine polynomial order, or None.
    signs : tuple of int
        Sign codes applied left-to-right.

    Returns
    -------
    lb, ub : ndarray, ndarray
    """
    n_baseline_cols = 3 + max(0, (n_baseline - 2) // 2)
    n_cos_cols = 0 if n_cos_terms is None else int(n_cos_terms) + 1
    p = n_baseline_cols + n_cos_cols

    lb = np.full(p, -np.inf)
    ub = np.full(p, np.inf)
    for i, s in enumerate(signs):
        if i >= p:
            break
        if s == -1:
            ub[i] = 0.0
        elif s == 1:
            lb[i] = 0.0
    return lb, ub


def weighted_least_squares(M, y, weights):
    """Unconstrained weighted least squares ``M θ ≈ y``; returns ``θ`` only.

    Parameters
    ----------
    M : ndarray, shape (n, p)
        Design matrix.
    y : ndarray, shape (n,)
        Data vector.
    weights : ndarray, shape (n,)
        Diagonal inverse-variance weights (ones for SNR-normalized data).

    Returns
    -------
    ndarray, shape (p,)
        Best-fit parameters.
    """
    theta, _, _ = bounded_weighted_least_squares(
        M, y, weights, bounds=None, return_active=False,
    )
    return theta


def bounded_weighted_least_squares(M, y, weights, bounds=None, lambda_ridge=0.0,
                                   ridge_diag=None, return_active=False):
    """Box-constrained weighted least squares with optional Tikhonov ridge.

    Solves ``min || sqrt(W)(M θ − y) ||² + λ || R θ ||²`` subject to
    ``lb ≤ θ ≤ ub``.

    Parameters
    ----------
    M : ndarray, shape (n, p)
        Design matrix.
    y : ndarray, shape (n,)
        Data vector.
    weights : ndarray, shape (n,)
        Diagonal inverse-variance weights.
    bounds : (lb, ub) or None
        Per-parameter bounds (``np.inf`` allowed); None → unconstrained.
    lambda_ridge : float
        Tikhonov ridge strength.
    ridge_diag : ndarray or None
        Diagonal of ``R``; None → identity.
    return_active : bool
        If True, also return the boolean active-constraint mask.

    Returns
    -------
    theta, cov, chi2[, active_mask]
        ``active_mask`` (bool, shape (p,)) only when ``return_active``.
    """
    M = np.asarray(M, dtype=float)
    y = np.asarray(y, dtype=float)
    weights = np.asarray(weights, dtype=float)
    n, p = M.shape

    if ridge_diag is None:
        diag_R = np.ones(p)
    else:
        diag_R = np.asarray(ridge_diag, dtype=float)

    sqrt_w = np.sqrt(weights)

    if bounds is None:
        # Unconstrained: closed-form normal equations (MᵀW M + λ R²) θ = MᵀW y.
        MTW = M.T * weights
        A = MTW @ M
        if lambda_ridge > 0.0:
            A = A + float(lambda_ridge) * np.diag(diag_R ** 2)
        rhs = MTW @ y
        theta = np.linalg.solve(A, rhs)
        try:
            cov = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            cov = np.full((p, p), np.nan)
        residuals = M @ theta - y
        chi2 = float(np.sum(weights * residuals * residuals))
        if return_active:
            return theta, cov, chi2, np.zeros(p, dtype=bool)
        return theta, cov, chi2

    from scipy.optimize import lsq_linear

    lb, ub = bounds
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)
    # Right-pad shorter bound vectors with ±inf so a baseline-only bounds vector
    # can be reused on a wider design (e.g. one that appends cosine columns).
    if lb.size > p or ub.size > p:
        raise ValueError(
            f"bounds longer ({lb.size},{ub.size}) than design columns ({p})"
        )
    if lb.size < p:
        lb = np.concatenate([lb, np.full(p - lb.size, -np.inf)])
    if ub.size < p:
        ub = np.concatenate([ub, np.full(p - ub.size, +np.inf)])

    # lsq_linear solves ordinary LS; fold the weights in by whitening the rows.
    Mw = M * sqrt_w[:, None]
    yw = y * sqrt_w
    if lambda_ridge > 0.0:
        # Impose λ‖Rθ‖² by appending √λ·R as extra rows with target 0.
        sqrt_lam = np.sqrt(float(lambda_ridge)) * diag_R
        M_aug = np.vstack([Mw, np.diag(sqrt_lam)])
        y_aug = np.concatenate([yw, np.zeros(p)])
    else:
        M_aug = Mw
        y_aug = yw

    res = lsq_linear(M_aug, y_aug, bounds=(lb, ub), method="bvls")
    theta = np.asarray(res.x, dtype=float)

    # BVLS active-set indicator (-1 lb, 0 free, +1 ub); tolerance-based detection
    # over-counts near-zero free coeffs when the bound is also at zero.
    active_mask = np.asarray(res.active_mask, dtype=int) != 0

    MTW = M.T * weights
    A = MTW @ M
    if lambda_ridge > 0.0:
        A = A + float(lambda_ridge) * np.diag(diag_R ** 2)

    # A pinned (active) parameter carries no fitted covariance: zero its row/col
    # and invert the Hessian only on the free sub-block (active-set covariance).
    cov = np.zeros((p, p))
    free_idx = np.where(~active_mask)[0]
    if free_idx.size > 0:
        A_free = A[np.ix_(free_idx, free_idx)]
        try:
            cov_free = np.linalg.inv(A_free)
        except np.linalg.LinAlgError:
            cov_free = np.full((free_idx.size, free_idx.size), np.nan)
        cov[np.ix_(free_idx, free_idx)] = cov_free

    residuals = M @ theta - y
    chi2 = float(np.sum(weights * residuals * residuals))

    if return_active:
        return theta, cov, chi2, active_mask
    return theta, cov, chi2


def amp_noise_sigma(sigma_pix, n=None, window=None):
    """Fourier-amplitude noise ``σ_A`` from per-pixel noise ``σ_pix`` (Parseval).

    For white noise on a length-``N`` real spectrum each rFFT component has std
    ``σ_A = σ_pix √(Σ w²/2)`` (``w`` the apodization window; unwindowed → √(N/2)).

    Parameters
    ----------
    sigma_pix : float or ndarray
        Per-pixel noise std.
    n : int, optional
        Spectrum length (unwindowed factor √(N/2)). Give exactly one of ``n``/``window``.
    window : ndarray, optional
        Apodization window (factor √(Σ window²/2)).

    Returns
    -------
    float or ndarray
        ``σ_A``.
    """
    if (n is None) == (window is None):
        raise ValueError("amp_noise_sigma: give exactly one of n=... or window=...")
    if window is not None:
        factor = np.sqrt(np.sum(np.asarray(window, dtype=float) ** 2) / 2.0)
    else:
        factor = np.sqrt(n / 2.0)
    return sigma_pix * factor
