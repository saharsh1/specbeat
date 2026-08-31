"""Detection power: non-centrality, constrained power, Fisher information.

Paper Sec. 3 and App. C.2 (eq. C.10). The alternative has non-centrality ``λ = α²·I_α``; the constrained
one-sided power is ``Φ̄(√c − √λ)`` with ``c`` from
:func:`specbeat.nulls.detection_threshold`.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from .nulls import detection_threshold

__all__ = ["noncentrality", "detection_power", "fisher_information_alpha"]


def noncentrality(alpha, I_alpha):
    """Non-centrality of the alternative, ``λ = α²·I_α``.

    Parameters
    ----------
    alpha : float or ndarray
        True companion amplitude(s).
    I_alpha : float
        Profiled Fisher information on ``α``.

    Returns
    -------
    float or ndarray
        Non-centrality ``λ``.
    """
    return np.asarray(alpha, float) ** 2 * I_alpha


def detection_power(alpha, I_alpha, fap=0.05, df=1):
    """Power of the one-sided constrained detection test.

    ``df = 1``: ``Φ̄(√c − √λ)``, ``λ = α²·I_α``, ``c = detection_threshold(fap, 1)``.
    → ``fap`` as ``α → 0`` (NOT ``2·fap`` — the bare ``ncx2`` tail is not used).
    ``df > 1`` has no simple closed form and raises ``NotImplementedError``.

    Parameters
    ----------
    alpha : float or ndarray
        True companion amplitude(s).
    I_alpha : float
        Profiled Fisher information on ``α``.
    fap : float
        False-alarm probability defining the threshold.
    df : int
        Number of companion parameters (only df=1 implemented).

    Returns
    -------
    float or ndarray
        Detection probability in [0, 1].
    """
    if df != 1:
        raise NotImplementedError(
            "constrained detection power has no simple closed form for df > 1"
        )
    c = detection_threshold(fap=fap, df=1)
    lam = noncentrality(alpha, I_alpha)
    power = norm.sf(np.sqrt(c) - np.sqrt(lam))
    return power if np.ndim(power) else float(power)


def fisher_information_alpha(template, design, weights):
    """Profiled Fisher information on ``α``: ``I_α = rᵀ W r``.

    ``r = (I − P) t`` is the part of the ripple template ``t`` orthogonal (in the
    ``W`` metric) to the baseline spanned by ``design``.

    ``design=None`` means **no baseline** (``r = t``, ``I_α = tᵀ W t``): the
    phase-channel case, which has no nuisance baseline. This is the seam the
    phase-Fisher test exercises (external rest frame vs. a projected-out shift
    column).

    Parameters
    ----------
    template : ndarray, shape (N,)
        Ripple template ``t_i``.
    design : ndarray, shape (N, k), or None
        Baseline design matrix, or None for no baseline.
    weights : ndarray, shape (N,)
        Inverse-variance weights ``w_i``.

    Returns
    -------
    float
        ``I_α``; the Cramér–Rao error on ``α`` is ``1/√I_α``.
    """
    t = np.asarray(template, float)
    w = np.asarray(weights, float)
    if design is None:
        return float((t * w) @ t)
    M = np.asarray(design, float)
    MtW = M.T * w
    coef = np.linalg.solve(MtW @ M, MtW @ t)
    r = t - M @ coef
    return float((r * w) @ r)
