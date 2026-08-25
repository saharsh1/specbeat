"""Fourier-space broadening kernels (plain floats; no astropy).

Conventions: velocities in km/s, ζ in s/km, kernel-width parameters
(``vG``, ``gamma``) are **FWHM** in km/s. All kernels are normalized so that
``K(ζ=0) = 1`` (amplitude convention), i.e. the transform of a *unit-area*
velocity-space kernel.

Arguments are plain floats throughout: km/s for velocities, s/km for zeta.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "rotational_kernel",
    "k",
    "gaussian_broadening_FT",
    "lorentzian_broadening_FT",
    "velocity_shift",
    "logK_approx",
    "get_kernel",
]


def rotational_kernel(zeta, vsini, epsilon, return_log=True):
    """Rotational broadening kernel in Fourier space (Gray 1976).

    The single definition of the rotational kernel in the package; everything
    else refers to this one.

    Parameters
    ----------
    zeta : array-like
        Fourier conjugate variable (s/km).
    vsini : float
        Projected rotational velocity (km/s).
    epsilon : float
        Linear limb-darkening coefficient (0 = uniform disk).
    return_log : bool, optional
        If True (default) return ``log|kernel|``; else ``|kernel|``.

    Returns
    -------
    ndarray
        Kernel log-amplitude or amplitude, normalized to 1 at ζ = 0.
    """
    from scipy.special import j1

    zeta = np.asarray(zeta, dtype=float)
    x = 2.0 * np.pi * zeta * vsini
    denom = 3.0 - epsilon
    out = np.empty_like(x, dtype=float)

    small = np.isclose(x, 0.0, atol=1e-12)
    big = ~small

    out[small] = (3.0 * (1.0 - epsilon) + 2.0 * epsilon) / denom

    if np.any(big):
        xb = x[big]
        t1 = (6.0 * (1.0 - epsilon) / denom) * (j1(xb) / xb)
        t2 = (6.0 * epsilon / denom) * (
            (np.sin(xb) - xb * np.cos(xb)) / xb ** 3
        )
        out[big] = t1 + t2

    if return_log:
        return np.log(np.abs(out))
    return out


# Short alias, matching the notation in the paper.
k = rotational_kernel


def gaussian_broadening_FT(zeta, vG=3.0):
    """FT of a unit-area Gaussian (instrumental/thermal) kernel, ``G(0) = 1``.

    ``G(ζ) = exp(-(π ζ)² vG² / (4 ln 2))`` with ``vG`` the FWHM in km/s.

    Parameters
    ----------
    zeta : array-like
        Fourier conjugate axis (s/km).
    vG : float
        Gaussian FWHM (km/s).

    Returns
    -------
    ndarray
        Real FT, normalized so ``G(0) = 1``.
    """
    zeta = np.asarray(zeta, dtype=float)
    return np.exp(-(np.pi * zeta) ** 2 * vG ** 2 / (4.0 * np.log(2.0)))


def lorentzian_broadening_FT(zeta, gamma=3.0):
    """FT of a unit-area Lorentzian (natural/pressure) kernel, ``G(0) = 1``.

    ``G(ζ) = exp(-π γ |ζ|)`` with ``γ`` the FWHM in km/s.

    Parameters
    ----------
    zeta : array-like
        Fourier conjugate axis (s/km).
    gamma : float
        Lorentzian FWHM (km/s).

    Returns
    -------
    ndarray
        Real FT, normalized so ``G(0) = 1``.
    """
    zeta = np.asarray(zeta, dtype=float)
    return np.exp(-np.pi * gamma * np.abs(zeta))


def velocity_shift(zeta, v_shift):
    """Complex FT phase factor for a velocity shift ``v_shift`` (km/s).

    ``exp(-2πi ζ v_shift)``.

    Parameters
    ----------
    zeta : array-like
        Fourier conjugate axis (s/km).
    v_shift : float
        Velocity shift (km/s).

    Returns
    -------
    ndarray, complex
        The velocity-shift phase factor.
    """
    zeta = np.asarray(zeta, dtype=float)
    return np.exp(-2j * np.pi * zeta * v_shift)


def logK_approx(zeta, vG, vsini, epsilon, vgamma, more_terms=False):
    """Taylor-series approximation of ``ln|K|`` for the combined kernel.

    ``ln K_G = -π² vG² ζ² / (4 ln 2)``, ``ln K_L = -π vgamma |ζ|``, plus the
    rotational ζ² (and optional ζ⁴, ζ⁶) corrections. Widths are FWHM (km/s).

    Parameters
    ----------
    zeta : array-like
        Fourier conjugate axis (s/km).
    vG : float
        Gaussian FWHM (km/s).
    vsini : float
        Projected rotational velocity (km/s).
    epsilon : float
        Linear limb-darkening coefficient.
    vgamma : float
        Lorentzian FWHM (km/s).
    more_terms : bool, optional
        If True, include 4th- and 6th-order rotational corrections.

    Returns
    -------
    ndarray
        Approximate combined log-amplitude.
    """
    z = np.abs(np.asarray(zeta, dtype=float))
    pi = np.pi

    t1 = -pi * vgamma * z
    t2 = -(pi ** 2) * (vG ** 2 / (4.0 * np.log(2.0))
                       - (epsilon / 15.0 - 0.5) * vsini ** 2) * z ** 2

    if more_terms:
        t4 = (pi ** 4) * (epsilon / 63.0 - 1.0 / 24.0) * vsini ** 4 * z ** 4
        t6 = (pi ** 6) * (19.0 * epsilon / 4536.0 - 1.0 / 144.0) * vsini ** 6 * z ** 6
    else:
        t4, t6 = 0.0, 0.0

    return t1 + t2 + t4 + t6


def get_kernel(zeta=None, dv=0.1, N=1000, v=0.0, vG=3.0, vsini=15.0,
               epsilon=0.5, gamma=10.0):
    """Combined FT kernel: velocity shift × rotational × Gaussian × Lorentzian.

    L2-normalized (unit norm on the returned ζ grid), which is the convention
    :func:`specbeat.simulate.generate_simulated_spectrum` expects. Note that this
    differs by a constant from the ``K(0) = 1`` convention of the individual
    kernels above.

    Parameters
    ----------
    zeta : array-like or None
        Fourier grid (s/km); if None, built from ``N`` and ``dv``.
    dv : float
        Velocity step (km/s).
    N : int
        Grid length.
    v : float
        Velocity shift (km/s).
    vG, vsini, epsilon, gamma : float
        Kernel parameters (FWHM km/s where applicable).

    Returns
    -------
    ndarray, complex
        The L2-normalized combined kernel.
    """
    if zeta is None:
        zeta = np.fft.rfftfreq(int(N), d=dv)
    Kinst = gaussian_broadening_FT(zeta, vG=vG)
    Krot = rotational_kernel(zeta, vsini, epsilon, return_log=False)
    Kshift = velocity_shift(zeta, v)
    Klor = lorentzian_broadening_FT(zeta, gamma=gamma)
    kernel = Kshift * Krot * Kinst * Klor

    norm = np.sqrt(np.vdot(kernel, kernel).real)
    kernel = kernel / norm
    return kernel
