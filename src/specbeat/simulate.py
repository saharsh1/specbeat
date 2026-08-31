"""Synthetic single-/double-lined spectrum generator (plain floats; no astropy).

The speed of light is the module constant :data:`C_KMS` (km/s).

Circular-wrap guard
-------------------
The spectra live on a finite, periodic (FFT) segment. A line placed within
``vsini + |v| + few*vG`` of a segment edge wraps around and contaminates the
opposite edge. :func:`generate_simulated_spectrum` emits a ``RuntimeWarning``
when a component's broadened extent falls within that margin of the grid edge
(it does not raise — near-edge use is legitimate but must be flagged).
"""

from __future__ import annotations

import warnings

import numpy as np

from . import kernels as bk

__all__ = [
    "C_KMS",
    "generate_wavelength_grid",
    "generate_simulated_spectrum",
    "kernel_amp",
    "binary_log_amplitude",
    "add_noise",
]

#: Speed of light in km/s.
C_KMS = 299792.458


def generate_wavelength_grid(wavelength_0=5875.6, dv=0.4, N=2000):
    """Uniform log-λ (constant velocity step) wavelength grid centred on λ0.

    Parameters
    ----------
    wavelength_0 : float
        Central wavelength (Angstrom).
    dv : float
        Velocity step per pixel (km/s).
    N : int
        Number of pixels.

    Returns
    -------
    ndarray
        Wavelength grid (Angstrom), length ``N``.
    """
    loglam0 = np.log(wavelength_0)
    dloglam = dv / C_KMS
    if N % 2 == 0:
        idx = np.arange(-N // 2, N // 2)
    else:
        idx = np.arange(-(N // 2), N // 2 + 1)
    loglam_grid = loglam0 + idx * dloglam
    return np.exp(loglam_grid)


def generate_simulated_spectrum(
    wavelength,
    dv=0.1,
    N=None,
    *,
    wavelength_0=5875.6,
    line_depth=0.7,
    v=-30.0,
    vG=3.0,
    gamma=3.0,
    vsini=20.0,
    epsilon=0.5,
    second_component=False,
    v2=30.0,
    vG2=3.0,
    gamma2=3.0,
    vsini2=40.0,
    depth_ratio=0.5,
    epsilon2=None,
    alpha=None,
):
    """Simulate a (possibly double-lined) spectrum on a log-λ grid.

    LSF-limited paper convention: the primary line rides on a flat continuum; a
    faint companion (``second_component=True``) is added at velocity ``v2`` with
    line-depth ratio ``depth_ratio`` (aliased by ``alpha`` for the paper-eq.-12
    amplitude ``α``; if both are given, ``alpha`` wins).

    Parameters
    ----------
    wavelength : array-like
        Wavelength grid (Angstrom).
    dv : float
        Velocity step per pixel (km/s).
    N : int or None
        Number of points (default ``len(wavelength)``).
    wavelength_0 : float
        Central wavelength of the line (Angstrom).
    line_depth : float
        Primary line depth.
    v : float
        Primary velocity shift (km/s).
    vG, gamma, vsini, epsilon : float
        Primary kernel parameters (FWHM km/s where applicable).
    second_component : bool
        If True, add a companion line.
    v2, vG2, gamma2, vsini2, depth_ratio, epsilon2 : float
        Companion parameters. ``depth_ratio`` is the companion/primary line-depth
        ratio (= ``α``).
    alpha : float or None
        Alias for ``depth_ratio`` (paper eq. 12 amplitude). Overrides
        ``depth_ratio`` when not None.

    Returns
    -------
    spectrum : ndarray
        Simulated spectrum (sum of components; continuum-subtracted line residual).
    broadened_line_profile : ndarray
        Primary broadened profile.
    broadened_line_profile2 : ndarray or None
        Companion broadened profile (if present).

    Warns
    -----
    RuntimeWarning
        When a component's broadened extent falls within
        ``vsini + |v| + few*vG`` of a grid edge (circular-wrap contamination).
    """
    if N is None:
        N = len(wavelength)

    # Circular-wrap guard: the grid is periodic, centred on wavelength_0, so its
    # nearest edge sits half_window = (N/2)*dv km/s from a rest-frame line. A
    # component whose broadened reach (|v| + vsini + WRAP_VG_MARGIN*vG) crosses
    # that edge wraps around and contaminates the far edge. Warn (do not raise):
    # near-edge use is legitimate but must be flagged (trap 12).
    half_window = (N / 2.0) * dv
    WRAP_VG_MARGIN = 3.0

    def _wrap_reach(v_c, vsini_c, vG_c):
        return abs(v_c) + abs(vsini_c) + WRAP_VG_MARGIN * abs(vG_c)

    reaches = [_wrap_reach(v, vsini, vG)]
    if second_component:
        reaches.append(_wrap_reach(v2, vsini2, vG2))
    if max(reaches) >= half_window:
        warnings.warn(
            "Simulated line's broadened extent reaches within "
            f"{WRAP_VG_MARGIN}*vG of the segment edge (half-window "
            f"{half_window:.1f} km/s): circular-wrap contamination is likely.",
            RuntimeWarning,
            stacklevel=2,
        )

    if alpha is not None:
        depth_ratio = alpha

    zeta = np.fft.rfftfreq(N, d=dv)
    x_grid = np.arange(N)
    x0 = np.interp(wavelength_0, wavelength, x_grid) * dv
    line_profile_hat = np.exp(-2j * np.pi * zeta * x0)
    line_profile_hat /= np.linalg.norm(line_profile_hat)

    kernel_hat = bk.get_kernel(
        zeta=zeta, dv=dv, N=N, v=v, vG=vG, vsini=vsini, epsilon=epsilon,
        gamma=gamma,
    )

    broadened_line_profile = np.fft.irfft(
        line_profile_hat * kernel_hat, n=N, norm="ortho")
    broadened_line_profile *= -line_depth / np.max(broadened_line_profile)

    broadened_line_profile2 = None

    if second_component:
        line_depth2 = line_depth * depth_ratio
        epsilon2 = epsilon if epsilon2 is None else epsilon2

        line_profile2_hat = np.exp(-2j * np.pi * zeta * x0)
        line_profile2_hat /= np.linalg.norm(line_profile2_hat)

        kernel_hat2 = bk.get_kernel(
            zeta=zeta, dv=dv, N=N, v=v2, vG=vG2, vsini=vsini2, epsilon=epsilon2,
            gamma=gamma2,
        )

        broadened_line_profile2 = np.fft.irfft(
            line_profile2_hat * kernel_hat2, n=N, norm="ortho")
        broadened_line_profile2 *= -line_depth2 / np.max(broadened_line_profile2)

        spectrum = broadened_line_profile + broadened_line_profile2
    else:
        spectrum = broadened_line_profile

    return spectrum, broadened_line_profile, broadened_line_profile2


def kernel_amp(zeta, vG, gamma, vsini, epsilon, v=0.0):
    """Amplitude ``|K(ζ)|`` of the combined broadening kernel (``|K(0)| = 1``).

    Product of the Gaussian, Lorentzian, and rotational amplitudes (the velocity
    shift is pure phase and does not affect ``|K|``).

    Parameters
    ----------
    zeta : array-like
        Fourier conjugate axis (s/km).
    vG, gamma, vsini, epsilon : float
        Kernel parameters (FWHM km/s where applicable).
    v : float, optional
        Velocity shift (km/s); ignored by ``|K|`` but accepted for symmetry.

    Returns
    -------
    ndarray
        ``|K(ζ)|``, normalized to 1 at ζ = 0.
    """
    zeta = np.asarray(zeta, dtype=float)
    kg = bk.gaussian_broadening_FT(zeta, vG=vG)
    kl = bk.lorentzian_broadening_FT(zeta, gamma=gamma)
    kr = np.abs(bk.rotational_kernel(zeta, vsini, epsilon, return_log=False))
    return kg * kl * kr


def binary_log_amplitude(zeta, alpha, delta_v, *, vG=3.0, gamma=3.0, vsini=0.0,
                         epsilon=0.5):
    """LSF-limited log-|FT| of a single-star-plus-companion spectrum (paper eq. 12).

    In the ``R(ζ) = 1`` regime the composite log-amplitude is the single-star
    baseline plus a cosine beat::

        y(ζ) = ln|K(ζ)| + ln(1 + α cos(2π ζ Δv))
             ≈ a0 + a1 ζ + a2 ζ² + a3 cos(2π ζ a4)   (paper eq. 12)

    with ``a4 = Δv``. This is the model the amplitude channel fits.

    Parameters
    ----------
    zeta : array-like
        Fourier conjugate axis (s/km), DC dropped.
    alpha : float
        Companion line-depth ratio (paper eq. 12 amplitude).
    delta_v : float
        Companion velocity separation ``Δv`` (km/s).
    vG, gamma, vsini, epsilon : float
        Primary kernel parameters (FWHM km/s where applicable).

    Returns
    -------
    ndarray
        The composite log-amplitude on ``zeta``.
    """
    zeta = np.asarray(zeta, dtype=float)
    ln_K = np.log(kernel_amp(zeta, vG, gamma, vsini, epsilon))
    beat = np.log1p(alpha * np.cos(2.0 * np.pi * zeta * delta_v))
    return ln_K + beat


def add_noise(spec, sigma, rng=None, seed=None):
    """Add white Gaussian per-pixel noise of std ``sigma`` to a spectrum.

    Parameters
    ----------
    spec : array-like
        Input spectrum.
    sigma : float
        Per-pixel noise std.
    rng : numpy.random.Generator or None
        Explicit generator (takes precedence over ``seed``).
    seed : int or None
        Seed for a fresh generator when ``rng`` is None.

    Returns
    -------
    ndarray
        ``spec + N(0, sigma²)``.
    """
    spec = np.asarray(spec, dtype=float)
    if rng is None:
        rng = np.random.default_rng(seed)
    return spec + rng.normal(0.0, sigma, spec.shape)
