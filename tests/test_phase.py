"""Tests for the phase channel (``specbeat.phase``), paper eq. 14.

The observable is ``z = arg(-S̃·e^{+2πiζ·rest_shift}) ≈ -α sin(2πζΔv)``;
the design column is ``-sin(2πζΔv)·ζ^p`` so the ``p=0`` coefficient is ``+α``.
"""

from __future__ import annotations

import numpy as np
import pytest

from specbeat import phase
from specbeat.simulate import generate_simulated_spectrum, generate_wavelength_grid


WL0, DV, NPIX = 5875.6, 5.0, 128
VG, GAMMA, EPS, DEPTH = 20.0, 5.0, 0.5, 0.5
X0_VEL = (NPIX // 2) * DV


def _spectrum(alpha, dv, sigma=0.0, seed=0, second=True):
    wave = generate_wavelength_grid(WL0, DV, NPIX)
    spec, _, _ = generate_simulated_spectrum(
        wave, DV, wavelength_0=WL0, line_depth=DEPTH, v=0.0, vsini=0,
        epsilon=EPS, vG=VG, gamma=GAMMA, second_component=second,
        v2=dv, vG2=VG, gamma2=GAMMA, vsini2=0, depth_ratio=alpha,
    )
    spec = np.asarray(spec, float)
    if sigma > 0:
        spec = spec + np.random.default_rng(seed).normal(0, sigma, spec.shape)
    return spec


def _fft(spec):
    zeta = np.fft.rfftfreq(NPIX, d=DV)[1:]
    return np.fft.rfft(spec)[1:], zeta


def _fake_fft_with_phase(zeta, alpha, dv, amp=None):
    if amp is None:
        amp = np.ones_like(zeta)
    S0 = amp * np.exp(-1j * alpha * np.sin(2 * np.pi * zeta * dv))
    return -S0


# --- phase_observable --------------------------------------------------------
def test_observable_single_star_phase_near_zero():
    spec = _spectrum(0.0, 22.0, sigma=0.0, second=False)
    fft, zeta = _fft(spec)
    z, w = phase.phase_observable(fft, zeta, X0_VEL, sigma_ft=1.0)
    lo = zeta < 0.02
    assert np.all(np.abs(z[lo]) < 1e-3)
    assert np.all(np.isfinite(w)) and np.all(w >= 0)


def test_observable_weights_inverse_variance():
    spec = _spectrum(0.25, 22.0)
    fft, zeta = _fft(spec)
    _, w1 = phase.phase_observable(fft, zeta, X0_VEL, sigma_ft=1.0)
    _, w2 = phase.phase_observable(fft, zeta, X0_VEL, sigma_ft=2.0)
    assert np.allclose(w1, 4.0 * w2)


# --- phase_template ----------------------------------------------------------
def test_template_shape_and_columns():
    zeta = np.fft.rfftfreq(NPIX, d=DV)[1:]
    dv = 22.0
    M = phase.phase_template(zeta, dv, n_sin_terms=1)
    assert M.shape == (len(zeta), 2)
    s = -np.sin(2 * np.pi * zeta * dv)
    assert np.allclose(M[:, 0], s)
    assert np.allclose(M[:, 1], s * zeta)


def test_template_signed_dv_flips():
    zeta = np.fft.rfftfreq(NPIX, d=DV)[1:]
    Mp = phase.phase_template(zeta, +22.0, n_sin_terms=0)
    Mm = phase.phase_template(zeta, -22.0, n_sin_terms=0)
    assert np.allclose(Mp, -Mm)


# --- fit_phase_alpha exact algebra -------------------------------------------
def test_exact_alpha_recovery_on_fabricated_phase():
    zeta = np.fft.rfftfreq(NPIX, d=DV)[1:]
    alpha, dv = 0.3, 22.0
    fft = _fake_fft_with_phase(zeta, alpha, dv)
    out = phase.fit_phase_alpha(fft, zeta, rest_shift_kms=0.0, dv_rel=dv,
                                sigma_ft=1.0, n_sin_terms=0)
    assert out["alpha_hat"] == pytest.approx(alpha, rel=1e-9)
    assert out["df"] == 1


def test_T_equals_alphahat2_times_Stt():
    zeta = np.fft.rfftfreq(NPIX, d=DV)[1:]
    fft = _fake_fft_with_phase(zeta, 0.2, 17.0)
    out = phase.fit_phase_alpha(fft, zeta, rest_shift_kms=0.0, dv_rel=17.0,
                                sigma_ft=1.3, n_sin_terms=0)
    z, w = phase.phase_observable(fft, zeta, 0.0, sigma_ft=1.3)
    t = phase.phase_template(zeta, 17.0, n_sin_terms=0)[:, 0]
    Stt = np.sum(w * t * t)
    assert out["T"] == pytest.approx(out["alpha_hat"] ** 2 * Stt, rel=1e-9)


def test_sigma_alpha_matches_fisher():
    zeta = np.fft.rfftfreq(NPIX, d=DV)[1:]
    fft = _fake_fft_with_phase(zeta, 0.25, 22.0)
    out = phase.fit_phase_alpha(fft, zeta, rest_shift_kms=0.0, dv_rel=22.0,
                                sigma_ft=0.7, n_sin_terms=0)
    z, w = phase.phase_observable(fft, zeta, 0.0, sigma_ft=0.7)
    t = phase.phase_template(zeta, 22.0, n_sin_terms=0)[:, 0]
    assert out["sigma_alpha"] == pytest.approx(1.0 / np.sqrt(np.sum(w * t * t)),
                                               rel=1e-9)


@pytest.mark.parametrize("n_sin", [0, 1, 2])
def test_df_bookkeeping(n_sin):
    zeta = np.fft.rfftfreq(NPIX, d=DV)[1:]
    fft = _fake_fft_with_phase(zeta, 0.2, 22.0)
    out = phase.fit_phase_alpha(fft, zeta, rest_shift_kms=0.0, dv_rel=22.0,
                                sigma_ft=1.0, n_sin_terms=n_sin)
    assert out["df"] == n_sin + 1
    assert np.asarray(out["theta"]).shape == (n_sin + 1,)


# --- end-to-end recovery on synthetic spectra --------------------------------
def test_alpha_recovery_noiseless_spectrum():
    spec = _spectrum(0.25, 22.0, sigma=0.0)
    fft, zeta = _fft(spec)
    out = phase.fit_phase_alpha(fft, zeta, X0_VEL, 22.0, sigma_ft=1.0, n_sin_terms=0)
    assert out["alpha_hat"] == pytest.approx(0.25, abs=0.05)
    assert out["alpha_hat"] > 0.15
    assert out["T"] > 0


def test_negative_dv_recovers_positive_alpha():
    spec = _spectrum(0.25, -22.0, sigma=0.0)
    fft, zeta = _fft(spec)
    out = phase.fit_phase_alpha(fft, zeta, X0_VEL, dv_rel=-22.0, sigma_ft=1.0,
                                n_sin_terms=0)
    assert out["alpha_hat"] == pytest.approx(0.25, abs=0.05)
    assert out["alpha_hat"] > 0.15
    out_wrong = phase.fit_phase_alpha(fft, zeta, X0_VEL, dv_rel=+22.0, sigma_ft=1.0,
                                      n_sin_terms=0)
    assert out_wrong["alpha_hat"] < -0.15


# --- multi-epoch seam (v1 raises for K > 1) ----------------------------------
def test_multi_epoch_not_implemented_v1():
    """v1 is single-epoch: a 2-D (K>1) fft stack must raise NotImplementedError."""
    zeta = np.fft.rfftfreq(NPIX, d=DV)[1:]
    fftA = _fake_fft_with_phase(zeta, 0.2, +22.0)
    fftB = _fake_fft_with_phase(zeta, 0.2, -22.0)
    with pytest.raises(NotImplementedError):
        phase.fit_phase_alpha(np.vstack([fftA, fftB]), zeta,
                              rest_shift_kms=0.0, dv_rel=np.array([+22.0, -22.0]),
                              sigma_ft=1.0, n_sin_terms=0)
