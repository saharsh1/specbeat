"""Integration tests spanning several specbeat modules.

D1. ``detect.detection_statistic`` end-to-end on a seeded synthetic composite
    recovers the injected ``α`` within the Cramér–Rao bound and clears the
    detection threshold.
D2. ``simulate.binary_log_amplitude`` (paper eq. 12) matches ``log|FT|`` of the
    two-component ``generate_simulated_spectrum`` at low ζ.
"""

from __future__ import annotations

import numpy as np
import pytest

from specbeat import detect, nulls
from specbeat.simulate import (
    C_KMS,
    binary_log_amplitude,
    generate_simulated_spectrum,
    generate_wavelength_grid,
)


# --- D1: end-to-end detection recovers alpha within the CRB ------------------
def test_detection_statistic_recovers_injected_alpha():
    dv = 1.0
    N = 512
    alpha_true = 0.35
    dv_true = 70.0
    wave = generate_wavelength_grid(5875.6, dv, N)
    spec, _, _ = generate_simulated_spectrum(
        wave, dv, wavelength_0=5875.6, line_depth=0.6, v=0.0, vsini=10.0,
        vG=10.0, gamma=5.0, epsilon=0.5, second_component=True,
        v2=dv_true, vsini2=10.0, vG2=10.0, gamma2=5.0, depth_ratio=alpha_true)
    spec = np.asarray(spec, float)
    rng = np.random.default_rng(0)
    spec = spec + rng.normal(0.0, 2e-3, N)

    v_grid = np.arange(0.0, 150.0, 5.0)
    out = detect.detection_statistic(spec, dv, v_grid, zeta_max=0.03)

    # Detected above the 5% one-sided threshold.
    assert out["T"] > nulls.detection_threshold(fap=0.05, df=1)
    # alpha recovered within a few CRB sigma (allow O(alpha^2)/(R-1) model bias).
    assert abs(out["alpha_hat"] - alpha_true) < 5.0 * out["sigma_alpha"] + 0.05
    # Best velocity near the injection.
    assert abs(out["v_best"] - dv_true) <= 5.0


# --- D2: eq.12 model vs full-sim FFT at low zeta -----------------------------
def test_binary_log_amplitude_matches_sim_fft():
    dv = 1.0
    N = 2048
    alpha = 0.05
    dv_true = 60.0
    vG, gamma, vsini, eps = 15.0, 5.0, 0.0, 0.5
    wave = generate_wavelength_grid(5875.6, dv, N)
    spec, _, _ = generate_simulated_spectrum(
        wave, dv, wavelength_0=5875.6, line_depth=0.5, v=0.0, vsini=vsini,
        vG=vG, gamma=gamma, epsilon=eps, second_component=True,
        v2=dv_true, vsini2=vsini, vG2=vG, gamma2=gamma, depth_ratio=alpha)
    spec = np.asarray(spec, float)

    zeta = np.fft.rfftfreq(N, d=dv)[1:]
    band = zeta <= 0.03
    # Slice to the band before taking the log: past it the amplitude underflows
    # to zero and log() would warn about a value that is never used.
    sim_log = np.log(np.abs(np.fft.rfft(spec))[1:][band])
    model_log = binary_log_amplitude(zeta[band], alpha, dv_true, vG=vG,
                                      gamma=gamma, vsini=vsini, epsilon=eps)

    # Compare shapes after removing a constant offset (DC normalization differs);
    # the residual RMS at low ζ is small for a faint companion.
    diff = (sim_log - model_log)
    diff = diff - diff.mean()
    assert np.sqrt(np.mean(diff ** 2)) < 0.02


def test_c_kms_value():
    # simulate.C_KMS replaces astropy's const.c.to('km/s').value.
    assert C_KMS == pytest.approx(299792.458, abs=1e-3)
