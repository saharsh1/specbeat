"""Phase-channel Fisher information: the Δv scaling of the α precision.

The phase observable is ``z ≈ -α sin(2π ζ Δv)`` with no baseline, so the Fisher
information on ``α`` is ``I_α = Σ_i w_i sin²(2π ζ_i Δv)``, ``w_i = |S|²/σ_ft²``,
and the CRB precision is ``σ_α = 1/√I_α``.

Two regimes, both exercised through :func:`specbeat.power.fisher_information_alpha`:

* **External rest frame** (``design=None``, no self-calibration): for small Δv,
  ``sin(2π ζ Δv) ≈ 2π ζ Δv`` so ``I_α ∝ Δv²`` and ``σ_α ∝ Δv⁻¹`` — slope −1 in
  log–log. This is the phase channel's clean ``Δα/α ∝ Δv⁻¹`` floor.
* **Self-calibrated** (project out a velocity-shift column ``∝ ζ``): the leading
  ``ζ Δv`` term is absorbed by the baseline, leaving ``O(ζ³ Δv³)``, so
  ``I_α ∝ Δv⁶`` and ``σ_α ∝ Δv⁻³`` — slope −3.

Weights are computed inline from a closed-form Gaussian |FT| (no package stubs),
so this test is a self-contained oracle for the scaling law.
"""

from __future__ import annotations

import numpy as np

from specbeat import power


def _setup():
    dv_pix = 1.0
    N = 2048
    zeta = np.fft.rfftfreq(N, d=dv_pix)[1:]
    zeta = zeta[zeta <= 0.02]                     # low-ζ band: asymptote holds
    vG = 20.0                                      # km/s FWHM
    S = np.exp(-(np.pi * zeta * vG) ** 2 / (4.0 * np.log(2.0)))  # Gaussian |FT|
    w = S ** 2                                      # |S|^2 / sigma^2 (sigma=1)
    return zeta, w


def _slope(dvs, sigmas):
    return np.polyfit(np.log(dvs), np.log(sigmas), 1)[0]


def test_external_rest_frame_slope_minus_one():
    zeta, w = _setup()
    dvs = np.geomspace(0.05, 0.5, 8)              # 2π ζmax dv <~ 0.06 << 1
    sig = []
    for dv in dvs:
        t = np.sin(2 * np.pi * zeta * dv)
        I = power.fisher_information_alpha(t, None, w)
        sig.append(1.0 / np.sqrt(I))
    slope = _slope(dvs, np.array(sig))
    np.testing.assert_allclose(slope, -1.0, atol=0.15)


def test_self_calibrated_degrades_to_slope_minus_three():
    zeta, w = _setup()
    dvs = np.geomspace(0.05, 0.5, 8)
    design = (-2.0 * np.pi * zeta)[:, None]        # velocity-shift column ∝ ζ
    sig = []
    for dv in dvs:
        t = np.sin(2 * np.pi * zeta * dv)
        I = power.fisher_information_alpha(t, design, w)
        sig.append(1.0 / np.sqrt(I))
    slope = _slope(dvs, np.array(sig))
    np.testing.assert_allclose(slope, -3.0, atol=0.3)
