"""Injection-recovery / completeness path of the MC module.

Covers ``run_injection_mc``, ``completeness``, and ``calibrate_null`` — the
power half of the module, and the path that must NOT threshold a velocity-scanned
``T`` against the fixed-v Chernoff value 2.706: the scanned null is
heavier-tailed, so 2.706 would be anti-conservative.
"""
from __future__ import annotations

import numpy as np
import pytest

from specbeat import mc, nulls


def _setup():
    # A modest-SNR primary keeps the H0 null at small T (median ~ a few); a very
    # strong static line instead inflates the null to hundreds, because its
    # un-absorbed structure is (correctly) flagged by the binary fit — the
    # paper's "static blend looks like a companion" effect (sec 3.2), which is a
    # regime study, not what this completeness unit test is checking.
    dv, n_pix = 2.0, 256
    template = mc.make_template(dv, n_pix, line_depth=0.20, vG=12.0, gamma=3.0)
    v_grid = np.linspace(20.0, 120.0, 11)
    return dict(dv=dv, n_pix=n_pix, template=template, v_grid=v_grid,
                sigma_pix=0.02)


def test_injection_no_null_returns_T_only():
    """Without a null sample, no detection decision is fabricated."""
    s = _setup()
    out = mc.run_injection_mc(
        alpha=0.2, delta_v=60.0, sigma_pix=s["sigma_pix"], n_pix=s["n_pix"],
        dv=s["dv"], v_grid=s["v_grid"], n_draws=20, seed=1,
        template=s["template"])
    assert "T" in out and out["T"].shape == (20,)
    assert "detected" not in out and "threshold" not in out


def test_injection_uses_calibrated_scanned_threshold_not_2706():
    """With a matching null, the threshold is the Gamma-null level, not 2.706."""
    s = _setup()
    null = mc.run_null_mc(
        s["sigma_pix"], s["n_pix"], s["dv"], s["v_grid"], n_draws=400,
        seed=7, template=s["template"])
    out = mc.run_injection_mc(
        alpha=0.25, delta_v=60.0, sigma_pix=s["sigma_pix"], n_pix=s["n_pix"],
        dv=s["dv"], v_grid=s["v_grid"], n_draws=50, seed=2,
        template=s["template"], null_T=null["T"], fap=0.05)
    assert "detected" in out and "threshold" in out
    fit = nulls.fit_gamma_null(null["T"])
    expected = nulls.gamma_null_threshold(0.05, fit["a"], fit["theta"], fit["w0"])
    assert out["threshold"] == pytest.approx(expected)
    # The scanned null is heavier-tailed than the fixed-v df=1 mixture, so its
    # 5% threshold must sit ABOVE 2.706 (the regression the reviewer caught).
    assert out["threshold"] > nulls.detection_threshold(0.05, df=1)
    assert np.array_equal(out["detected"], out["T"] >= out["threshold"])


@pytest.mark.slow
def test_completeness_monotone_and_bounded():
    """Stronger injections are detected more often; completeness stays in [0, 1]."""
    s = _setup()
    null = mc.run_null_mc(
        s["sigma_pix"], s["n_pix"], s["dv"], s["v_grid"], n_draws=600,
        seed=11, template=s["template"])
    fit = nulls.fit_gamma_null(null["T"])
    thr = nulls.gamma_null_threshold(0.05, fit["a"], fit["theta"], fit["w0"])

    # Compare a floor-level injection against a strong one (points too close
    # together sit in the ~fap noise floor and their order is not meaningful).
    comps = {}
    for alpha in (0.02, 0.5):
        inj = mc.run_injection_mc(
            alpha=alpha, delta_v=60.0, sigma_pix=s["sigma_pix"], n_pix=s["n_pix"],
            dv=s["dv"], v_grid=s["v_grid"], n_draws=300, seed=3,
            template=s["template"])
        c = mc.completeness(inj["T"], thr)
        assert 0.0 <= c <= 1.0
        comps[alpha] = c
    assert comps[0.5] > comps[0.02] + 0.3   # a real gradient, not all flat
    assert comps[0.02] <= 0.2               # weak injection ~ false-alarm floor
    assert comps[0.5] >= 0.7                # strong injection reliably detected

    # Anti-conservative check: thresholding at the fixed-v 2.706 can only INFLATE
    # completeness relative to the (heavier) calibrated scanned threshold.
    inj_mid = mc.run_injection_mc(
        alpha=0.15, delta_v=60.0, sigma_pix=s["sigma_pix"], n_pix=s["n_pix"],
        dv=s["dv"], v_grid=s["v_grid"], n_draws=300, seed=3,
        template=s["template"])
    c_calib = mc.completeness(inj_mid["T"], thr)
    c_fixed = mc.completeness(inj_mid["T"], nulls.detection_threshold(0.05, df=1))
    assert c_fixed >= c_calib


def test_calibrate_null_keys_and_ranges():
    s = _setup()
    null = mc.run_null_mc(
        s["sigma_pix"], s["n_pix"], s["dv"], s["v_grid"], n_draws=400,
        seed=5, template=s["template"])
    cal = mc.calibrate_null(null["T"])
    assert set(cal) >= {"gamma", "effective_chi2", "w0"}
    assert 0.0 <= cal["w0"] <= 1.0
    assert cal["gamma"]["a"] > 0 and cal["gamma"]["theta"] > 0
