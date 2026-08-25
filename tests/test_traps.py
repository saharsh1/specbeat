"""Fifteen ways to get this wrong, one test each.

Every one of these was a real mistake at some point, either in the research code
this was extracted from or while writing this package. They are regression tests
against the failure, not unit tests of an interface.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import chi2, gamma, norm

from specbeat import amplitude as amp
from specbeat import detect, mc, nulls, power
from specbeat.simulate import generate_simulated_spectrum, generate_wavelength_grid


# --- Trap 1: chi2.sf(x, 0) NaN guard; df=1 is exactly half a chi2_1 tail ------
def test_trap01_chi2_sf_zero_df_guard():
    for T in (0.5, 2.706, 9.0):
        p = nulls.pvalue(T, df=1)
        assert np.isfinite(p), "pvalue must not be NaN (chi2.sf(.,0) guard)"
        np.testing.assert_allclose(p, 0.5 * chi2.sf(T, 1), rtol=1e-12)
    # df=2 uses the df-1==1 term and must also stay finite.
    assert np.isfinite(nulls.pvalue(3.0, df=2))


# --- Trap 2: threshold is 2.706, NOT 3.841 -----------------------------------
def test_trap02_threshold_is_2p706_not_3p841():
    c = nulls.detection_threshold(fap=0.05, df=1)
    assert c == pytest.approx(2.706, abs=1e-3)
    assert c != pytest.approx(3.841, abs=1e-2)


# --- Trap 3: boundary atom in the right tail (T<=0 => p=1, sigma=0) -----------
def test_trap03_boundary_atom_right_tail():
    for T in (0.0, -1e-9, -5.0):
        assert nulls.pvalue(T, df=1) == pytest.approx(1.0, abs=0.0)
        assert nulls.significance(T, df=1) == pytest.approx(0.0, abs=0.0)


# --- Trap 4: CLIP-BEFORE-MAX bit-identical between detect and mc --------------
def test_trap04_clip_before_max_bit_identical():
    """detect.detection_statistic and mc's per-draw T are bit-identical.

    Both must route through the single ``blind_null_Tmax`` owner of the
    clip-before-max rule, on the SAME spectrum and SAME sigma_ft.
    """
    dv = 1.0
    n_pix = 256
    v_grid = np.arange(-100.0, 101.0, 10.0)
    sigma_pix = 0.02
    sigma_ft = amp.amp_noise_sigma(sigma_pix, n=n_pix)

    out = mc.run_null_mc(sigma_pix, n_pix, dv, v_grid, n_draws=5, seed=0,
                         noise="white", sigma_ft=sigma_ft, return_draws=True)

    assert "draws" in out and "T" in out
    for spec, T_mc in zip(out["draws"], out["T"]):
        T_detect = detect.detection_statistic(spec, dv, v_grid, sigma_ft=sigma_ft)["T"]
        np.testing.assert_array_equal(T_detect, T_mc)


# --- Trap 5: p_emp uses (1+k)/(n+1), '>=', and finite filtering ---------------
def test_trap05_pemp_rank_and_finite_filter():
    null = np.linspace(0.0, 5.0, 100)          # n = 100 finite draws
    # k = 0: nothing exceeds a huge T_obs.
    assert mc.p_emp(1e6, null) == pytest.approx(1.0 / (100 + 1))
    # T_obs = 0 with a non-negative null: every draw satisfies '>= 0' -> ~1.
    assert mc.p_emp(0.0, null) == pytest.approx((1 + 100) / (100 + 1))
    # NaNs are dropped from the denominator (n_finite, not len).
    null_nan = np.concatenate([null, np.full(10, np.nan)])
    assert mc.p_emp(1e6, null_nan) == pytest.approx(1.0 / (100 + 1))


# --- Trap 6: log-of-noisy-amplitude (Rice) tail biases UP; truncation fixes ---
def test_trap06_rice_tail_bias_removed_by_snr_cutoff():
    dv = 1.0
    N = 1024
    wave = generate_wavelength_grid(5875.6, dv, N)
    signal, _, _ = generate_simulated_spectrum(
        wave, dv, wavelength_0=5875.6, line_depth=0.5, v=0.0, vsini=10.0,
        vG=8.0, gamma=6.0, epsilon=0.5, second_component=False)
    signal = np.asarray(signal, float)
    noiseless_amp = np.abs(np.fft.rfft(signal))[1:]

    rng = np.random.default_rng(0)
    noisy = signal + rng.normal(0.0, 5e-3, N)
    amp_arr, log_amp, zeta, sigma_A = amp.fourier_log_amplitude(
        noisy, dv, normalize_to_snr=False)
    amp_SNR = amp_arr / sigma_A

    z_loose, _, _ = amp.truncate_to_snr_cutoff(
        zeta, log_amp, amp_arr, amp_SNR, threshold=2.0)
    z_strict, _, _ = amp.truncate_to_snr_cutoff(
        zeta, log_amp, amp_arr, amp_SNR, threshold=5.0)

    # (a) stricter SNR cut removes more of the tail.
    assert len(z_strict) < len(z_loose)

    # (b) in the removed noise tail the observed log|FT| sits ABOVE the true
    # (decayed) signal envelope: the upward log-of-noise bias. Restrict to the
    # points where the noiseless amplitude is still representable; further out it
    # underflows to zero, and log(0) = -inf would satisfy the comparison for free.
    tail = slice(len(z_strict), None)
    finite = noiseless_amp[tail] > 0.0
    assert finite.sum() > 10
    assert (np.mean(log_amp[tail][finite])
            > np.mean(np.log(noiseless_amp[tail][finite])) + 1.0)


# --- Trap 7: sigma_A from the Rayleigh tail == sigma_pix*sqrt(N/2) ------------
def test_trap07_amp_noise_sigma_matches_rayleigh_tail():
    rng = np.random.default_rng(1)
    N = 8192
    sigma_pix = 0.05
    noise = rng.normal(0.0, sigma_pix, N)
    amp_arr = np.abs(np.fft.rfft(noise))[1:]
    # Rayleigh median = scale * sqrt(ln 4); recover the scale.
    sigma_R_tail = np.median(amp_arr) / np.sqrt(np.log(4.0))
    expected = amp.amp_noise_sigma(sigma_pix, n=N)
    np.testing.assert_allclose(sigma_R_tail, expected, rtol=0.05)


# --- Trap 8: DC dropped (fit zeta grid is rfftfreq[1:]) -----------------------
def test_trap08_dc_dropped():
    dv = 1.0
    N = 256
    rng = np.random.default_rng(2)
    spec = rng.standard_normal(N)
    _, _, zeta, _ = amp.fourier_log_amplitude(spec, dv)
    assert zeta[0] > 0.0
    np.testing.assert_allclose(zeta[0], np.fft.rfftfreq(N, d=dv)[1], rtol=1e-12)
    assert len(zeta) == N // 2  # rfftfreq has N//2+1, minus the dropped DC


# --- Trap 9: upper_limit stays finite at alpha_hat = -10 sigma ----------------
def test_trap09_upper_limit_stable_deep_negative():
    sigma = 0.01
    ul = nulls.upper_limit(-10.0 * sigma, sigma, cl=0.95)
    assert np.isfinite(ul) and ul > 0.0
    # matches the stable isf form directly.
    s = norm.cdf(-10.0)
    expected = -10.0 * sigma + sigma * norm.isf(s * (1.0 - 0.95))
    np.testing.assert_allclose(ul, expected, rtol=1e-6)


# --- Trap 10: constrained power -> fap (NOT 2*fap) as alpha -> 0 --------------
def test_trap10_power_to_fap_at_zero_alpha():
    for fap in (0.05, 0.01):
        p0 = power.detection_power(0.0, 2500.0, fap=fap, df=1)
        assert p0 == pytest.approx(fap, rel=1e-6)
        assert p0 != pytest.approx(2 * fap, rel=1e-6)


# --- Trap 11: fit_gamma_null uses floc=0, ks_stat only, warns a<1 ------------
def test_trap11_gamma_null_floc0_no_kspvalue_warns():
    rng = np.random.default_rng(11)
    pos = gamma.rvs(3.0, scale=2.0, size=80_000, random_state=rng)
    fit = nulls.fit_gamma_null(pos)
    np.testing.assert_allclose(fit["a"], 3.0, rtol=0.05)
    np.testing.assert_allclose(fit["theta"], 2.0, rtol=0.05)
    assert "ks_stat" in fit
    assert "ks_p" not in fit and "ks_pvalue" not in fit
    # a < 1 warns.
    pos2 = gamma.rvs(0.5, scale=2.0, size=50_000,
                     random_state=np.random.default_rng(12))
    with pytest.warns(RuntimeWarning):
        nulls.fit_gamma_null(pos2)


# --- Trap 12: simulator circular-wrap guard ----------------------------------
def test_trap12_wrap_guard_warns_near_edge():
    dv = 5.0
    N = 128            # half-window = 320 km/s
    wave = generate_wavelength_grid(5875.6, dv, N)
    # Push the line to within (vsini + |v| + few*vG) of the segment edge.
    with pytest.warns(RuntimeWarning):
        generate_simulated_spectrum(
            wave, dv, wavelength_0=5875.6, line_depth=0.5, v=300.0, vsini=20.0,
            vG=20.0, gamma=5.0, epsilon=0.5, second_component=False)


# --- Trap 13: v-scan recovers injected v within one step (step<=1/(4 zmax)) ---
def test_trap13_vscan_recovers_within_one_step():
    dv = 1.0
    N = 512
    zeta_max = 0.02
    step = 1.0 / (4.0 * zeta_max)      # = 12.5 km/s; use a finer grid
    step = min(step, 10.0)
    dv_true = 63.0                     # deliberately off-grid
    wave = generate_wavelength_grid(5875.6, dv, N)
    spec, _, _ = generate_simulated_spectrum(
        wave, dv, wavelength_0=5875.6, line_depth=0.5, v=0.0, vsini=10.0,
        vG=10.0, gamma=5.0, epsilon=0.5, second_component=True,
        v2=dv_true, vsini2=10.0, vG2=10.0, gamma2=5.0, depth_ratio=0.4)
    v_grid = np.arange(0.0, 150.0 + step, step)
    out = detect.detection_statistic(np.asarray(spec, float), dv, v_grid,
                                     zeta_max=zeta_max)
    assert abs(out["v_best"] - dv_true) <= step + 1e-9


# --- Trap 14: small-v degeneracy: amplitude I_alpha -> 0 as v -> 0 ------------
def test_trap14_small_v_fisher_collapses():
    n = 400
    zeta = np.linspace(1e-4, 0.05, n)
    design = np.column_stack([np.ones(n), np.abs(zeta), zeta ** 2])
    weights = np.ones(n)
    vs = np.array([1.0, 2.0, 5.0, 10.0, 20.0])   # increasing v
    I = np.array([
        power.fisher_information_alpha(np.cos(2 * np.pi * zeta * v), design, weights)
        for v in vs
    ])
    # Monotone increasing in v; collapses toward 0 as v -> 0.
    assert np.all(np.diff(I) > 0)
    assert I[0] < I[-1] * 1e-3


# --- Trap 15: AR(1) inflates the Gamma scale theta above 2 -------------------
@pytest.mark.slow
def test_trap15_ar1_inflates_gamma_scale():
    dv = 1.0
    n_pix = 512
    v_grid = np.arange(-150.0, 151.0, 15.0)
    sigma_pix = 0.02
    sigma_ft = amp.amp_noise_sigma(sigma_pix, n=n_pix)
    n_draws = 600

    T_white = mc.run_null_mc(sigma_pix, n_pix, dv, v_grid, n_draws, seed=7,
                             noise="white", sigma_ft=sigma_ft)["T"]
    T_red = mc.run_null_mc(sigma_pix, n_pix, dv, v_grid, n_draws, seed=7,
                           noise=("ar1", 5.0), sigma_ft=sigma_ft)["T"]

    fit_white = nulls.fit_gamma_null(T_white)
    fit_red = nulls.fit_gamma_null(T_red)

    # White (var_ratio ~ 1) => theta ~ 2*var_ratio <= ~2; red noise inflates it.
    assert fit_white["theta"] <= 2.5
    assert fit_red["theta"] > 2.0
    assert fit_red["theta"] > fit_white["theta"]
