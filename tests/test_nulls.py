"""Acceptance tests for ``specbeat.nulls`` (boundary-aware one-sided null).

Power, Fisher information and non-centrality are tested in ``test_power.py``.

Independent oracle: ``scipy.stats.{chi2, gamma, norm}`` + explicit re-derivation
of each closed form. The module is never called against itself.
"""

import numpy as np
import pytest
from scipy.special import polygamma
from scipy.stats import chi2, gamma, kstest, norm

from specbeat import nulls


# --- independent oracle helpers (the spec's closed forms) --------------------

def _chi2_sf(x, df):
    """Survival P(chi2_df >= x), with chi2_0 a point mass at 0."""
    x = np.asarray(x, float)
    if df == 0:
        return np.where(x > 0.0, 0.0, 1.0)
    return chi2.sf(x, df)


def _mixture_sf(T, df):
    return 0.5 * (_chi2_sf(T, df - 1) + _chi2_sf(T, df))


def _truncated_ul(alpha_hat, sigma, cl):
    q = norm.cdf((0.0 - alpha_hat) / sigma)
    return alpha_hat + sigma * norm.ppf(q + cl * (1.0 - q))


RTOL = 1e-9


# =============================================================================
class TestSignificanceSqrtIdentity:
    def test_significance_equals_sqrt_T(self):
        for T in (0.5, 1.0, 2.706, 5.0, 9.0):
            np.testing.assert_allclose(
                nulls.significance(T, df=1), np.sqrt(T), rtol=RTOL
            )

    def test_significance_accepts_array(self):
        T = np.array([0.5, 1.0, 4.0, 9.0])
        np.testing.assert_allclose(
            nulls.significance(T, df=1), np.sqrt(T), rtol=RTOL
        )


# =============================================================================
class TestDetectionThreshold:
    def test_threshold_df1_fap05_is_2p706(self):
        c = nulls.detection_threshold(fap=0.05, df=1)
        np.testing.assert_allclose(c, 2.705543454095418, rtol=1e-6)
        np.testing.assert_allclose(c, chi2.ppf(1 - 2 * 0.05, 1), rtol=RTOL)

    def test_threshold_df1_fap05_sqrt_is_norm_ppf_095(self):
        c = nulls.detection_threshold(fap=0.05, df=1)
        np.testing.assert_allclose(np.sqrt(c), norm.ppf(0.95), rtol=1e-6)

    def test_threshold_df1_fap01_sqrt_is_norm_ppf_099(self):
        c = nulls.detection_threshold(fap=0.01, df=1)
        np.testing.assert_allclose(np.sqrt(c), norm.ppf(0.99), rtol=1e-6)

    def test_threshold_general_df_inverts_the_mixture(self):
        for df in (1, 2, 3):
            for fap in (0.05, 0.01):
                c = nulls.detection_threshold(fap=fap, df=df)
                np.testing.assert_allclose(_mixture_sf(c, df), fap, rtol=1e-6)
                np.testing.assert_allclose(nulls.pvalue(c, df=df), fap, rtol=1e-6)


# =============================================================================
class TestPvalueOneSided:
    @pytest.mark.parametrize("T", [0.5, 1.0, 2.706, 4.0, 9.0])
    def test_pvalue_df1_equals_half_chi2_tail(self, T):
        np.testing.assert_allclose(
            nulls.pvalue(T, df=1), 0.5 * chi2.sf(T, 1), rtol=RTOL
        )

    @pytest.mark.parametrize("T", [0.5, 1.0, 2.706, 4.0, 9.0])
    def test_pvalue_df1_equals_normsf_sqrtT(self, T):
        np.testing.assert_allclose(
            nulls.pvalue(T, df=1), norm.sf(np.sqrt(T)), rtol=RTOL
        )

    def test_pvalue_exact_factor_two_vs_two_sided(self):
        for T in (0.5, 1.0, 2.706, 4.0, 9.0):
            np.testing.assert_allclose(
                nulls.pvalue(T, df=1) / chi2.sf(T, 1), 0.5, rtol=RTOL
            )

    def test_pvalue_accepts_array(self):
        T = np.array([0.5, 1.0, 4.0, 9.0])
        np.testing.assert_allclose(
            nulls.pvalue(T, df=1), 0.5 * chi2.sf(T, 1), rtol=RTOL
        )


# =============================================================================
class TestBoundaryEdgeCase:
    """spec s3 BOUNDARY: T <= 0 => p == 1, significance == 0 (NOT 0.5)."""

    @pytest.mark.parametrize("df", [1, 2, 3])
    def test_pvalue_at_zero_is_one(self, df):
        assert nulls.pvalue(0.0, df=df) == pytest.approx(1.0, abs=0.0)

    @pytest.mark.parametrize("df", [1, 2, 3])
    def test_pvalue_negative_is_one(self, df):
        assert nulls.pvalue(-3.0, df=df) == pytest.approx(1.0, abs=0.0)

    @pytest.mark.parametrize("df", [1, 2, 3])
    def test_significance_at_zero_is_zero(self, df):
        assert nulls.significance(0.0, df=df) == pytest.approx(0.0, abs=0.0)

    @pytest.mark.parametrize("df", [1, 2, 3])
    def test_significance_negative_is_zero(self, df):
        assert nulls.significance(-3.0, df=df) == pytest.approx(0.0, abs=0.0)

    def test_boundary_is_not_half(self):
        assert nulls.pvalue(0.0, df=1) != pytest.approx(0.5)


# =============================================================================
class TestMixtureDfGreaterThanOne:
    @pytest.mark.parametrize("df", [2, 3])
    @pytest.mark.parametrize("T", [0.5, 2.0, 5.0, 9.0])
    def test_pvalue_matches_general_mixture(self, df, T):
        np.testing.assert_allclose(
            nulls.pvalue(T, df=df), _mixture_sf(T, df), rtol=RTOL
        )

    def test_df1_mixture_reduces_to_half_chi2_1(self):
        for T in (0.5, 2.706, 9.0):
            np.testing.assert_allclose(
                nulls.pvalue(T, df=1), 0.5 * chi2.sf(T, 1), rtol=RTOL
            )


# =============================================================================
class TestConstrainNonneg:
    SIGMA = 0.012

    def test_negative_alpha_clamps_and_zeroes_T(self):
        a_uncon = -4.0 * self.SIGMA
        res = nulls.constrain_nonneg(a_uncon, self.SIGMA, T_uncon=7.3, df=1)
        assert res["constraint_active"] is True
        assert res["alpha_hat"] == pytest.approx(0.0, abs=0.0)
        assert res["T"] == pytest.approx(0.0, abs=0.0)
        assert res["p_value"] == pytest.approx(1.0, abs=0.0)

    def test_negative_alpha_ul_uses_raw_not_clamped(self):
        a_uncon = -4.0 * self.SIGMA
        res = nulls.constrain_nonneg(a_uncon, self.SIGMA, T_uncon=7.3, df=1)
        expected = _truncated_ul(a_uncon, self.SIGMA, cl=0.95)
        np.testing.assert_allclose(res["alpha_ul95"], expected, rtol=1e-6)
        assert res["alpha_ul95"] < 1.645 * self.SIGMA

    def test_positive_alpha_passes_through(self):
        a_uncon = 3.0 * self.SIGMA
        res = nulls.constrain_nonneg(a_uncon, self.SIGMA, T_uncon=6.5, df=1)
        assert res["constraint_active"] is False
        np.testing.assert_allclose(res["alpha_hat"], a_uncon, rtol=RTOL)
        np.testing.assert_allclose(res["T"], 6.5, rtol=RTOL)
        np.testing.assert_allclose(res["p_value"], _mixture_sf(6.5, 1), rtol=1e-6)

    @pytest.mark.parametrize("df", [1, 2, 3])
    def test_positive_alpha_pvalue_uses_mixture(self, df):
        res = nulls.constrain_nonneg(2.0 * self.SIGMA, self.SIGMA, T_uncon=5.0, df=df)
        np.testing.assert_allclose(res["p_value"], _mixture_sf(5.0, df), rtol=1e-6)

    def test_returns_all_expected_keys(self):
        res = nulls.constrain_nonneg(0.01, self.SIGMA, T_uncon=4.0, df=1)
        expected_keys = {
            "alpha_hat", "constraint_active", "T", "df", "p_value",
            "alpha_ul95", "alpha_uncon", "sigma_alpha",
        }
        assert expected_keys <= set(res.keys())


# =============================================================================
class TestUpperLimit:
    SIGMA = 0.012

    def test_default_matches_truncated_normal(self):
        for a_hat in (-4 * self.SIGMA, -self.SIGMA, 0.0, self.SIGMA, 3 * self.SIGMA):
            expected = _truncated_ul(a_hat, self.SIGMA, cl=0.95)
            np.testing.assert_allclose(
                nulls.upper_limit(a_hat, self.SIGMA, cl=0.95), expected, rtol=1e-6
            )

    def test_monotone_increasing_in_alpha_hat(self):
        a = np.linspace(-5 * self.SIGMA, 5 * self.SIGMA, 11)
        uls = np.array([nulls.upper_limit(x, self.SIGMA, cl=0.95) for x in a])
        assert np.all(np.diff(uls) > 0)

    def test_large_alpha_collapses_to_wald(self):
        a_hat = 20.0 * self.SIGMA
        np.testing.assert_allclose(
            nulls.upper_limit(a_hat, self.SIGMA, cl=0.95),
            a_hat + norm.ppf(0.95) * self.SIGMA, rtol=1e-6,
        )

    def test_minus_four_sigma_concrete_value(self):
        ul = nulls.upper_limit(-4 * self.SIGMA, self.SIGMA, cl=0.95)
        np.testing.assert_allclose(ul / self.SIGMA, 0.6597, rtol=1e-3)

    def test_power_constrained_is_max_of_truncated_and_floor(self):
        floor = norm.ppf(0.95) * self.SIGMA
        for a_hat in (-4 * self.SIGMA, -self.SIGMA, 0.0, self.SIGMA, 3 * self.SIGMA):
            trunc = nulls.upper_limit(a_hat, self.SIGMA, cl=0.95)
            pc = nulls.upper_limit(a_hat, self.SIGMA, cl=0.95, power_constrained=True)
            np.testing.assert_allclose(pc, max(trunc, floor), rtol=1e-6)

    def test_power_constrained_never_decreases_limit(self):
        for a_hat in np.linspace(-5 * self.SIGMA, 5 * self.SIGMA, 11):
            trunc = nulls.upper_limit(a_hat, self.SIGMA, cl=0.95)
            pc = nulls.upper_limit(a_hat, self.SIGMA, cl=0.95, power_constrained=True)
            assert pc >= trunc - 1e-12

    def test_extreme_negative_alpha_hat_stays_finite(self):
        # Trap 9: the stable isf form stays finite far into alpha_hat < 0.
        uls = []
        for n_sig in (6.0, 8.0, 12.0, 20.0, 30.0):
            ul = nulls.upper_limit(-n_sig * self.SIGMA, self.SIGMA, cl=0.95)
            assert np.isfinite(ul), f"UL not finite at -{n_sig} sigma"
            assert ul > 0.0, f"UL not positive at -{n_sig} sigma"
            uls.append(ul)
        assert np.all(np.diff(uls) < 0)


# =============================================================================
class TestEffectiveChi2:
    """MC-calibrated effective-chi2 null (spec s5.4)."""

    @pytest.mark.parametrize("T", [0.25, 0.5, 1.0, 2.706, 4.0, 9.0])
    def test_reduces_to_chernoff_df1_via_pvalue(self, T):
        np.testing.assert_allclose(
            nulls.effective_chi2_pvalue(T, nu_eff=1.0, w0=0.5),
            nulls.pvalue(T, df=1), rtol=1e-12,
        )

    @pytest.mark.parametrize("nu_eff, w0", [(1.0, 0.5), (5.0, 0.3), (8.0, 0.0)])
    @pytest.mark.parametrize("T", [0.0, -1.0, -5.0])
    def test_boundary_edge_is_one(self, nu_eff, w0, T):
        assert nulls.effective_chi2_pvalue(T, nu_eff=nu_eff, w0=w0) == \
            pytest.approx(1.0, abs=0.0)

    def test_boundary_edge_not_one_minus_w0(self):
        p = nulls.effective_chi2_pvalue(0.0, nu_eff=5.0, w0=0.3)
        assert p == pytest.approx(1.0, abs=0.0)
        assert p != pytest.approx(0.7)

    @pytest.mark.parametrize("nu_eff", [1.0, 2.0, 4.5, 8.0])
    @pytest.mark.parametrize("w0", [0.0, 0.3, 0.5, 0.7])
    @pytest.mark.parametrize("T", [0.5, 2.0, 5.0, 12.0])
    def test_formula_for_positive_T(self, nu_eff, w0, T):
        np.testing.assert_allclose(
            nulls.effective_chi2_pvalue(T, nu_eff=nu_eff, w0=w0),
            (1.0 - w0) * chi2.sf(T, nu_eff), rtol=1e-12,
        )

    def test_array_T_returns_array(self):
        T = np.array([0.5, 2.0, 5.0, 12.0])
        out = nulls.effective_chi2_pvalue(T, nu_eff=4.0, w0=0.3)
        assert isinstance(out, np.ndarray)
        assert out.shape == T.shape

    def test_scalar_T_returns_float(self):
        assert isinstance(nulls.effective_chi2_pvalue(3.0, nu_eff=4.0, w0=0.3), float)

    def test_strictly_decreasing_in_T(self):
        T = np.linspace(0.05, 30.0, 400)
        p = nulls.effective_chi2_pvalue(T, nu_eff=5.0, w0=0.3)
        assert np.all(np.diff(p) < 0)

    @pytest.mark.parametrize("w0_true, nu_true", [(0.30, 5.0), (0.0, 8.0), (0.5, 1.0)])
    def test_fit_recovers_mixture_params(self, w0_true, nu_true):
        rng = np.random.default_rng(12345)
        N = 200_000
        n0 = rng.binomial(N, w0_true)
        cont = chi2.rvs(nu_true, size=N - n0, random_state=rng)
        samples = np.concatenate([np.zeros(n0), cont])
        rng.shuffle(samples)
        fit = nulls.fit_effective_chi2(samples)
        se_w0 = np.sqrt(max(w0_true * (1 - w0_true), 1e-6) / N)
        np.testing.assert_allclose(fit["w0"], w0_true, atol=5 * se_w0 + 1e-4)
        se_nu = np.sqrt(2.0 * nu_true / (N - n0))
        np.testing.assert_allclose(fit["nu_eff"], nu_true, atol=6 * se_nu)
        np.testing.assert_allclose(fit["var_ratio"], 1.0, atol=0.05)

    def test_fit_returns_expected_keys(self):
        rng = np.random.default_rng(7)
        samples = chi2.rvs(3.0, size=10_000, random_state=rng)
        fit = nulls.fit_effective_chi2(samples)
        assert {"nu_eff", "w0", "var_ratio"} <= set(fit.keys())

    def test_var_ratio_departs_from_one_for_scaled_chi2(self):
        rng = np.random.default_rng(31337)
        nu = 5.0
        samples = 2.0 * chi2.rvs(nu, size=300_000, random_state=rng)
        fit = nulls.fit_effective_chi2(samples)
        np.testing.assert_allclose(fit["var_ratio"], 2.0, rtol=0.05)
        assert fit["var_ratio"] > 1.5

    def test_logp_error_delta_method_against_manual(self):
        rng = np.random.default_rng(5)
        nu_true, w0_true, n = 8.0, 0.3, 4000
        pos = chi2.rvs(nu_true, size=int(n * (1 - w0_true)), random_state=rng)
        samples = np.concatenate([np.zeros(n - pos.size), pos])
        T = 30.0
        out = nulls.effective_chi2_pvalue_error(T, samples)
        fit = nulls.fit_effective_chi2(samples)
        eps = 1e-4 * fit["nu_eff"]
        dlnS = (np.log(chi2.sf(T, fit["nu_eff"] + eps))
                - np.log(chi2.sf(T, fit["nu_eff"] - eps))) / (2 * eps)
        var_nu = np.var(pos) / pos.size
        var_term_w0 = fit["w0"] / ((1.0 - fit["w0"]) * samples.size)
        sd_manual = np.sqrt(dlnS ** 2 * var_nu + var_term_w0)
        np.testing.assert_allclose(out["sd_ln_p"], sd_manual, rtol=1e-3)
        assert 0.0 < out["sd_ln_p"] < 1.0

    def test_logp_error_degenerate_sample(self):
        out = nulls.effective_chi2_pvalue_error(5.0, np.zeros(100))
        assert np.isnan(out["sd_ln_p"])


# =============================================================================
class TestGammaNullFit:
    """``fit_gamma_null`` (spec s5.2). floc=0; ks_stat only; warns a<1."""

    def test_returns_expected_keys(self):
        rng = np.random.default_rng(1)
        pos = gamma.rvs(3.0, scale=2.0, size=20_000, random_state=rng)
        fit = nulls.fit_gamma_null(pos)
        expected = {"w0", "a", "theta", "ks_stat", "n_pos", "mean_pos", "var_pos"}
        assert expected <= set(fit.keys())

    def test_ml_recovers_known_a_theta(self):
        rng = np.random.default_rng(101)
        a_true, theta_true, n = 3.0, 2.0, 100_000
        pos = gamma.rvs(a_true, scale=theta_true, size=n, random_state=rng)
        fit = nulls.fit_gamma_null(pos)
        np.testing.assert_allclose(fit["a"], a_true, rtol=0.05)
        np.testing.assert_allclose(fit["theta"], theta_true, rtol=0.05)
        assert fit["w0"] == pytest.approx(0.0, abs=0.0)
        assert fit["n_pos"] == n

    def test_ks_stat_present_no_pvalue_exposed(self):
        rng = np.random.default_rng(505)
        pos = gamma.rvs(3.0, scale=2.0, size=20_000, random_state=rng)
        fit = nulls.fit_gamma_null(pos)
        assert "ks_stat" in fit
        assert "ks_p" not in fit
        assert "ks_pvalue" not in fit
        d = kstest(pos, "gamma", args=(fit["a"], 0.0, fit["theta"])).statistic
        np.testing.assert_allclose(fit["ks_stat"], d, rtol=1e-9)

    def test_too_few_positives_returns_nan_not_exception(self):
        samples = np.concatenate([np.zeros(500), np.array([1.0, 2.0, 3.0])])
        fit = nulls.fit_gamma_null(samples, min_pos=30)
        assert np.isnan(fit["a"])
        assert np.isnan(fit["theta"])
        assert np.isnan(fit["ks_stat"])
        assert fit["n_pos"] == 3
        assert fit["w0"] == pytest.approx(500 / 503)

    def test_shape_below_one_warns(self):
        rng = np.random.default_rng(404)
        pos = gamma.rvs(0.5, scale=2.0, size=50_000, random_state=rng)
        with pytest.warns(RuntimeWarning):
            fit = nulls.fit_gamma_null(pos)
        assert fit["a"] < 1.0


# =============================================================================
class TestGammaNullPvalue:
    @pytest.mark.parametrize("T", [0.5, 2.0, 5.0, 12.0])
    @pytest.mark.parametrize("a", [0.5, 1.5, 4.0])
    @pytest.mark.parametrize("theta", [1.0, 2.0, 3.5])
    @pytest.mark.parametrize("w0", [0.0, 0.3, 0.5])
    def test_formula_for_positive_T(self, T, a, theta, w0):
        np.testing.assert_allclose(
            nulls.gamma_null_pvalue(T, a, theta, w0),
            (1.0 - w0) * gamma.sf(T, a, scale=theta), rtol=1e-12,
        )

    @pytest.mark.parametrize("T", [0.0, -1.0, -5.0])
    def test_boundary_edge_is_one(self, T):
        assert nulls.gamma_null_pvalue(T, 3.0, 2.0, 0.3) == pytest.approx(1.0, abs=0.0)

    def test_scalar_T_returns_float(self):
        assert isinstance(nulls.gamma_null_pvalue(3.0, 3.0, 2.0, 0.3), float)

    @pytest.mark.parametrize("nu, w0", [(1.0, 0.5), (5.0, 0.3), (8.0, 0.0)])
    @pytest.mark.parametrize("T", [0.5, 2.0, 9.0, 20.0])
    def test_nests_effective_chi2_at_a_half_nu_theta_two(self, nu, w0, T):
        np.testing.assert_allclose(
            nulls.gamma_null_pvalue(T, nu / 2.0, 2.0, w0),
            nulls.effective_chi2_pvalue(T, nu, w0), rtol=1e-12,
        )

    @pytest.mark.parametrize("T", [0.5, 2.706, 4.0, 9.0])
    def test_reduces_to_df1_chernoff(self, T):
        np.testing.assert_allclose(
            nulls.gamma_null_pvalue(T, 0.5, 2.0, 0.5), 0.5 * chi2.sf(T, 1), rtol=1e-12,
        )
        np.testing.assert_allclose(
            nulls.gamma_null_pvalue(T, 0.5, 2.0, 0.5), nulls.pvalue(T, df=1), rtol=1e-12,
        )


# =============================================================================
class TestGammaNullPvalueError:
    def _make_sample(self, seed, a_true=3.0, theta_true=2.0, w0_true=0.3, N=8000):
        rng = np.random.default_rng(seed)
        n0 = int(round(w0_true * N))
        pos = gamma.rvs(a_true, scale=theta_true, size=N - n0, random_state=rng)
        return np.concatenate([np.zeros(n0), pos])

    def test_returns_expected_keys(self):
        out = nulls.gamma_null_pvalue_error(20.0, self._make_sample(14))
        expected = {"sd_ln_p", "sd_log10_p", "ga", "gth", "var_a", "var_theta",
                    "cov_at", "fit"}
        assert expected <= set(out.keys())

    def test_delta_method_against_manual(self):
        samples = self._make_sample(11)
        T = 20.0
        out = nulls.gamma_null_pvalue_error(T, samples)
        fit = nulls.fit_gamma_null(samples)
        a, theta, w0, n_pos = fit["a"], fit["theta"], fit["w0"], fit["n_pos"]
        psi1 = polygamma(1, a)
        D = a * psi1 - 1.0
        var_a = a / (D * n_pos)
        var_theta = theta ** 2 * psi1 / (D * n_pos)
        cov_at = -theta / (D * n_pos)
        eps_a = max(1e-4 * a, 1e-6)
        ga = (np.log(gamma.sf(T, a + eps_a, scale=theta))
              - np.log(gamma.sf(T, a - eps_a, scale=theta))) / (2 * eps_a)
        eps_th = max(1e-4 * theta, 1e-6)
        gth = (np.log(gamma.sf(T, a, scale=theta + eps_th))
               - np.log(gamma.sf(T, a, scale=theta - eps_th))) / (2 * eps_th)
        var_lnp = (ga ** 2 * var_a + 2.0 * ga * gth * cov_at
                   + gth ** 2 * var_theta + w0 / ((1.0 - w0) * samples.size))
        np.testing.assert_allclose(out["sd_ln_p"], np.sqrt(var_lnp), rtol=1e-3)
        assert 0.0 < out["sd_ln_p"] < 1.0

    def test_theta_gradient_matches_analytic(self):
        samples = self._make_sample(12)
        T = 18.0
        out = nulls.gamma_null_pvalue_error(T, samples)
        a, theta = out["fit"]["a"], out["fit"]["theta"]
        analytic = (T / theta) * gamma.pdf(T, a, scale=theta) \
            / gamma.sf(T, a, scale=theta)
        np.testing.assert_allclose(out["gth"], analytic, rtol=1e-3)

    def test_fisher_block_signs(self):
        out = nulls.gamma_null_pvalue_error(20.0, self._make_sample(13))
        a = out["fit"]["a"]
        D = a * polygamma(1, a) - 1.0
        assert D > 0.0
        assert out["var_a"] > 0.0
        assert out["var_theta"] > 0.0
        assert out["cov_at"] < 0.0

    def test_sd_log10_is_sd_ln_over_ln10(self):
        out = nulls.gamma_null_pvalue_error(20.0, self._make_sample(15))
        np.testing.assert_allclose(
            out["sd_log10_p"], out["sd_ln_p"] / np.log(10.0), rtol=1e-12
        )

    def test_degenerate_sample_is_nan(self):
        out = nulls.gamma_null_pvalue_error(5.0, np.zeros(100))
        assert np.isnan(out["sd_ln_p"])


# =============================================================================
class TestGammaNullThreshold:
    """NEW: ``gamma_null_threshold(fap, a, theta, w0)`` inverter."""

    @pytest.mark.parametrize("a, theta, w0", [(2.7, 2.4, 0.15), (3.5, 5.5, 0.25),
                                              (0.5, 2.0, 0.5)])
    @pytest.mark.parametrize("fap", [0.05, 0.01, 1e-3])
    def test_inverts_gamma_null_pvalue(self, a, theta, w0, fap):
        c = nulls.gamma_null_threshold(fap, a, theta, w0)
        # The threshold must reproduce the requested FAP through the p-value.
        np.testing.assert_allclose(nulls.gamma_null_pvalue(c, a, theta, w0), fap,
                                   rtol=1e-6)

    @pytest.mark.parametrize("a, theta, w0", [(2.7, 2.4, 0.15), (3.5, 5.5, 0.25)])
    def test_matches_gamma_isf_closed_form(self, a, theta, w0):
        fap = 0.02
        c = nulls.gamma_null_threshold(fap, a, theta, w0)
        np.testing.assert_allclose(c, gamma.isf(fap / (1.0 - w0), a, scale=theta),
                                   rtol=1e-9)

    def test_fap_at_or_above_continuum_mass_returns_zero(self):
        # fap >= 1 - w0 -> the requested tail sits inside the atom -> c = 0.
        w0 = 0.4
        assert nulls.gamma_null_threshold(1.0 - w0, 3.0, 2.0, w0) == pytest.approx(0.0)
        assert nulls.gamma_null_threshold(0.9, 3.0, 2.0, w0) == pytest.approx(0.0)

    def test_threshold_decreases_with_larger_fap(self):
        a, theta, w0 = 2.7, 2.4, 0.15
        cs = [nulls.gamma_null_threshold(f, a, theta, w0)
              for f in (1e-4, 1e-3, 1e-2, 5e-2)]
        assert np.all(np.diff(cs) < 0)
