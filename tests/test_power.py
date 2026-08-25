"""Acceptance tests for ``specbeat.power`` (non-centrality, power, Fisher).

Oracle: the closed forms ``lambda = alpha^2 I``, ``P_det = Phi_bar(sqrt(c) -
sqrt(lambda))``, and ``I_alpha = r^T W r`` with an explicit W-orthogonal
projection.
"""

import numpy as np
import pytest
from scipy.stats import norm

from specbeat import nulls, power


# =============================================================================
class TestNoncentrality:
    def test_noncentrality_is_alpha_squared_times_I(self):
        for alpha, I in ((0.05, 2500.0), (0.1, 900.0)):
            np.testing.assert_allclose(
                power.noncentrality(alpha, I), alpha ** 2 * I, rtol=1e-9
            )

    def test_noncentrality_accepts_array(self):
        alpha = np.array([0.0, 0.05, 0.1])
        np.testing.assert_allclose(
            power.noncentrality(alpha, 1000.0), alpha ** 2 * 1000.0, rtol=1e-9
        )


# =============================================================================
class TestDetectionPower:
    """spec s3.3: power of the ONE-SIDED CONSTRAINED test."""

    I_ALPHA = 2500.0

    def _expected(self, alpha, fap=0.05):
        c = nulls.detection_threshold(fap=fap, df=1)
        lam = np.asarray(alpha, float) ** 2 * self.I_ALPHA
        return norm.sf(np.sqrt(c) - np.sqrt(lam))

    def test_power_matches_constrained_one_sided_form(self):
        for fap in (0.05, 0.01):
            for alpha in (0.0, 0.02, 0.05, 0.1):
                np.testing.assert_allclose(
                    power.detection_power(alpha, self.I_ALPHA, fap=fap, df=1),
                    self._expected(alpha, fap), rtol=1e-9,
                )

    def test_power_at_zero_alpha_equals_fap(self):
        # Trap 10: constrained one-sided power -> fap (NOT 2*fap) as alpha->0.
        for fap in (0.05, 0.01):
            np.testing.assert_allclose(
                power.detection_power(0.0, self.I_ALPHA, fap=fap, df=1), fap, rtol=1e-6
            )

    def test_power_monotone_increasing_in_alpha(self):
        alpha = np.array([0.0, 0.01, 0.02, 0.05, 0.1, 0.2])
        p = power.detection_power(alpha, self.I_ALPHA, fap=0.05, df=1)
        assert np.all(np.diff(p) > 0)

    def test_power_to_one_for_large_alpha(self):
        p = power.detection_power(5.0, self.I_ALPHA, fap=0.05, df=1)
        np.testing.assert_allclose(p, 1.0, atol=1e-8)

    def test_power_accepts_array(self):
        alpha = np.array([0.0, 0.05, 0.2])
        np.testing.assert_allclose(
            power.detection_power(alpha, self.I_ALPHA, fap=0.05, df=1),
            self._expected(alpha), rtol=1e-9,
        )

    def test_df_gt_one_not_implemented(self):
        with pytest.raises(NotImplementedError):
            power.detection_power(0.05, self.I_ALPHA, fap=0.05, df=2)


# =============================================================================
class TestFisherInformation:
    def test_fisher_information_orthogonal_projection(self):
        n = 200
        zeta = np.linspace(0.01, 1.0, n)
        design = np.column_stack([np.ones(n), zeta ** 2])
        weights = np.ones(n)
        raw = np.cos(2 * np.pi * zeta * 3.0)
        Q, _ = np.linalg.qr(design)
        resid = raw - Q @ (Q.T @ raw)
        template = design @ np.array([0.7, -0.3]) + resid
        I_alpha = power.fisher_information_alpha(template, design, weights)
        np.testing.assert_allclose(I_alpha, float(resid @ resid), rtol=1e-9)

    def test_fisher_information_weighted_metric(self):
        n = 50
        zeta = np.linspace(0.01, 1.0, n)
        design = np.column_stack([np.ones(n), zeta ** 2])
        weights = 1.0 + zeta
        template = 2.0 * design[:, 1]
        I_alpha = power.fisher_information_alpha(template, design, weights)
        np.testing.assert_allclose(I_alpha, 0.0, atol=1e-9)

    def test_fisher_information_no_baseline_is_weighted_norm(self):
        # design=None -> r = t -> I_alpha = t^T W t (the phase-channel seam).
        n = 80
        rng = np.random.default_rng(3)
        t = rng.standard_normal(n)
        w = 1.0 + rng.random(n)
        I_alpha = power.fisher_information_alpha(t, None, w)
        np.testing.assert_allclose(I_alpha, float((t * w) @ t), rtol=1e-12)
