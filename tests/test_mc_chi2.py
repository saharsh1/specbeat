"""Monte Carlo: chi^2 / dof_eff distribution under the null (``specbeat.amplitude``).

A 500-draw null calibration: the empirical chi^2/dof_eff lies in the analytic
neighbourhood. Marked ``slow``.

Effective dof per realization: ``dof_eff = N − (p − n_active)`` where
``n_active`` is the number of active inequality constraints at the optimum.
"""

from __future__ import annotations

import numpy as np
import pytest

from specbeat import amplitude as amp


@pytest.mark.slow
def test_mc_chi2_dof_eff_under_null():
    rng = np.random.default_rng(20260509)

    n_pad = 256
    dv = 1.0
    zeta_full = np.fft.rfftfreq(n_pad, d=dv)[1:]
    zeta = zeta_full[zeta_full <= 0.05]
    M = amp.construct_log_amp_model_matrix(zeta, trial_v=0.0, n_cos_terms=None,
                                           n_baseline=2)
    weights = np.ones(M.shape[0])
    bounds = amp.make_baseline_bounds(n_baseline=2, n_cos_terms=None)

    N = M.shape[0]
    p = M.shape[1]
    theta_best = np.array([0.5, -2.0, -10.0])

    n_trials = 500
    chi2_arr = np.empty(n_trials)
    dof_eff_arr = np.empty(n_trials, dtype=int)

    for i in range(n_trials):
        y = M @ theta_best + rng.standard_normal(N)
        _, _, chi2, mask = amp.bounded_weighted_least_squares(
            M, y, weights, bounds=bounds, return_active=True)
        chi2_arr[i] = chi2
        dof_eff_arr[i] = N - (p - int(mask.sum()))

    ratio = chi2_arr / dof_eff_arr
    median = float(np.median(ratio))
    q95 = float(np.quantile(ratio, 0.95))

    median_dof = float(np.median(dof_eff_arr))
    expected_q95 = 1.0 + 1.645 * np.sqrt(2.0 / median_dof)

    assert 0.7 <= median <= 1.3, f"median chi2/dof_eff = {median:.3f}, expected ~1.0"
    assert 0.7 * expected_q95 <= q95 <= 1.3 * expected_q95, (
        f"q95 chi2/dof_eff = {q95:.3f}, expected ~ {expected_q95:.3f} (±30%)"
    )
