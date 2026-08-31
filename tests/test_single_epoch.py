"""The README's scope promise: multi-epoch input raises NotImplementedError."""
import numpy as np
import pytest

from specbeat import amplitude, detect, phase


TWO_EPOCHS = np.zeros((2, 64))


def test_fourier_log_amplitude_rejects_epoch_stack():
    with pytest.raises(NotImplementedError):
        amplitude.fourier_log_amplitude(TWO_EPOCHS, 1.0)


def test_detection_statistic_rejects_epoch_stack():
    with pytest.raises(NotImplementedError):
        detect.detection_statistic(TWO_EPOCHS, 1.0, np.array([30.0, 60.0]),
                                   sigma_ft=0.1, zeta_max=0.05)


def test_phase_fit_rejects_epoch_stack():
    with pytest.raises(NotImplementedError):
        phase.fit_phase_alpha(np.zeros((2, 2, 8), dtype=complex),
                              np.linspace(0.001, 0.008, 8),
                              rest_shift_kms=np.zeros(2), dv_rel=np.full(2, 60.0),
                              sigma_ft=np.full(2, 0.1))
