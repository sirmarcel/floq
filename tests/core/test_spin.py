from unittest import TestCase
from tests.assertions import CustomAssertions
import numpy as np
import floq.core.spin as spin


def single_hf(controls, omega):
    a1 = controls[0]
    b1 = controls[1]
    a2 = controls[2]
    b2 = controls[3]
    return np.array([[[0, 0.25*(1j*a2 + b2)],
                      [0.25*1j*(a2 + 1j*b2), 0]],
                     [[0, 0.25*(1j*a1 + b1)],
                      [0.25*1j*(a1 + 1j*b1), 0]],
                     [[omega/2.0, 0],
                      [0, -(omega/2.0)]],
                     [[0, -0.25j*(a1 - 1j*b1)],
                      [0.25*(-1j*a1 + b1), 0]],
                     [[0, -0.25j*(a2 - 1j*b2)],
                     [0.25*(-1j*a2 + b2), 0]]])


def dhf(amp):
    dhf_b1 = np.array([[[0., 0.], [0., 0.]],
                       [[0., 0.25], [-0.25, 0.]],
                       [[0., 0.], [0., 0.]],
                       [[0., -0.25], [0.25, 0.]],
                       [[0., 0.], [0., 0.]]])

    dhf_a1 = np.array([[[0., 0.], [0., 0.]],
                       [[0., 0. + 0.25j], [0. + 0.25j, 0.]],
                       [[0., 0.], [0., 0.]],
                       [[0., 0. - 0.25j], [0. - 0.25j, 0.]],
                       [[0., 0.], [0., 0.]]])

    dhf_b2 = np.array([[[0., 0.25], [-0.25, 0.]],
                       [[0., 0.], [0., 0.]],
                       [[0., 0.], [0., 0.]],
                       [[0., 0.], [0., 0.]],
                       [[0., -0.25], [0.25, 0.]]])

    dhf_a2 = np.array([[[0., 0. + 0.25j], [0. + 0.25j, 0.]],
                       [[0., 0.], [0., 0.]],
                       [[0., 0.], [0., 0.]],
                       [[0., 0.], [0., 0.]],
                       [[0., 0. - 0.25j], [0. - 0.25j, 0.]]])

    return amp*np.array([dhf_a1, dhf_b1, dhf_a2, dhf_b2])


class TestSpinHf(TestCase, CustomAssertions):

    def test_build_single_hf(self):
        controls = np.array([1.2, 2.3, 3.4, 5.4])
        freq = 2.5

        target = single_hf(controls, freq)
        result = spin.hf(2, freq, controls)

        self.assertArrayEqual(target, result)


class TestSpindHf(TestCase, CustomAssertions):
    def test_build_single_dhf(self):
        amp = 1.25

        target = dhf(amp)
        result = spin.dhf(2, amp)

        self.assertArrayEqual(target, result)
