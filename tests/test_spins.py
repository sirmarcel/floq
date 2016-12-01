import unittest
import numpy as np
import floq.systems.spins as spins
import floq.core.fixed_system as fs
import assertions
import floq.evolution as ev


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


class TestSpinEnsemble(unittest.TestCase, assertions.CustomAssertions):

    def test_build_single_hf(self):
        controls = np.array([1.2, 2.3, 3.4, 5.4])
        amp = 1.1
        freq = 2.5
        target = single_hf(amp*controls, freq)

        ss = spins.SpinEnsemble(1, 2, 1, 1, 1)
        result = ss._build_single_hf(freq, amp, controls)

        self.assertArrayEqual(target, result)

    def test_build_single_dhf(self):
        amp = 1.25
        target = dhf(amp)

        ss = spins.SpinEnsemble(1, 2, 1, 1, 1)
        result = ss._build_single_dhf(amp)

        self.assertArrayEqual(target, result)

    def test_dhf(self):
        amps = np.array([1.25, 0.9, 1.8])
        target = np.array([dhf(amp) for amp in amps])

        ss = spins.SpinEnsemble(3, 2, 1, 1, amps)
        result = ss._build_dhf()

        self.assertArrayEqual(target, result)

    def test_get_single_system(self):
        amps = np.array([1.25, 0.9, 1.8])
        freqs = np.array([1.7, 2.0, 2.1])
        controls = np.array([1.2, 1.3, 2.6, 3.9])
        hf1 = single_hf(amps[0]*controls, freqs[0])
        dhf1 = dhf(amps[0])
        first_system = fs.FixedSystem(hf1, dhf1, 3, 1.0, 1.0)

        ss = spins.SpinEnsemble(3, 2, 1.0, freqs, amps)
        system = ss.get_single_system(0, controls, 1.0)

        self.assertEqual(first_system, system)


class TestSpinEnsembleEvolution(unittest.TestCase, assertions.CustomAssertions):
    def setUp(self):
        amps = np.array([1.2, 1.1, 0.7, 0.6])
        freqs = np.array([0.8, 1.1, 0.9, 1.2])
        self.ensemble = spins.SpinEnsemble(4, 2, 1.0, freqs, amps)

        self.controls = np.array([1.5, 1.3, 1.4, 1.1])
        self.t = 3.0
        self.ensemble.set_nz(self.controls, self.t)

        self.us = np.zeros([4, 2, 2], dtype=np.complex128)
        for i in xrange(4):
            sys = self.ensemble.get_single_system(i, self.controls, self.t)
            self.us[i], nz = ev.evolve_system(sys)

    def test_get_us(self):
        us = self.ensemble.get_us(self.controls, self.t)
        self.assertArrayEqual(us, self.us)
       