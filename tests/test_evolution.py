import unittest
import numpy as np
import assertions
import rabi
import floq.evolution as ev
import floq.helpers as h
import floq.fixed_system as fs
import floq.errors as er


class TestEvolveSystem(unittest.TestCase, assertions.CustomAssertions):
    def setUp(self):
        g = 0.5
        e1 = 1.2
        e2 = 2.8
        hf = rabi.hf(g, e1, e2)
        dhf = np.array([rabi.hf(1.0, 0, 0)])

        nz = 3
        dim = 2
        omega = 5.0
        t = 20.5
        s = fs.FixedSystem(hf, dhf, nz, omega, t)

        self.u = rabi.u(g, e1, e2, omega, t)
        self.ucal = ev.evolve_system(s)

        self.um = np.matrix(self.ucal)

    def test_gives_unitary(self):
        uu = self.um*self.um.getH()
        identity = np.identity(2)
        self.assertArrayEqual(uu, identity, 8)

    def test_is_correct_u(self):
        self.assertArrayEqual(self.u, self.ucal, 8)


class TestEvolveSystemWithDerivs(unittest.TestCase, assertions.CustomAssertions):
    def setUp(self):
        g = 0.5
        e1 = 1.2
        e2 = 2.8
        hf = rabi.hf(g, e1, e2)
        dhf = np.array([rabi.hf(1.0, 0, 0)])

        nz = 3
        dim = 2
        omega = 5.0
        t = 1.5
        s = fs.FixedSystem(hf, dhf, nz, omega, t)

        self.du = np.array([[-0.43745 + 0.180865j, 0.092544 - 0.0993391j],
                            [-0.0611011 - 0.121241j, -0.36949-0.295891j]])
        [self.ucal, self.ducal, self.nz] = ev.evolve_system_with_derivatives(s)

    def test_is_correct_du(self):
        self.assertArrayEqual(self.ducal, self.du)


class TestTestNz(unittest.TestCase):
    def test_false_if_nz_too_small(self):
        g = 0.5
        e1 = 1.2
        e2 = 2.8
        hf = rabi.hf(g, e1, e2)
        dhf = np.array([rabi.hf(1.0, 0, 0)])

        nz = 3
        dim = 2
        omega = 5.0
        t = 1.5
        s = fs.FixedSystem(hf, dhf, nz, omega, t)
        self.assertFalse(ev.test_nz(s)[0])


    def test_true_if_nz_okay(self):
        g = 0.5
        e1 = 1.2
        e2 = 2.8
        hf = rabi.hf(g, e1, e2)
        dhf = np.array([rabi.hf(1.0, 0, 0)])

        nz = 21
        dim = 2
        omega = 5.0
        t = 1.5
        s = fs.FixedSystem(hf, dhf, nz, omega, t)
        self.assertTrue(ev.test_nz(s)[0])
