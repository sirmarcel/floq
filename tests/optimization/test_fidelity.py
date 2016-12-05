from unittest import TestCase
from tests.assertions import CustomAssertions
import floq.optimization.fidelity as fid
from floq.systems.spins import SpinEnsemble
import numpy as np


class TestEnsembleFidelity(TestCase, CustomAssertions):
    def setUp(self):
        self.ensemble = SpinEnsemble(2, 2, 1.5, np.array([1.1, 1.1]), np.array([1, 1]))


    def test_correct_in_one_case(self):
        target = np.array([[0.105818 - 0.324164j, -0.601164 - 0.722718j],
                           [0.601164 - 0.722718j, 0.105818 + 0.324164j]])

        f = fid.EnsembleFidelity(self.ensemble, fid.OperatorDistance, t=1.0, target=target)

        self.assertTrue(np.isclose(f.f(np.array([1.5, 1.5, 1.5, 1.5])), 0.0, atol=1e-6))
