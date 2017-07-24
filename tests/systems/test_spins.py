from tests.assertions import CustomAssertions
import numpy as np
import floq.systems.spins as spins


class TestRandomisedSpinEnsemble(CustomAssertions):
    def test_init(self):
        # Not a very sophisticated test -- at least verify that it's not broken
        rand = spins.RandomisedSpinEnsemble(10, 10, 1.0, 8.2, 0.25)
        self.assertIsInstance(rand.systems, list)


class TestSpinEnsemble(CustomAssertions):

    def setUp(self):
        self.amps = np.array([1.2, 1.1, 0.7, 0.6])
        self.freqs = np.array([0.8, 1.1, 0.9, 1.2])
        self.ensemble = spins.SpinEnsemble(4, 2, 1.0, self.freqs, self.amps)

        self.controls = np.array([1.5, 1.3, 1.4, 1.1])
        self.t = 3.0


    def test_systems_works(self):
        self.assertIsInstance(self.ensemble.systems, list)


    def test_single_system_evolves_correctly(self):
        system = self.ensemble.systems[0]
        result = system.u(self.controls, self.t)

        single = spins.SpinSystem(2, self.amps[0], self.freqs[0], 1.0)
        target = single.u(self.controls, self.t)

        self.assertArrayEqual(result, target, decimals=10)
