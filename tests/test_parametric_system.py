import unittest
import numpy as np
import floq.systems.parametric_system as ps
import floq.core.fixed_system
import floq.errors as er
import rabi
import assertions
from mock import MagicMock, patch


class TestParametricSystemBase(unittest.TestCase):
    def test_hf_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            p = ps.ParametericSystemBase()
            p._hf(None)

    def test_dhf_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            p = ps.ParametericSystemBase()
            p._dhf(None)


class TestParametricSystemBaseCaching(unittest.TestCase):
    def setUp(self):
        self.ctrls1 = np.array([1.2, 1.1])
        self.ctrls2 = np.array([1.3, 1.1])

        self.real = ps.ParametericSystemBase()
        self.real._hf = MagicMock()
        self.real._dhf = MagicMock()
        self.real._fixed_system = MagicMock()

        self.real._last_controls = None
        self.real._last_t = None

        self.real.nz = MagicMock()
        self.real.omega = MagicMock()

    def test_u_caches_if_same(self):
        with patch('floq.core.fixed_system.FixedSystem') as mock:
            self.real.u(self.ctrls1, 1.0)
            self.real.u(self.ctrls1, 1.0)
            mock.assert_called_once()

    def test_u_does_not_cache_if_not_same(self):
        with patch('floq.core.fixed_system.FixedSystem') as mock:
            self.real.u(self.ctrls1, 1.0)
            self.real.u(self.ctrls2, 1.0)
            self.assertEqual(mock.call_count, 2)