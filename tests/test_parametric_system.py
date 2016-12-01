import unittest
import numpy as np
import floq.systems.parametric_system as ps
import floq.errors as er
import rabi
import assertions


class TestParametricSystemBase(unittest.TestCase):
    def test_hf_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            p = ps.ParametericSystemBase()
            p._hf(None)

    def test_dhf_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            p = ps.ParametericSystemBase()
            p._dhf(None)
