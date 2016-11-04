import unittest
import numpy as np
import floq.parametric_system as ps
import floq.errors as er
import rabi
import assertions


class TestParametricSystemBase(unittest.TestCase):
    def test_get_system_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            p = ps.ParametericSystemBase()
            p.get_system(None, None)
