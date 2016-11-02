import unittest
import numpy as np
import floq.fixed_system as fs
import floq.errors as er
import rabi
import assertions

class TestFixedSystemInit(unittest.TestCase,assertions.CustomAssertions):

    def setUp(self):
        self.hf = np.zeros([5,10,10])
        self.dhf = np.zeros([3,10,10])
        self.omega = 3.0
        self.t = 1.0
        self.nz = 9
        self.problem = fs.FixedSystem(self.hf,self.dhf,self.nz,self.omega,self.t)


    def test_set_hf(self):
        self.assertArrayEqual(self.problem.hf,self.hf)

    def test_set_dhf(self):
        self.assertArrayEqual(self.problem.dhf,self.dhf)


    def test_set_nz(self):
        self.assertEqual(self.problem.nz,self.nz)

    def test_raise_error_if_nz_even(self):
        with self.assertRaises(er.UsageError):
            fs.FixedSystem(self.hf,self.dhf,10,self.omega,self.t)

    def test_set_k_dim(self):
        self.assertEqual(self.problem.k_dim,9*10)

    def test_set_nz_max(self):
        self.assertEqual(self.problem.nz_max,4)

    def test_set_nz_min(self):
        self.assertEqual(self.problem.nz_min,-4)


    def test_set_omega(self):
        self.assertEqual(self.problem.omega,self.omega)

    def test_set_t(self):
        self.assertEqual(self.problem.t,self.t)


    def test_set_dim(self):
        self.assertEqual(self.problem.dim,10)

    def test_set_nc(self):
        self.assertEqual(self.problem.nc,5)

    def test_set_np(self):
        self.assertEqual(self.problem.np,3)