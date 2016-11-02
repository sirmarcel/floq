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

    def test_set_dim(self):
        self.assertEqual(self.problem.params.dim,10)

    def test_set_nc(self):
        self.assertEqual(self.problem.params.nc,5)

    def test_set_np(self):
        self.assertEqual(self.problem.params.np,3)

class TestFixedSystemParametersInit(unittest.TestCase):
    def setUp(self):
        self.dim = 2
        self.nz = 5
        self.nc = 2
        self.np = 3
        self.omega = 3.0
        self.t = 4.0
        self.decimals = 5
        
        self.p = fs.FixedSystemParameters(self.dim,self.nz,self.nc,self.np,self.omega,self.t,self.decimals)

    def test_raise_error_if_nz_even(self):
        with self.assertRaises(er.UsageError):
            fs.FixedSystemParameters(self.dim,self.nz*2,self.nc,self.np,self.omega,self.t,self.decimals)

    def test_set_k_dim(self):
        self.assertEqual(self.p.k_dim,self.dim*self.nz)

    def test_set_nz_max(self):
        self.assertEqual(self.p.nz_max,2)

    def test_set_nz_min(self):
        self.assertEqual(self.p.nz_min,-2)