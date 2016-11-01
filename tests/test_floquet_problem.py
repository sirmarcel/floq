import unittest
import numpy as np
import floq.floquet_problem as fp
import rabi
import assertions

class TestFloquetProblemInit(unittest.TestCase):

    def setUp(self):
        self.hf = np.zeros([5,10,10])
        self.dhf = np.zeros([3,10,10])
        self.omega = 1.0
        self.t = 1.0
        self.nz = 10
        self.problem = fp.FloquetProblem(self.hf,self.dhf,self.nz,self.omega,self.t)

    def test_set_dim_correctly(self):
        self.assertEqual(self.problem.parameters.dim,10)

    def test_set_nc_correctly(self):
        self.assertEqual(self.problem.parameters.nc,5)

    def test_set_np_correctly(self):
        self.assertEqual(self.problem.parameters.np,3)

class TestFloquetProblemEvolution(unittest.TestCase,assertions.CustomAssertions):

    def setUp(self):
        g = 0.7
        e1 = 1.3
        e2 = 4.5
        hf = rabi.hf(g,e1,e2)
        dhf = np.array([rabi.hf(1.0,0.0,0.0)])
        
        nz = 11
        omega = 5.0
        t = 1.5
        self.p = fp.FloquetProblem(hf,dhf,nz,omega,t)

        self.target_u = rabi.u(g,e1,e2,omega,t)
        
    def test_calculate_u(self):
        self.p.calculate_u()
        u = self.p._u
        self.assertArrayEqual(u,self.target_u)
