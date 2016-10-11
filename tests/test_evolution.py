import unittest
import numpy as np
import floq.evolution as ev
import floq.helpers as h

def rabi_hf(n_comp,w,g,e1,e2):
    hf = np.zeros([n_comp,2,2])
    hf[h.num_to_i(-1,nfreq)] = np.array([[0,0],[g,0]])
    hf[h.num_to_i(0,nfreq)] = np.array([[e1,0],[0,e2]])
    hf[h.num_to_i(1,nfreq)] = np.array([[0,g],[0,0]])

class TestBuildK(unittest.TestCase):
    def setUp(self):
        self.dim = 2
        self.a = -1.*np.ones([self.dim,self.dim])
        self.b = np.zeros([self.dim,self.dim])
        self.c = np.ones([self.dim,self.dim])
        self.z = np.zeros([self.dim,self.dim])

        a = self.a
        b = self.b
        c = self.c
        z = self.z
        i = np.identity(self.dim)

        self.goalk = np.array(np.bmat([[b-2*i,a,z,z,z],[c,b-i,a,z,z],[z,c,b,a,z],[z,z,c,b+i,a],[z,z,z,c,b+2*i]]))
        self.hf = np.array([a,b,c])

    def test_build(self):
        builtk = ev.build_k(self.hf,5,1)
        self.assertTrue(np.array_equal(builtk,self.goalk))