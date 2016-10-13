import unittest
import numpy as np
import floq.evolution as ev
import floq.helpers as h

def rabi_hf(n_comp,w,g,e1,e2):
    hf = np.zeros([n_comp,2,2])
    hf[h.num_to_i(-1,nfreq)] = np.array([[0,0],[g,0]])
    hf[h.num_to_i(0,nfreq)] = np.array([[e1,0],[0,e2]])
    hf[h.num_to_i(1,nfreq)] = np.array([[0,g],[0,0]])

def generate_fake_spectrum(unique_evas,dim,omega,n_zones):
    evas = np.array([])
    for i in xrange(0,n_zones):
        offset = h.i_to_num(i,n_zones)
        new = unique_evas + offset*omega*np.ones(dim)
        evas = np.append(evas,new)
    return evas

class TestFindEigensystem(unittest.TestCase):
    def setUp(self):
        n_zones = 21
        dim = 3
        omega = 2.0
        self.target_eves = np.array([0.123,0.1823,0.881])
        spectrum = generate_fake_spectrum(self.target_eves,dim,omega,n_zones)
        k = np.diag(spectrum)
        eve1 = np.zeros(dim*n_zones)
        eve2 = np.zeros(dim*n_zones)
        eve3 = np.zeros(dim*n_zones)
        eve1[h.num_to_i(0,n_zones)] = 1.0
        eve2[h.num_to_i(1,n_zones)] = 1.0
        eve3[h.num_to_i(2,n_zones)] = 1.0
        self.target_eves = [eve1,eve2,eve3]
        self.evas,self.eves = ev.find_eigensystem(k,dim,omega)

    def test_finds_evas(self):
        self.assertTrue(np.array_equal(self.evas,self.target_eves))

    def test_finds_eves(self):
        print self.eves[0]
        print self.target_eves[0]
        
        self.assertTrue(np.array_equal(self.eves,self.target_eves))


class TestBuildK(unittest.TestCase):
    def setUp(self):
        dim = 2
        a = -1.*np.ones([dim,dim])
        b = np.zeros([dim,dim])
        c = np.ones([dim,dim])
        z = np.zeros([dim,dim])

        i = np.identity(dim)

        self.goalk = np.array(np.bmat([[b-2*i,a,z,z,z],[c,b-i,a,z,z],[z,c,b,a,z],[z,z,c,b+i,a],[z,z,z,c,b+2*i]]))
        self.hf = np.array([a,b,c])

    def test_build(self):
        builtk = ev.build_k(self.hf,5,1)
        self.assertTrue(np.array_equal(builtk,self.goalk))

class TestFindUniqueEvas(unittest.TestCase):
    
    def test_finds_unique_evas_if_all_positive(self):
        dim = 3
        omega = 1.5
        us = np.array([0.3134,0.587,0.6324])
        e = generate_fake_spectrum(us,dim,omega,11)
        unique_e = ev.find_unique_evas(e,dim,omega)
        print unique_e
        self.assertTrue(np.array_equal(unique_e,us))

    def test_finds_unique_evas_if_not_all_positive(self):
        dim = 3
        omega = 2.0
        us = np.array([-0.3,0.544,0.6])
        e = generate_fake_spectrum(us,dim,omega,11)
        unique_e = ev.find_unique_evas(e,dim,omega)
        self.assertTrue(np.array_equal(unique_e,us))

    def test_raises_error_if_degenerate(self):
       dim = 3
       omega = 2.0
       us = np.array([0.3552,0.3552,0.6])
       e = generate_fake_spectrum(us,dim,omega,11)
       with self.assertRaises(ev.FloquetError):
        ev.find_unique_evas(e,dim,omega)
