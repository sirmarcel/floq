import unittest
import numpy as np
import assertions
import floq.evolution as ev
import floq.helpers as h

def rabi_hf(g,e1,e2):
    hf = np.zeros([3,2,2])
    hf[0] = np.array([[0,0],[g,0]])
    hf[1] = np.array([[e1,0],[0,e2]])
    hf[2] = np.array([[0,g],[0,0]])
    return hf

def generate_fake_spectrum(unique_evas,dim,omega,n_zones):
    evas = np.array([])
    for i in xrange(0,n_zones):
        offset = h.i_to_num(i,n_zones)
        new = unique_evas + offset*omega*np.ones(dim)
        evas = np.append(evas,new)
    return evas

class TestDoEvolution(unittest.TestCase):
    def setUp(self):
        g = 1.25
        e1 = 1.2
        e2 = 1.6
        hf = rabi_hf(g,e1,e2)
        n_zones = 81
        dim = 2
        omega = 1.5
        t = 8.0
        self.ucal = ev.do_evolution(hf,dim,n_zones,omega,t)

    def test_is_correct_u(self):
        self.assertTrue(1 == 0) 

class TestCalculateU(unittest.TestCase,assertions.CustomAssertions):
    def test_u(self):
        # Note: The "manual" calculation was done with matlab
        omega = 3.56
        t = 8.123
        energies = [0.23, 0.42]
        dim = 2
        n_zones = 3

        a = np.array([1.53+2.33j,2.45],dtype='complex128')
        b = np.array([7.161,1.656+1.2j],dtype='complex128')
        c = np.array([2.3663j,8.112],dtype='complex128')

        e1 = np.array([1.563 + 1.893j, 1.83 + 1.142j, 0.552 + 0.997j, 0.766 + 
 1.162j, 1.756 + 0.372j, 0.689 + 0.902j])
        e2 = np.array([1.328 + 1.94j, 1.866 + 0.055j, 1.133 + 0.162j, 1.869 + 
 1.342j, 1.926 + 1.587j, 1.735 + 0.942j])
        eves = ev.separate_components(np.array([e1,e2]),n_zones)

        phi = ev.calculate_phi(eves)
        psi = ev.calculate_psi(eves,dim,n_zones,omega,t)

        target = np.array([[18.0985 + 7.75776j, 17.6485 + 11.4563j],[4.32948 - 0.849366j, 7.34917 - 0.802564j]]).round(4)
        u = ev.calculate_u(phi,psi,energies,dim,n_zones,omega,t).round(4)
       
        self.assertArrayEqual(u,target)


class TestCalculatePsi(unittest.TestCase,assertions.CustomAssertions):
    def test_sum(self):
        omega = 2.34
        t = 1.22

        a = np.array([1.53,2.45],dtype='complex128')
        b = np.array([7.161,1.656],dtype='complex128')
        c = np.array([2.3663,8.112],dtype='complex128')

        e1 = np.array([a,a,c])
        e1_sum = np.exp(1j*omega*t)*a+a+np.exp(-1j*omega*t)*c
        e2 = np.array([c,a,b])
        e2_sum = np.exp(1j*omega*t)*c+a+np.exp(-1j*omega*t)*b

        target = [e1_sum,e2_sum]
        eves = np.array([e1,e2])

        calculated_sum = ev.calculate_psi(eves,2,3,omega,t)

        self.assertArrayEqual(calculated_sum,target)

class TestCalculatePhi(unittest.TestCase,assertions.CustomAssertions):
    def test_sum(self):
        a = np.array([1.53,2.45])
        b = np.array([7.161,1.656])
        c = np.array([2.3663,8.112])

        e1 = np.array([a,a,c])
        e1_sum = a+a+c
        e2 = np.array([c,a,b])
        e2_sum = c+a+b

        target = [e1_sum,e2_sum]
        calculated_sum = ev.calculate_phi([e1,e2])

        self.assertArrayEqual(calculated_sum,target)


class TestSeparateComponents(unittest.TestCase,assertions.CustomAssertions):
    def test_split(self):
        a = np.array([1.23,2.45])
        b = np.array([6.123,1.656])
        c = np.array([2.323,3.112])

        e1 = np.concatenate((a,b,c))
        e1_split = np.array([a,b,c])
        e2 = np.concatenate((c,a,b))
        e2_split = np.array([c,a,b])

        target = [e1_split,e2_split]

        split = ev.separate_components([e1,e2],3)

        self.assertArrayEqual(split,target)

class TestFindEigensystem(unittest.TestCase,assertions.CustomAssertions):
    def setUp(self):
        n_zones = 3
        dim = 2
        omega = 2.0
        self.target_evas = np.array([0.123,0.1823])
        spectrum = generate_fake_spectrum(self.target_evas,dim,omega,n_zones)
        k = np.diag(spectrum)
        eve1 = np.array([[0,0],[1.,0],[0,0]])
        eve2 = np.array([[0,0],[0,1.0],[0,0]])
        self.target_eves = [eve1,eve2]
        self.evas,self.eves = ev.find_eigensystem(k,dim,n_zones,omega)

    def test_finds_evas(self):
        self.assertArrayEqual(self.evas,self.target_evas)

    def test_finds_eves(self):
        self.assertArrayEqual(self.eves,self.target_eves)

    def test_casts_as_complex128(self):
        self.assertEqual(self.eves.dtype,'complex128')


class TestBuildK(unittest.TestCase,assertions.CustomAssertions):
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
        self.assertArrayEqual(builtk,self.goalk)

class TestFindUniqueEvas(unittest.TestCase,assertions.CustomAssertions):
    
    def test_finds_unique_evas_if_all_positive(self):
        dim = 3
        omega = 1.5
        us = np.array([0.3134,0.587,0.6324])
        e = generate_fake_spectrum(us,dim,omega,11)
        unique_e = ev.find_unique_evas(e,dim,omega)
        print unique_e
        self.assertArrayEqual(unique_e,us)

    def test_finds_unique_evas_if_not_all_positive(self):
        dim = 3
        omega = 2.0
        us = np.array([-0.3,0.544,0.6])
        e = generate_fake_spectrum(us,dim,omega,11)
        unique_e = ev.find_unique_evas(e,dim,omega)
        self.assertArrayEqual(unique_e,us)

    def test_raises_error_if_degenerate(self):
       dim = 3
       omega = 2.0
       us = np.array([0.3552,0.3552,0.6])
       e = generate_fake_spectrum(us,dim,omega,11)
       with self.assertRaises(ev.FloquetError):
        ev.find_unique_evas(e,dim,omega)
