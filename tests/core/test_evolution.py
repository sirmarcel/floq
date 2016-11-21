import unittest
import numpy as np
import tests.assertions as assertions
import tests.rabi as rabi
import floq.core.evolution as ev
import floq.helpers as h
import floq.fixed_system as fs
import floq.errors as er


def generate_fake_spectrum(unique_vals, dim, omega, nz):
    vals = np.array([])
    for i in xrange(0, nz):
        offset = h.i_to_n(i, nz)
        new = unique_vals + offset*omega*np.ones(dim)
        vals = np.append(vals, new)
    return vals


class TestBuildK(unittest.TestCase, assertions.CustomAssertions):
    def setUp(self):
        dim = 2
        self.p = fs.FixedSystemParameters.optional(dim, nz=5, nc=3, omega=1)

        a = -1.*np.ones([dim, dim])
        b = np.zeros([dim, dim])
        c = np.ones([dim, dim])
        z = np.zeros([dim, dim])
        i = np.identity(dim)

        self.goalk = np.array(
            np.bmat(
                [[b-2*i, a, z, z, z],
                 [c, b-i, a, z, z],
                 [z, c, b, a, z],
                 [z, z, c, b+i, a],
                 [z, z, z, c, b+2*i]]))
        self.hf = np.array([a, b, c])

    def test_build(self):
        builtk = ev.build_k(self.hf, self.p)
        self.assertArrayEqual(builtk, self.goalk)


class TestBuilddK(unittest.TestCase, assertions.CustomAssertions):
    def setUp(self):
        dim = 2
        self.p = fs.FixedSystemParameters.optional(dim, 5, 3, np=2, omega=1)

        a = -1.*np.ones([dim, dim])
        b = np.zeros([dim, dim])
        c = np.ones([dim, dim])
        z = np.zeros([dim, dim])
        i = np.identity(dim)

        dk1 = np.array(
            np.bmat(
                [[b, a, z, z, z],
                 [c, b, a, z, z],
                 [z, c, b, a, z],
                 [z, z, c, b, a],
                 [z, z, z, c, b]]))
        dk2 = np.array(
            np.bmat(
                [[b, b, z, z, z],
                 [a, b, b, z, z],
                 [z, a, b, b, z],
                 [z, z, a, b, b],
                 [z, z, z, a, b]]))

        self.goaldk = np.array([dk1, dk2])
        self.dhf = np.array([[a, b, c], [b, b, a]])

    def test_build(self):
        builtdk = ev.build_dk(self.dhf, self.p)
        self.assertArrayEqual(builtdk, self.goaldk)



class TestFindEigensystem(unittest.TestCase, assertions.CustomAssertions):
    def setUp(self):
        self.target_vals = np.array([-0.235, 0.753])
        # random matrix with known eigenvalues:
        # {-1.735, -0.747, -0.235, 0.753, 1.265, 2.253}
        k = np.array([[-0.0846814, -0.0015136 - 0.33735j, -0.210771 + 0.372223j,
                       0.488512 - 0.769537j, -0.406266 + 0.315634j, -0.334452 +
                        0.251584j], [-0.0015136 + 0.33735j,
                       0.809781, -0.416533 - 0.432041j, -0.571074 -
                        0.669052j, -0.665971 + 0.387569j, -0.297409 -
                        0.0028969j], [-0.210771 - 0.372223j, -0.416533 +
                        0.432041j, -0.0085791, 0.110085 + 0.255156j,
                       0.958938 - 0.17233j, -0.91924 + 0.126004j], [0.488512 +
                        0.769537j, -0.571074 + 0.669052j,
                       0.110085 - 0.255156j, -0.371663,
                       0.279778 + 0.477653j, -0.496302 + 1.04898j], [-0.406266 -
                        0.315634j, -0.665971 - 0.387569j, 0.958938 + 0.17233j,
                       0.279778 - 0.477653j, -0.731623,
                       0.525248 + 0.0443422j], [-0.334452 - 0.251584j, -0.297409 +
                        0.0028969j, -0.91924 - 0.126004j, -0.496302 - 1.04898j,
                       0.525248 - 0.0443422j, 1.94077]], dtype='complex128')

        e1 = np.array([[0.0321771 - 0.52299j, 0.336377 + 0.258732j],
                       [0.371002 + 0.0071587j, 0.237385 + 0.205185j],
                       [0.525321 + 0.j, 0.0964822 + 0.154715j]])
        e2 = np.array([[0.593829 + 0.j, -0.105998 - 0.394563j],
                       [-0.0737891 - 0.419478j, 0.323414 + 0.350387j],
                       [-0.05506 - 0.169033j, -0.0165495 + 0.199498j]])
        self.target_vecs = np.array([e1, e2])

        omega = 2.1
        nz = 3
        dim = 2
        p = fs.FixedSystemParameters.optional(dim, nz, omega=omega, decimals=3, sparse=False)

        self.vals, self.vecs = ev.find_eigensystem(k, p)

    def test_finds_vals(self):
        self.assertArrayEqual(self.vals, self.target_vals)

    def test_finds_vecs(self):
        self.assertArrayEqual(self.vecs, self.target_vecs, decimals=3)

    def test_casts_as_complex128(self):
        self.assertEqual(self.vecs.dtype, 'complex128')


class TestFindUniquevals(unittest.TestCase, assertions.CustomAssertions):
    
    def test_finds_unique_vals_if_all_positive(self):
        dim = 3
        omega = 1.5
        nz = 11
        p = fs.FixedSystemParameters.optional(dim, nz, omega=omega)

        us = np.array([0.3134, 0.587, 0.6324])
        e = generate_fake_spectrum(us, dim, omega, nz)

        unique_e = ev.find_unique_vals(e, p)
        self.assertArrayEqual(unique_e, us)

    def test_finds_unique_vals_if_not_all_positive(self):
        dim = 3
        omega = 2.0
        nz = 11
        p = fs.FixedSystemParameters.optional(dim, nz, omega=omega)

        us = np.array([-0.3, 0.544, 0.6])
        e = generate_fake_spectrum(us, dim, omega, 11)

        unique_e = ev.find_unique_vals(e, p)
        self.assertArrayEqual(unique_e, us)

    def test_raises_error_if_degenerate(self):
        dim = 3
        omega = 2.0
        nz = 11
        p = fs.FixedSystemParameters.optional(dim, nz, omega=omega)
        us = np.array([0.3552, 0.3552, 0.6])
        e = generate_fake_spectrum(us, dim, omega, 11)
        with self.assertRaises(er.EigenvalueNumberError):
            ev.find_unique_vals(e, p)


class TestSeparateComponents(unittest.TestCase, assertions.CustomAssertions):
    def test_split(self):
        a = np.array([1.23, 2.45])
        b = np.array([6.123, 1.656])
        c = np.array([2.323, 3.112])

        e1 = np.concatenate((a, b, c))
        e1_split = np.array([a, b, c])
        e2 = np.concatenate((c, a, b))
        e2_split = np.array([c, a, b])

        target = np.array([e1_split, e2_split])

        split = ev.separate_components([e1, e2], 3)

        self.assertArrayEqual(split, target)



class TestCalculatePhi(unittest.TestCase, assertions.CustomAssertions):
    def test_sum(self):
        a = np.array([1.53, 2.45])
        b = np.array([7.161, 1.656])
        c = np.array([2.3663, 8.112])

        e1 = np.array([a, a, c])
        e1_sum = a+a+c
        e2 = np.array([c, a, b])
        e2_sum = c+a+b

        target = np.array([e1_sum, e2_sum])
        calculated_sum = ev.calculate_phi([e1, e2])

        self.assertArrayEqual(calculated_sum, target)


class TestCalculatePsi(unittest.TestCase, assertions.CustomAssertions):
    def test_sum(self):
        omega = 2.34
        t = 1.22
        p = fs.FixedSystemParameters.optional(2, 3, omega=omega, t=t)

        a = np.array([1.53, 2.45], dtype='complex128')
        b = np.array([7.161, 1.656], dtype='complex128')
        c = np.array([2.3663, 8.112], dtype='complex128')

        e1 = np.array([a, a, c])
        e1_sum = np.exp(-1j*omega*t)*a+a+np.exp(1j*omega*t)*c
        e2 = np.array([c, a, b])
        e2_sum = np.exp(-1j*omega*t)*c+a+np.exp(1j*omega*t)*b

        target = np.array([e1_sum, e2_sum])
        vecs = np.array([e1, e2])

        calculated_sum = ev.calculate_psi(vecs, p)

        self.assertArrayEqual(calculated_sum, target)


class TestCalculateU(unittest.TestCase, assertions.CustomAssertions):
    def test_u(self):
        omega = 3.56
        t = 8.123
        dim = 2
        nz = 3
        p = fs.FixedSystemParameters.optional(dim, nz, omega=omega, t=t)

        energies = [0.23, 0.42]

        e1 = np.array([1.563 + 1.893j, 1.83 + 1.142j, 0.552 + 0.997j,
                       0.766 + 1.162j, 1.756 + 0.372j, 0.689 + 0.902j])
        e2 = np.array([1.328 + 1.94j, 1.866 + 0.055j, 1.133 + 0.162j,
                       1.869 + 1.342j, 1.926 + 1.587j, 1.735 + 0.942j])
        vecs = ev.separate_components(np.array([e1, e2]), nz)

        phi = ev.calculate_phi(vecs)
        psi = ev.calculate_psi(vecs, p)

        target = np.array([[29.992 + 14.079j, 29.125 + 18.169j],
                           [5.117 - 1.363j, 5.992 - 2.462j]]).round(3)
        u = ev.calculate_u(phi, psi, energies, p).round(3)

        self.assertArrayEqual(u, target)



class TestCalculateDU(unittest.TestCase, assertions.CustomAssertions):
    def setUp(self):
        g = 0.5
        e1 = 1.2
        e2 = 2.8
        hf = rabi.hf(g, e1, e2)
        dhf = np.array([rabi.hf(1.0, 0, 0)])
 
        nz = 21
        dim = 2
        omega = 5.0
        t = 1.5
        p = fs.FixedSystemParameters.optional(dim, nz, nc=3, omega=omega, t=t, np=1)

        k = ev.build_k(hf, p)
        vals, vecs = ev.find_eigensystem(k, p)
        psi = ev.calculate_psi(vecs, p)


        self.du = np.array([[-0.43745 + 0.180865j, 0.092544 - 0.0993391j],
                            [-0.0611011 - 0.121241j, -0.36949 - 0.295891j]])
        dk = ev.build_dk(dhf, p)
        self.ducal = ev.calculate_du(dk, psi, vals, vecs, p)

    def test_is_correct_du(self):
        self.assertArrayEqual(self.ducal, self.du)
