import unittest
import numpy as np
import floq.helpers as h


class TestFourierIndexToNormalIndex(unittest.TestCase):

    def test_start(self):
        self.assertEqual(h.n_to_i(-40, 81), 0)

    def test_end(self):
        self.assertEqual(h.n_to_i(40, 81), 80)

    def test_middle(self):
        self.assertEqual(h.n_to_i(0, 81), 40)

    def test_in_between(self):
        self.assertEqual(h.n_to_i(-3, 81), 37)

    def test_too_big_a_bit(self):
        self.assertEqual(h.n_to_i(5, 7), h.n_to_i(-2, 7))

    def test_too_big_a_lot(self):
        self.assertEqual(h.n_to_i(5+7, 7), h.n_to_i(-2, 7))

    def test_too_small_a_bit(self):
        self.assertEqual(h.n_to_i(-6, 7), h.n_to_i(1, 7))

    def test_too_small_a_lot(self):
        self.assertEqual(h.n_to_i(-6-14, 7), h.n_to_i(1, 7))


class TestNormalIndexToFourierIndex(unittest.TestCase):

    def test_start(self):
        self.assertEqual(h.i_to_n(0, 81), -40)

    def test_end(self):
        self.assertEqual(h.i_to_n(80, 81), 40)

    def test_middle(self):
        self.assertEqual(h.i_to_n(40, 81), 0)

    def test_in_between(self):
        self.assertEqual(h.i_to_n(37, 81), -3)


class TestIsUnitary(unittest.TestCase):
    def test_true_if_unitary(self):
        u = np.array([[-0.288822 - 0.154483j, 0.20768 - 0.22441j, 0.0949032 - 0.0560178j, -0.385994 + 0.210021j, 0.423002 - 0.605778j, 0.135684 - 0.172261j], [0.0998628 - 0.364186j, 0.408817 - 0.35846j, -0.224508 - 0.550201j, 0.258427 + 0.263299j, -0.0297947 + 0.180679j, -0.0134853 + 0.197029j], [0.541087 - 0.216046j, -0.306777 + 0.0439077j, -0.479354 + 0.0395382j, -0.474755 + 0.264776j, -0.0971467 - 0.0167121j, 0.121192 - 0.115168j], [-0.0479833 - 0.133938j, 0.0696875 - 0.539678j, 0.314762 + 0.391157j, -0.376453 + 0.00569747j, -0.348676 + 0.2061j, 0.0588683 + 0.34972j], [-0.524482 + 0.213402j, 0.152127 + 0.111274j, -0.308402 - 0.134059j, -0.448647 + 0.120202j, -0.0680734 + 0.435883j, -0.295969 - 0.181141j], [-0.119405 + 0.235674j, 0.349453 + 0.247169j, -0.169971 + 0.0966179j, 0.0310919 + 0.129778j, -0.228356 + 0.00511762j, 0.793243 + 0.0977203j]])
        self.assertTrue(h.is_unitary(u, 1e-5))


    def test_true_if_not_unitary(self):
        u = np.array([[-5.288822 - 0.154483j, 0.20768 - 0.22441j, 0.0949032 - 0.0560178j, -0.385994 + 0.210021j, 0.423002 - 0.605778j, 0.135684 - 0.172261j], [0.0998628 - 0.364186j, 0.408817 - 0.35846j, -0.224508 - 0.550201j, 0.258427 + 0.263299j, -0.0297947 + 0.180679j, -0.0134853 + 0.197029j], [0.541087 - 0.216046j, -0.306777 + 0.0439077j, -0.479354 + 0.0395382j, -0.474755 + 0.264776j, -0.0971467 - 0.0167121j, 0.121192 - 0.115168j], [-0.0479833 - 0.133938j, 0.0696875 - 0.539678j, 0.314762 + 0.391157j, -0.376453 + 0.00569747j, -0.348676 + 0.2061j, 0.0588683 + 0.34972j], [-0.524482 + 0.213402j, 0.152127 + 0.111274j, -0.308402 - 0.134059j, -0.448647 + 0.120202j, -0.0680734 + 0.435883j, -0.295969 - 0.181141j], [-0.119405 + 0.235674j, 0.349453 + 0.247169j, -0.169971 + 0.0966179j, 0.0310919 + 0.129778j, -0.228356 + 0.00511762j, 0.793243 + 0.0977203j]])
        self.assertFalse(h.is_unitary(u))


class TestMakeEven(unittest.TestCase):
    def test_make_odd_even(self):
        self.assertEqual(h.make_even(3), 4)

    def test_leave_even_even(self):
        self.assertEqual(h.make_even(4), 4)
