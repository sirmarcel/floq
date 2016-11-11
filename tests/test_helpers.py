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
