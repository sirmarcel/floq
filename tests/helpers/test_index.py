from unittest import TestCase
import numpy as np
import floq.helpers.index as h


class TestFourierIndexToNormalIndex(TestCase):

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


class TestNormalIndexToFourierIndex(TestCase):

    def test_start(self):
        self.assertEqual(h.i_to_n(0, 81), -40)

    def test_end(self):
        self.assertEqual(h.i_to_n(80, 81), 40)

    def test_middle(self):
        self.assertEqual(h.i_to_n(40, 81), 0)

    def test_in_between(self):
        self.assertEqual(h.i_to_n(37, 81), -3)


class TestMakeEven(TestCase):
    def test_make_odd_even(self):
        self.assertEqual(h.make_even(3), 4)

    def test_leave_even_even(self):
        self.assertEqual(h.make_even(4), 4)
