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


class TestNormalIndexToFourierIndex(unittest.TestCase):

    def test_start(self):
        self.assertEqual(h.i_to_n(0, 81), -40)

    def test_end(self):
        self.assertEqual(h.i_to_n(80, 81), 40)

    def test_middle(self):
        self.assertEqual(h.i_to_n(40, 81), 0)

    def test_in_between(self):
        self.assertEqual(h.i_to_n(37, 81), -3)
