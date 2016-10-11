import unittest
import numpy as np
import floq.helpers as h

class TestNumToIndex(unittest.TestCase):
  
  def test_start(self):
    self.assertEqual(h.num_to_index(-40,81),0)

  def test_end(self):
    self.assertEqual(h.num_to_index(40,81),80)

  def test_middle(self):
    self.assertEqual(h.num_to_index(0,81),40)

  def test_in_between(self):
    self.assertEqual(h.num_to_index(-3,81),37)    

class TestIndexToNum(unittest.TestCase):

  def test_start(self):
    self.assertEqual(h.index_to_num(0,81),-40)

  def test_end(self):
    self.assertEqual(h.index_to_num(80,81),40)

  def test_middle(self):
    self.assertEqual(h.index_to_num(40,81),0)

  def test_in_between(self):
    self.assertEqual(h.index_to_num(37,81),-3)    