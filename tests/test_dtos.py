import unittest
import numpy as np
import floq.dtos

class TestFloquetProblem(unittest.TestCase):
  def setUp(self):
    self.dim = 2
    self.nw = 80
    self.w = 1
    self.hf = np.random.rand(self.nw,self.dim,self.dim)
    self.dto = floq.dtos.FloquetProblem(self.hf,self.w)

  def test_hf_must_be_array(self):
    with self.assertRaises(TypeError):
      floq.dtos.FloquetProblem('a'),self.hf

  def test_sets_nw(self):
    self.assertEqual(self.dto.nw,self.nw)

  def test_sets_dim(self):
    self.assertEqual(self.dto.dim,self.dim)