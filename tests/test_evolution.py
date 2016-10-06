import unittest
import numpy as np
import floq.evolution as ev

def rabi_hf(dim,nfreq,w,g,e1,e2):
  hf = np.zeros([nfreq,dim,dim])
  hf[ev.eta_to_i(-1,nfreq)] = np.array([[0,0],[g,0]])
  hf[ev.eta_to_i(0,nfreq)] = np.array([[e1,0],[0,e2]])
  hf[ev.eta_to_i(1,nfreq)] = np.array([[0,g],[0,0]])

class TestEtaToI(unittest.TestCase):
  def setUp(self):
    pass

  def test_start(self):
    self.assertEqual(ev.eta_to_i(-40,81),0)

  def test_end(self):
    self.assertEqual(ev.eta_to_i(40,81),80)

  def test_middle(self):
    self.assertEqual(ev.eta_to_i(0,81),40)    
