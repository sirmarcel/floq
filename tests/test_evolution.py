import unittest
import numpy as np
import floq.evolution as ev
import floq.helpers as h

def rabi_hf(dim,nfreq,w,g,e1,e2):
  hf = np.zeros([nfreq,dim,dim])
  hf[ev.eta_to_i(-1,nfreq)] = np.array([[0,0],[g,0]])
  hf[ev.eta_to_i(0,nfreq)] = np.array([[e1,0],[0,e2]])
  hf[ev.eta_to_i(1,nfreq)] = np.array([[0,g],[0,0]])