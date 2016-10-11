import numpy as np

def build_k(hf,dim,nw):
  """
  Build the Floquet-Hamiltonian K 
  from the Fourier transform of the system Hamiltonian
  """
  hf_n_modes = hf.shape[0]
  hf_cutoff = (hf_n_modes-1)/2
  k = numpy.zeros([dim*nw,dim*nw])
  for eta in xrange(-hf_cutoff,hf_cutoff+1):
    i = eta_to_i(eta,nw)