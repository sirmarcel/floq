import numpy as np

class FloquetProblem():
  """docstring for FloquetProblem"""
  def __init__(self, hf, w):
    if isinstance(hf,np.ndarray):
      self.hf = hf  
    else:
      raise TypeError('HF must be a Numpy ndarray')
    
    self.nw = hf.shape[0]
    self.dim = hf.shape[1]
    self.w = w