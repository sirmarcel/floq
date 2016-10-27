import numpy as np
import sys
sys.path.append('..')
import floq.evolution as ev
import floq.dtos as dtos
 
def fitFunc(x, a, b, c):
    return a*x**b+c

def rabi_hf(g,e1,e2):
    hf = np.zeros([3,2,2])
    hf[0] = np.array([[0,0],[g,0]])
    hf[1] = np.array([[e1,0],[0,e2]])
    hf[2] = np.array([[0,g],[0,0]])
    return hf

def do_u(nz):
  t = 5.23
  g = 0.123
  e1 = 1.0
  e2 = 2.5
  omega = 5.0
  p = dtos.FloquetProblemParameters(2,nz,omega,t)

  hf = rabi_hf(g,e1,e2)

  return ev.do_evolution(hf,p)

do_u(1001)