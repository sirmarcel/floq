import numpy as np
import floq.parametric_system as ps

def hf(g,parameters,omega=None):
    e1 = parameters[0]
    e2 = parameters[1]
    hf = np.zeros([3,2,2])
    hf[0] = np.array([[0,0],[g,0]])
    hf[1] = np.array([[e1,0],[0,e2]])
    hf[2] = np.array([[0,g],[0,0]])
    return hf

def dhf(g,parameters,omega=None):
    return np.array(hf(1.0,[0.0,0.0]))

def get_rabi_system(energies,omega):
    return ps.ParametricSystemWithFunctions(hf,dhf,21,omega,energies)