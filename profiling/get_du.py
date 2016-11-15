import sys
sys.path.append('..')
import numpy as np
import pdb
import scipy.optimize
import floq.systems.spins as spins
import floq.evolution as ev
import floq.optimization_task as ot
import floq.optimizer as opt
import floq.fidelity as fid

ncomp = 5
n = 1
freqs = 1.1*np.ones(n)+0.05*np.random.rand(n)
amps = 1.0*np.ones(n)+0.05*np.random.rand(n)
s = spins.SpinEnsemble(n, ncomp, 1.5, freqs, amps)

controls = 0.5*np.ones(2*ncomp)
s.set_nz(controls, 1.5)
systems = s.get_systems(controls, 1.5)

ev.evolve_system_with_derivatives(systems[0])