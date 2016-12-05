import sys
sys.path.append('..')
sys.path.append('museum_of_forks')
import numpy as np
from floq.systems.spins import SpinEnsemble
import timeit


def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)
    return wrapped


def time_u(get_u, hf, params):
    time = min(timeit.Timer(wrapper(get_u, hf, params)).repeat(5, 20))/20
    return " U: " + str(round(time*1000, 3)) + " ms per execution"


def time_du(func, hf, dhf, params):
    time = min(timeit.Timer(wrapper(func, hf, dhf, params)).repeat(3, 1))
    return "dU: " + str(round(time, 3)) + " s per execution"




ncomp = 2
n = 6
freqs = 0.0*np.ones(n)+1.0*np.random.rand(n)
amps = 1.0*np.ones(n)+0.05*np.random.rand(n)

init = 0.5*np.ones(2*ncomp)

s = SpinEnsemble(n, ncomp, 1.5, freqs, amps)

print "---- Current version (Numba)"
import floq.core.evolution as ev

print time_u(ev.get_u, hf, params)
print time_du(ev.get_u_and_du, hf, dhf, params)
