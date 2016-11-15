import sys
sys.path.append('..')
import numpy as np
import floq.systems.spins as spins
import timeit


def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)
    return wrapped

ncomp = 10
n = 1
freqs = 1.1*np.ones(n)
amps = 1.0*np.ones(n)
s = spins.SpinEnsemble(n, ncomp, 1.5, freqs, amps)

controls = 0.5*np.ones(2*ncomp)
s.set_nz(controls, 1.5)
system = s.get_systems(controls, 1.5)[0]

print "Current version"
import floq.core.evolution as ev

print timeit.timeit(wrapper(ev.do_evolution, system.hf, system.params), number=100)
print timeit.timeit(wrapper(ev.do_evolution_with_derivatives, system.hf,  system.dhf, system.params), number=1)


print "Sparse version"
import floq.museum.p2.evolution as ev

print timeit.timeit(wrapper(ev.do_evolution, system.hf, system.params), number=100)
print timeit.timeit(wrapper(ev.do_evolution_with_derivatives, system.hf,  system.dhf, system.params), number=1)

print "Baseline version (non sparse)"
import floq.museum.p1.evolution as ev

print timeit.timeit(wrapper(ev.do_evolution, system.hf, system.params), number=100)
print timeit.timeit(wrapper(ev.do_evolution_with_derivatives, system.hf,  system.dhf, system.params), number=1)
