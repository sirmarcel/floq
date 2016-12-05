import sys
sys.path.append('..')
sys.path.append('museum_of_forks')
from floq.systems.spins import SpinEnsemble
import numpy as np
import timeit


def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)
    return wrapped


def get_f(fid, base_controls):
    fid(np.random.rand(1)*base_controls)


def time_u(fidel, ctrl):
    time = min(timeit.Timer(wrapper(get_f, fidel, ctrl)).repeat(2, 5))/5
    return " U: " + str(round(time*1000, 3)) + " ms per execution"


def time_du(grad_fidel, ctrl):
    time = min(timeit.Timer(wrapper(get_f, grad_fidel, ctrl)).repeat(3, 1))
    return "dU: " + str(round(time, 3)) + " s per execution"


ncomp = 6
n = 20
freqs = 0.0*np.ones(n)+1.0*np.random.rand(n)
amps = 1.0*np.ones(n)+0.05*np.random.rand(n)
s = SpinEnsemble(n, ncomp, 1.5, freqs, amps)
ctrl = 0.5*np.ones(2*ncomp)
target = np.array([[0.105818 - 0.324164j, -0.601164 - 0.722718j],
                   [0.601164 - 0.722718j, 0.105818 + 0.324164j]])



print "---- Current version"
from floq.optimization.fidelity import EnsembleFidelity, OperatorDistance
fid = EnsembleFidelity(s, OperatorDistance, t=1.0, target=target)

print time_u(fid.f, ctrl)
print time_du(fid.df, ctrl)


print "---- Legacy version"
from museum_of_forks.p0.optimization.fidelity import EnsembleFidelity, OperatorDistance
fid = EnsembleFidelity(s, OperatorDistance, t=1.0, target=target)

print time_u(fid.f, ctrl)
print time_du(fid.df, ctrl)