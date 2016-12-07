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


def get_f_and_df(fid, base_controls):
    ctrl = np.random.rand(1)*base_controls
    fid.f(ctrl)
    fid.df(ctrl)


def run_opt(opt):
    opt.optimize()


def time_f(fidel, ctrl):
    time = min(timeit.Timer(wrapper(get_f, fidel, ctrl)).repeat(2, 5))/5
    return " F: " + str(round(time*1000, 3)) + " ms per execution"


def time_df(grad_fidel, ctrl):
    time = min(timeit.Timer(wrapper(get_f, grad_fidel, ctrl)).repeat(3, 1))
    return "dF: " + str(round(time, 3)) + " s per execution"


def time_f_and_df(fidel, ctrl):
    time = min(timeit.Timer(wrapper(get_f_and_df, fidel, ctrl)).repeat(3, 1))
    return "F+dF: " + str(round(time, 3)) + " s"


def time_opt(opt):
    time = min(timeit.Timer(wrapper(run_opt, opt)).repeat(2, 1))/1
    return "Op: " + str(round(time, 3)) + " s"


ncomp = 5
n = 100
freqs = 0.01*np.ones(n)-0.025+0.05*np.random.rand(n)
amps = 1.0*np.ones(n)-0.025+0.05*np.random.rand(n)
s = SpinEnsemble(n, ncomp, 1.5, freqs, amps)
ctrl = 0.5*np.ones(2*ncomp)
target = np.array([[0.208597 + 0.j, -0.691552 - 0.691552j],
                   [0.691552 - 0.691552j, 0.208597 + 0.j]])


from floq.optimization.optimizer import SciPyOptimizer

print "---- Karl's Version"
from floq.optimization.fidelity import OperatorDistance
from floq.parallel.worker import FidelityMaster

master = FidelityMaster(n, 2, s, OperatorDistance, t=1.0, target=target)
opt = SciPyOptimizer(master, ctrl, tol=1e-5)

print time_f(master.f, ctrl)
print time_df(master.df, ctrl)
print time_f_and_df(master, ctrl)
print time_opt(opt)
master.kill()


print "---- Queue Version"
from floq.optimization.fidelity import OperatorDistance
from floq.parallel.simple_ensemble import ParallelEnsembleFidelity

fid = ParallelEnsembleFidelity(s, OperatorDistance, t=1.0, target=target)
opt = SciPyOptimizer(fid, ctrl, tol=1e-5)

print time_f(fid.f, ctrl)
print time_df(fid.df, ctrl)
print time_f_and_df(fid, ctrl)
print time_opt(opt)


print "---- Legacy version"
from museum_of_forks.p0.optimization.fidelity import EnsembleFidelity, OperatorDistance
fid = EnsembleFidelity(s, OperatorDistance, t=1.0, target=target)
opt = SciPyOptimizer(fid, ctrl, tol=1e-5)

print time_f(fid.f, ctrl)
print time_df(fid.df, ctrl)
print time_f_and_df(fid, ctrl)
print time_opt(opt)
