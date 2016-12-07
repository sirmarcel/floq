# You got nothing to lose but your chains!
import multiprocessing as mp
import numpy as np
from floq.optimization.fidelity import FidelityComputerBase


def run_fid(pair):
    fid = pair[0]
    ctrl = pair[1]
    fid.f(ctrl)
    return fid


def run_dfid(pair):
    fid = pair[0]
    ctrl = pair[1]
    fid.df(ctrl)
    return fid


class ParallelEnsembleFidelity(FidelityComputerBase):
    """
    With a given Ensemble, and a FidelityComputer,
    calculate the average fidelity over the whole ensemble.
    """

    def __init__(self, ensemble, fidelity, **params):
        self.fidelities = [fidelity(sys, **params) for sys in ensemble.systems]
        self.pool = mp.Pool()


    def f(self, controls_and_t):
        self.dispatch_f_to_pool(controls_and_t)
        f = np.mean([fid.f(controls_and_t) for fid in self.fidelities])
        return f


    def df(self, controls_and_t):
        self.dispatch_df_to_pool(controls_and_t)
        df = np.mean([fid.df(controls_and_t) for fid in self.fidelities], axis=0)
        return df


    def dispatch_f_to_pool(self, controls_and_t):
        items = [[fid, controls_and_t] for fid in self.fidelities]
        self.fidelities = self.pool.map(run_fid, items)


    def dispatch_df_to_pool(self, controls_and_t):
        items = [[fid, controls_and_t] for fid in self.fidelities]
        self.fidelities = self.pool.map(run_dfid, items)