# Provide templates and implementations for FidelityComputer class,
# which wraps a ParametricSystem and computes F and dF for given controls
from floq.core.fidelities import d_operator_distance, operator_distance
import numpy as np


class FidelityComputerBase(object):
    """
    Define how to calculate a fidelity and its gradient from a
    ParametricSystem.

    This is a base class and needs to be sub-classed. Sub-classes
    should implement f(controls_and_t) and df(controls_and_t), where
    controls_and_t is an array of control amplitudes and possibly
    the duration of the pulse as last entry.

    The __init__ should take the form __init__(self, system, **kwargs)
    for compatibility with EnsembleFidelity.
    """

    def __init__(self, system):
        self.system = system


    def f(self, controls_and_t):
        raise NotImplementedError


    def df(self, controls_and_t):
        raise NotImplementedError



class EnsembleFidelity(FidelityComputerBase):
    """
    With a given Ensemble, and a FidelityComputer,
    calculate the average fidelity over the whole ensemble.
    """

    def __init__(self, ensemble, fidelity, **params):
        self.fidelities = [fidelity(sys, **params) for sys in ensemble.systems]

    def f(self, controls_and_t):
        return np.mean([fid.f(controls_and_t) for fid in self.fidelities])


    def df(self, controls_and_t):
        return np.mean([fid.df(controls_and_t) for fid in self.fidelities], axis=0)



class OperatorDistance(FidelityComputerBase):
    """
    Calculate the operator distance (see core.fidelities for details)
    for a given ParametricSystem and a fixed pulse duration t.
    """

    def __init__(self, system, t, target):
        super(OperatorDistance, self).__init__(system)

        self.t = t
        self.target = target


    def f(self, controls):
        u = self.system.u(controls, self.t)
        return operator_distance(u, self.target)


    def df(self, controls):
        u = self.system.u(controls, self.t)
        du = self.system.du(controls, self.t)
        return d_operator_distance(u, du, self.target)
