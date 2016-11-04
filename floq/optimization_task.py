import numpy as np
import floq.parametric_system as ps
import floq.evolution as ev


class OptimizationTaskBase(object):
    """
    Base class to specify an optimization problem,
    i.e. a physical (parametrised) system,
    a fidelity, a target and an initial control vector.
    """
    def system(self):
        raise NotImplementedError

    def fidelity(self, control, t):
        raise NotImplementedError

    def grad_fidelity(self, control, t):
        raise NotImplementedError

    def target_fidelity(self, control, t):
        raise NotImplementedError

    def target(self):
        raise NotImplementedError

    def target_fid(self):
        raise NotImplementedError


class OptimizationTaskWithFunctions(OptimizationTaskBase):
    def __init__(self, system, fid, dfid, target, target_fid, init):
        """
        system: Instance of ParametricSystemBase
        fid: callable fid(fixed_system,u,target)
        dfid: callable fid(fixed_system,u,du,target)
        """

        self.system = system
        self.fid = fid
        self.dfid = dfid
        self.target = target
        self.target_fid = target_fid
        self.init = init

    def fidelity(self, controls, t):
        fixed = self.system.get_system(controls, t)
        u = ev.evolve_system(fixed)
        return self.fid(fixed, u, self.target)

    def grad_fidelity(self, controls, t):
        fixed = self.system.get_system(controls, t)
        u, du = ev.evolve_system_with_derivatives(fixed)
        return self.dfid(fixed, u, du, self.target)