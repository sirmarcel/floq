import numpy as np
import floq.parametric_system as ps
import floq.evolution as ev


class OptimizationTaskBase(object):
    """
    Base class to specify an optimization problem,
    i.e. a physical (parametrised) system,
    a fidelity, a target and an initial control vector.
    """
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
        u, new_nz = ev.evolve_system(fixed)
        return self.fid(fixed, u, self.target)

    def grad_fidelity(self, controls, t):
        fixed = self.system.get_system(controls, t)
        u, du, new_nz = ev.evolve_system_with_derivatives(fixed)
        return self.dfid(fixed, u, du, self.target)


class EnsembleOptimizationTask(OptimizationTaskBase):
    """
    An optimization task where an ensemble of (non-interacting)
    systems is optimised.
    """

    def __init__(self, ensemble, fid, dfid, target, init):
        self.ensemble = ensemble
        self.fid = fid
        self.dfid = dfid
        self.target = target
        self.init = init

        self.ensemble.set_nz(init, 0.5/self.ensemble.omega)

    def fidelity(self, controls, t):
        us = self.ensemble.get_us(controls, t)

        fid = 0.0
        for u in us:
            fid += self.fid(u, self.target)
        mean_fid = fid/self.ensemble.n
        print mean_fid
        return mean_fid

    def grad_fidelity(self, controls, t):
        us, dus = self.ensemble.get_us_and_dus(controls, t)

        dfid = np.zeros(self.ensemble.np)
        for i in xrange(self.ensemble.n):
            dfid += self.dfid(us[i], dus[i], self.target)
        mean_dfid = dfid/self.ensemble.n
        print mean_dfid
        return mean_dfid
