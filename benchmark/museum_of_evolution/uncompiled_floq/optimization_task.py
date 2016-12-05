import numpy as np
import floq.systems.parametric_system as ps
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
        u = ev.evolve_system(fixed)
        return self.fid(fixed, u, self.target)

    def grad_fidelity(self, controls, t):
        fixed = self.system.get_system(controls, t)
        u, du = ev.evolve_system_with_derivatives(fixed)
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
        self.ensemble.increase_nz_until_ok(controls, t, step=4)

        fixed_systems = self.ensemble.get_systems(controls, t)

        fid = 0.0
        for sys in fixed_systems:
            u = ev.evolve_system(sys)
            fid += self.fid(sys, u, self.target)
        mean_fid = fid/self.ensemble.n
        print mean_fid
        return mean_fid

    def grad_fidelity(self, controls, t):
        self.ensemble.increase_nz_until_ok(controls, t, step=4)

        fixed_systems = self.ensemble.get_systems(controls, t)

        dfid = np.zeros(self.ensemble.np)
        for sys in fixed_systems:
            u, du = ev.evolve_system_with_derivatives(sys)
            dfid += self.dfid(sys, u, du, self.target)
        mean_dfid = dfid/self.ensemble.n
        print mean_dfid
        return mean_dfid
