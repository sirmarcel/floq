import scipy.optimize as opt
import numpy as np
import floq.optimization_task as ot


class OptimizerBase(object):
    def __init__(self, task):
        self.task = task

    def optimize(self):
        raise NotImplementedError


class SciPyOptimizer(OptimizerBase):
    def __init__(self, task, t, method='BFGS', tol=1e-5):
        self.task = task
        self.t = t
        self.method = method

        self.fid = self._wrap_t(self.task.fidelity, self.t)
        self.dfid = self._wrap_t(self.task.grad_fidelity, self.t)

    def _wrap_t(self, func, t):
        def wrapped(args):
            return func(args, t)
        return wrapped

    def optimize(self):
        res = opt.minimize(self.fid, self.task.init, jac=self.dfid, method=self.method)
        return res
