import scipy.optimize as opt
import numpy as np


class OptimizerBase(object):
    """
    Define a way to optimise a given fidelity.
    This is a base class.
    """
    def __init__(self, task):
        self.task = task

    def optimize(self):
        raise NotImplementedError


class SciPyOptimizer(OptimizerBase):
    """
    A wrapper around scipy.minimize.
    """
    def __init__(self, fid, init, method='BFGS', tol=1e-5, options={}):
        self.fid = fid
        self.init = init
        self.method = method
        self.tol = tol
        self.options = options


    def optimize(self):
        res = opt.minimize(self.fid.f, self.init, jac=self.fid.df, method=self.method,
                           tol=self.tol, callback=self.fid.iterate, options=self.options)
        return res
