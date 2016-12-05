# Provide templates and implementations for FidelityComputer class,
# which wraps a ParametricSystem and computes F and dF for given controls
from floq.core.fidelities import d_operator_distance, operator_distance


class FidelityComputerBase(object):
    """

    """

    def __init__(self, system):
        self.system = system


    def f(self, controls_and_t):
        raise NotImplementedError


    def df(self, controls_and_t):
        raise NotImplementedError



class OperatorDistance(FidelityComputerBase):
    """
    Calculate the operator distance (see core.fidelities for details)
    for a given ParametricSystem and a fixed pulse duration t.
    """

    def __init__(self, self.t, system, target):
        super(OperatorDistance, self).__init__(system)

        self.target = target


    def f(self, controls):
        u = self.system.u(controls, self.t)
        return operator_distance(u, self.target)


    def df(self, controls):
        u = self.system.u(controls, self.t)
        du = self.system.du(controls, self.t)
        return d_operator_distance(u, du, self.target)
