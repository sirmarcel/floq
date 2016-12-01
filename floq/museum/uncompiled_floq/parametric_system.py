import floq.core.fixed_system as fs
import floq.evolution as ev
import floq.errors as er
import floq.helpers.index as h


class ParametericSystemBase(object):
    """
    Base class to specify a physical system that still has open parameters,
    such as the control amplitudes, the control duration, or other arbitrary
    parameters in the Hamiltonian.

    This needs to be sub-classed, and a subclass should provide:
    - get_system(controls)

    """

    def get_system(self, controls, t):
        raise NotImplementedError("get_system not implemented.")


    def is_nz_ok(self, controls, t):
        system = self.get_system(controls, t)
        try:
            u = ev.evolve_system(system)
        except er.EigenvalueNumberError:
            return False

        return h.is_unitary(u)


    def set_nz(self, controls, t):
        if self.is_nz_ok(controls, t):
            self.decrease_nz_until_not_ok(controls, t, step=max(10, self.nz/5))
            self.decrease_nz_until_not_ok(controls, t, step=max(10, self.nz/10))
            self.decrease_nz_until_not_ok(controls, t, step=2)
            self.increase_nz_until_ok(controls, t, step=2)
        else:
            self.increase_nz_until_ok(controls, t, step=max(10, self.nz/5))
            self.decrease_nz_until_not_ok(controls, t, step=2)
            self.increase_nz_until_ok(controls, t, step=2)


    def increase_nz_until_ok(self, controls, t, step=2):
        while self.is_nz_ok(controls, t) is False:
            self.nz += h.make_even(step)


    def decrease_nz_until_not_ok(self, controls, t, step=2):
        while self.is_nz_ok(controls, t) and self.nz-step > 3:
            self.nz -= h.make_even(step)




class ParametricSystemWithFunctions(ParametericSystemBase):
    """
    A system with parametric hf and dhf, which are passed as callables to the constructor.

    hf has to have the form hf(a,parameters)
    """

    def __init__(self, hf, dhf, nz, omega, parameters):
        """
        hf: callable hf(controls,parameters,omega)
        dhf: callable dhf(controls,parameters,omega)
        omega: 2 pi/T, the period of the Hamiltonian
        nz: number of Fourier modes to be considered during evolution
        parameters: a data structure that holds parameters for hf and dhf
        (dictionary is probably the best idea)
        """
        self.hf = hf
        self.dhf = dhf
        self.omega = omega
        self.nz = nz
        self.parameters = parameters

    def calculate_hf(self, controls):
        return self.hf(controls, self.parameters, self.omega)

    def calculate_dhf(self, controls):
        return self.dhf(controls, self.parameters, self.omega)

    def get_system(self, controls, t):
        hf = self.calculate_hf(controls)
        dhf = self.calculate_dhf(controls)
        return fs.FixedSystem(hf, dhf, self.nz, self.omega, t)
