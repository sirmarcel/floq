import floq.fixed_system as fs
import floq.evolution as ev
import floq.errors as er


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

    def get_u(self, controls, t):
        raise NotImplementedError("get_u not implemented.")

    def get_u_and_du(self, controls, t):
        raise NotImplementedError("get_u_and_du not implemented.")


class EnsembleBase(object):
    """
    Base class to specify an ensemble of physical systems
    that still have open parameters.

    This is base class defining the API and cannot be used, it needs to be sub-classed.

    """

    def get_systems(self, controls, t):
        raise NotImplementedError("get_system not implemented.")

    def get_us(self, controls, t):
        raise NotImplementedError("get_u not implemented.")

    def get_us_and_dus(self, controls, t):
        raise NotImplementedError("get_u_and_du not implemented.")


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

    def get_u(self, controls, t):
        fixed_system = self.get_system(controls, t)
        u, new_nz = ev.evolve_system(fixed_system)
        self.nz = new_nz
        return u

    def get_u_and_du(self, controls, t):
        fixed_system = self.get_system(controls, t)
        u, du, new_nz = ev.evolve_system_with_derivatives(fixed_system)
        self.nz = new_nz
        return [u, du]
