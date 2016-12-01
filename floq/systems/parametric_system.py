import floq.core.fixed_system as fs
import floq.evolution as ev
import floq.errors as er


class ParametericSystemBase(object):
    """
    Specifies a physical system that still has open parameters,
    such as the control amplitudes, the control duration, or other arbitrary
    parameters in the Hamiltonian.

    The base class provides two functions:
    - u(controls, t)
    - du(controls, t),
    which implement basic caching and automatically update self.nz.

    To function, this needs to be sub-classed, and a subclass should provide:
    - _hf(controls),
    - _dhf(controls)
    as well as __init__, which has to set self.nz, self.omega and self.t, and needs to call
    this class' __init__.

    """
    def _hf(self, controls):
        raise NotImplementedError


    def _dhf(self, controls):
        raise NotImplementedError



    def u(self, controls, t):
        if self._last_controls == controls and self._last_t == t:
            return self._fixed_system.u
        else:
            self._last_controls = controls
            self._last_t = t

            hf = self._hf(controls)
            dhf = self._dhf(controls)
            self._fixed_system = fs.FixedSystem(hf, dhf, self.nz, self.omega, self.t)

            u = self._fixed_system.u
            self.nz = self._fixed_system.params.nz
            return u


    def du(self, controls, t):
        if self._last_controls == controls and self._last_t == t:
            return self._fixed_system.du
        else:
            self._last_controls = controls
            self._last_t = t

            hf = self._hf(controls)
            dhf = self._dhf(controls)
            self._fixed_system = fs.FixedSystem(hf, dhf, self.nz, self.omega, self.t)

            du = self._fixed_system.du
            self.nz = self._fixed_system.params.nz
            return du


class ParametricSystemWithFunctions(ParametericSystemBase):
    """
    A ParametricSystem that wraps callables hf and dhf.
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

    def _hf(self, controls):
        return self.hf(controls, self.parameters, self.omega)


    def _dhf(self, controls):
        return self.dhf(controls, self.parameters, self.omega)
