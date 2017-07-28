import numpy as np
from benchmark.museum_of_forks.p0.systems.ensemble import EnsembleBase
from benchmark.museum_of_forks.p0.systems.parametric_system import ParametricSystemBase
import benchmark.museum_of_forks.p0.core.spin as spin


class SpinEnsemble(EnsembleBase):
    """
    A system of n non-interacting spins, where each
    spin is described by the Hamiltonian
    H(t) = w/2 s_z + 1/2 sum_k (a_k s_x + b_k s_y) sin(k omega t).

    Commonly, the values for w will be set slightly different
    for each spin, and a_k and b_k will be multiplied by some
    attenuation factor for each spin. Fidelities etc. will then be computed as ensemble averages.
    """

    def __init__(self, n, ncomp, omega, freqs, amps):
        """
        Initialise a SpinEnsemble instance with
        - n: number of spins
        - ncomp: number of components in the control pulse
         -> hf will have nc = 2*comp+1 components
        - omega: base frequency of control pulse
        - freqs: vector of n frequencies
        - amps: vector of n amplitudes
        """

        self.n = n
        self.ncomp = ncomp
        self.omega = omega
        self.freqs = freqs
        self.amps = amps
        self.max_amp = np.argmax(amps)

        self.np = 2*ncomp  # number of control parameters
        self.nc = 2*ncomp+1
        self.nz = 3*np.ones(self.n, dtype=int)

        self._systems = [SpinSystem(ncomp, amps[i], freqs[i], omega) for i in xrange(n)]

    @property
    def systems(self):
        return self._systems



class SpinSystem(ParametricSystemBase):
    """
    Describes a single spin with an amplitude (amp) that
    attenuates the control pulse, and a detuning (freq),
    controlled by a pulse with ncomp components and a carrier
    frequency of omega.

    The Hamiltonian is
    H(t) = w/2 s_z + 1/2 sum_k (a_k s_x + b_k s_y) sin(k omega t),
    with k going over ncomp components of the pulse.

    ncomp implies 2*ncomp = np control parameters,
    and np+1 non-zero Fourier components in Hf.
    """

    def __init__(self, ncomp, amp, freq, omega):
        """
        Initialise a SpinSystem.
        """
        super(SpinSystem, self).__init__()

        self.ncomp = ncomp
        self.amp = amp
        self.freq = freq
        self.omega = omega

        self.nz = 3
        self.dhf = spin.dhf(ncomp, amp)  # independent of controls!


    def _hf(self, controls):
        # Compute hf, given 2*ncomp control parameters,
        # ordered as follows:
        # [a1, b1, a2, b2, ... a_nc, b_nc]

        return spin.hf(self.ncomp, self.freq, self.amp*controls)


    def _dhf(self, controls):
        # dHf is independent of the controls, return cached value
        return self.dhf
