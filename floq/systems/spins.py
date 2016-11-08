import numpy as np
import floq.parametric_system as ps


class SpinEnsemble(object):
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

        self.np = 2*ncomp  # number of control parameters
        self.nc = 2*ncomp+1

        self._dhf = None

    @property
    def dhf(self):
        # dhf is independent of the controls,
        # so we only need to compute it once
        if self_dhf is not None:
            return self._dhf
        else:
            self._dhf = self._build_dhf()
            return self._dhf


    def get_single_system(self, i, controls, t):
        """
        Build an instance of FixedSystem for the ith spin,
        with the given controls -- these are expected to be
        2*ncomp values ordered as follows:
        [a1, b1, a2, b2, ... a_nc, b_nc]
        """
        hf = self._build_single_hf(self.freq[i], self.amps[i], controls)
        dhf = self._build_single_dhf(self.freq[i], self.amps[i], controls)


    def _build_single_hf(self, freq, amp, controls):
        # assemble hf for one spin

        hf = np.zeros([self.nc, 2, 2], dtype='complex128')

        for k in xrange(0, self.ncomp):
            # the controls are ordered in reverse
            # compared to how they are placed in hf
            a = controls[-2*k-2]*amp
            b = controls[-2*k-1]*amp

            # The controls are placed symmetrically around
            # the centre of hf, so we can place them at the
            # same time to save us some work!
            hf[k, :, :] = np.array([[0.0, 0.25*(1j*a+b)],
                                    [0.25*(1j*a-b), 0.0]])

            hf[-k-1, :, :] = np.array([[0.0, -0.25*(1j*a+b)],
                                       [0.25*(-1j*a+b), 0.0]])

        # Set centre (with Fourier index 0)
        hf[self.ncomp] = np.array([[freq/2.0, 0.0],
                                   [0.0, -freq/2.0]])

        return hf

    def _build_single_dhf(self, amp):
        dhf = np.zeros([self.np, self.nc, 2, 2], dtype='complex128')

        for k in xrange(0, self.ncomp):
            i_a = -2*k-2
            i_b = -2*k-1

            dhf[i_a, k, :, :] = np.array([[0.0, 0.25j*amp],
                                          [0.25j*amp, 0.0]])
            dhf[i_a, -k-1, :, :] = np.array([[0.0, -0.25j*amp],
                                             [-0.25j*amp, 0.0]])

            dhf[i_b, k, :, :] = np.array([[0.0, 0.25*amp],
                                          [-0.25*amp, 0.0]])
            dhf[i_b, -k-1, :, :] = np.array([[0.0, -0.25*amp],
                                             [0.25*amp, 0.0]])

        return dhf

    def _build_dhf(self):
        dhf = np.zeros([self.n, self.np, self.nc, 2, 2], dtype='complex128')
        for i in xrange(0, self.n):
            dhf[i, :, :, :] = self._build_single_dhf(self.amps[i])

        return dhf
