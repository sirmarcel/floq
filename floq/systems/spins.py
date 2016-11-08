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

    def __init__(self, n, nc, omega, freqs, amps):
        """
        Initialise a SpinEnsemble instance with
        - n: number of spins
        - nc: number of components in the control pulse
        - omega: base frequency of control pulse
        - freqs: vector of n frequencies
        - amps: vector of n amplitudes
        """

        self.n = n
        self.nc = nc
        self.omega = omega
        self.freqs = freqs
        self.amps = amps


    def get_single_system(self, i, controls, t):
        """
        Build an instance of FixedSystem for the ith spin,
        with the given controls -- these are expected to be
        2*nc values ordered as follows:
        [a1, b1, a2, b2, ... a_nc, b_nc]
        """
        hf = self._build_single_hf(self.freq[i], self.amps[i], controls)
        dhf = self._build_single_dhf(self.freq[i], self.amps[i], controls)


    def _build_single_hf(self, nc, freq, amp, controls):
        # assemble hf for one spin

        hf = np.zeros([2*nc+1, 2, 2], dtype='complex128')
        for k in xrange(0, nc):
            # the controls are ordered in reverse
            # compared to how they are placed in hf
            a = controls[-2*k-2]*amp
            b = controls[-2*k-1]*amp

            hf[k, :, :] = np.array([[0.0, 0.25*(1j*a+b)],
                                    [0.25*(1j*a-b), 0.0]])

            hf[-k-1, :, :] = np.array([[0.0, -0.25*(1j*a+b)],
                                       [0.25*(-1j*a+b), 0.0]])

        hf[nc] = np.array([[freq/2.0, 0.0],
                           [0.0, -freq/2.0]])

        return hf
