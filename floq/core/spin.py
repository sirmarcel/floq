# Numerical functions needed for spin systems
import numpy as np


def hf(ncomp, freq, controls):
    # assemble hf for one spin, given a detuning
    # freq, the control amplitudes controls, and
    # ncomp components of the control pulse

    nc = 2*ncomp+1  # number of components in hf

    hf = np.zeros([nc, 2, 2], dtype='complex128')

    for k in xrange(0, ncomp):
        # the controls are ordered in reverse
        # compared to how they are placed in hf
        a = controls[-2*k-2]
        b = controls[-2*k-1]

        # The controls are placed symmetrically around
        # the centre of hf, so we can place them at the
        # same time to save us some work!
        hf[k, :, :] = np.array([[0.0, 0.25*(1j*a+b)],
                                [0.25*(1j*a-b), 0.0]])

        hf[-k-1, :, :] = np.array([[0.0, -0.25*(1j*a+b)],
                                   [0.25*(-1j*a+b), 0.0]])

    # Set centre (with Fourier index 0)
    hf[ncomp] = np.array([[freq/2.0, 0.0],
                          [0.0, -freq/2.0]])

    return hf


def dhf(ncomp, amp):
    # Assemble dhf for one spin, given ncomp components
    # in the control pulse, and an amplitude of amp

    nc = 2*ncomp+1
    npm = 2*ncomp
    dhf = np.zeros([npm, nc, 2, 2], dtype='complex128')

    for k in xrange(0, ncomp):
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
