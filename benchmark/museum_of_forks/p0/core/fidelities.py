import numpy as np


def transfer_fidelity(u, target):
    """
    Compute how well the unitary u transfers an
    initial state |i> to a final state |f>, quantified by
    fid = |<f| u |i>|^2.

    Note that target is expected to contain both the
    initial and final state, as follows:
    target[0, :] = |i>
    target[1, :] = |f>.
    """
    i = target[0, :]
    f = np.conj(np.transpose(target[1, :]))
    ui = np.dot(u, i)
    fui = np.dot(f, ui)
    return np.abs(fui)**2


def operator_fidelity(u, target):
    """
    Calculate the operator fidelity between the unitaries
    u and target, defined as 1/dim * Re(trace(target^\dagger u)).

    This quantity is maximal when u = target, when it is equal to 1.
    """
    dim = u.shape[0]
    hs = hilbert_schmidt_product(target, u)
    return hs.real/dim


def d_operator_fidelity(u, dus, target):
    """
    Calculate the gradient of the operator fidelity.
    """
    return np.array([operator_fidelity(du, target) for du in dus])


def operator_distance(u, target):
    """
    Calculate a quantity proportional to the
    Hilbert-Schmidt distance
    tr( (u-target)(u^\dagger-target^\dagger) ) = 2 (dim - Re(tr(target^\dagger u))),
    to be precise the following quantity:
    1.0 - 1/dim * Re(trace(target^\dagger u)).

    This quantity is minimised when u = target, which makes it useful for use
    with the minimisation routines built into SciPy.
    """
    return 1.0 - operator_fidelity(u, target)


def d_operator_distance(u, dus, target):
    """
    Calculate the gradient of the operator distance.
    """
    df = d_operator_fidelity(u, dus, target)
    return -df


def hilbert_schmidt_product(a, b):
    """
    Compute the Hilbert-Schmidt inner product between
    operators a and b.
    """
    a = np.matrix(a)
    return np.trace(np.dot(a.getH(), b))
