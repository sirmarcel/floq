import numpy as np


def operator_fidelity(system, u, target):
    """
    Calculate the operator fidelity between the unitaries
    u and target, defined as 1/dim * Re(trace(target^\dagger u)).

    This quantity is maximal when u = target, when it is equal to 1.
    """
    hs = hilbert_schmidt_product(target, u)
    return hs.real/system.params.dim


def d_operator_fidelity(system, u, dus, target):
    """
    Calculate the gradient of the operator fidelity.
    """
    return np.array([operator_fidelity(system, du, target) for du in dus])


def operator_distance(system, u, target):
    """
    Calculate a quantity proportional to the
    Hilbert-Schmidt distance
    tr( (u-target)(u^\dagger-target^\dagger) ) = 2 (dim - Re(tr(target^\dagger u))),
    to be precise the following quantity:
    1.0 - 1/dim * Re(trace(target^\dagger u)).

    This quantity is minimised when u = target, which makes it useful for use
    with the minimisation routines built into SciPy.
    """
    return 1.0 - operator_fidelity(system, u, target)


def d_operator_distance(system, u, dus, target):
    """
    Calculate the gradient of the operator distance.
    """
    df = d_operator_fidelity(system, u, dus, target)
    return -df


def hilbert_schmidt_product(a, b):
    """
    Compute the Hilbert-Schmidt inner product between
    operators a and b.
    """
    a = np.matrix(a)
    return np.trace(np.dot(a.getH(), b))
