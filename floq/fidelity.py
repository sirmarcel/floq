import numpy as np


def operator_fidelity(system, u, target):
    hs = hilbert_schmidt_product(target, u)
    return hs.real/system.params.dim


def d_operator_fidelity(system, u, dus, target):
    return np.array([operator_fidelity(system, du, target) for du in dus])


def operator_distance(system, u, target):
    return 1.0 - np.abs(operator_fidelity(system, u, target))


def d_operator_distance(system, u, dus, target):
    df = d_operator_fidelity(system, u, dus, target)
    return -df*np.sign(df)


def hilbert_schmidt_product(a, b):
    a = np.matrix(a)
    return np.trace(np.dot(a.getH(), b))
