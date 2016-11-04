import numpy as np


def trace_overlap(system, u, target):
    hs = hilbert_schmidt_product(target, u)
    return hs.real/system.params.dim


def d_trace_overlap(system, u, dus, target):
    return np.array([trace_overlap(system, du, target) for du in dus])


def overlap_to_minimise(system, u, target):
    return 1.0 - np.abs(trace_overlap(system, u, target))


def d_overlap_to_minimise(system, u, dus, target):
    df = d_trace_overlap(system, u, dus, target)
    return -df*np.sign(df)


def hilbert_schmidt_product(a, b):
    a = np.matrix(a)
    return np.trace(np.dot(a.getH(), b))
