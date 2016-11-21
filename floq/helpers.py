import numpy as np
from numba import autojit


@autojit
def n_to_i(num, n):
    """
    Translate num, ranging from
    -(n-1)/2 through (n-1)/2
    into an index i from 0 to n-1

    If num > (n-1)/2, map it into the interval

    This is necessary to translate from a physical
    Fourier mode number to an index in an array.
    """
    cutoff = (n-1)/2
    return (num+cutoff) % n


@autojit
def i_to_n(i, n):
    """
    Translate index i, ranging from 0 to n-1
    into a number from -(n-1)/2 through (n-1)/2

    This is necessary to translate from an index to a physical
    Fourier mode number.
    """
    cutoff = (n-1)/2
    return i-cutoff


def is_unitary(u, tolerance=1e-10):
    unitary = np.eye(u.shape[0])
    umat = np.mat(u)
    product = umat.H * umat
    return np.allclose(product, unitary, atol=tolerance)


@autojit
def make_even(n):
    if n % 2 == 0:
        return n
    else:
        return n+1


@autojit(nopython=True)
def numba_zeros(dims):
    ary = np.empty(dims, dtype=np.complex128)
    ary[:] = 0.0+0.0j
    return ary


@autojit(nopython=True)
def numba_outer(a, b):
    m = a.shape[0]
    n = b.shape[0]
    result = np.empty((m, n), dtype=np.complex128)
    for i in range(m):
        for j in range(n):
            result[i, j] = a[i]*b[j]
    return result