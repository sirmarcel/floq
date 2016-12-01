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


@autojit
def make_even(n):
    if n % 2 == 0:
        return n
    else:
        return n+1
