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
    if num > cutoff:
        return (num % n)-n+cutoff
    if num < -cutoff:
        return -(-num % n)+n+cutoff
    else:
        return num+cutoff


def i_to_n(i, n):
    """
    Translate index i, ranging from 0 to n-1
    into a number from -(n-1)/2 through (n-1)/2

    This is necessary to translate from an index to a physical
    Fourier mode number.
    """
    cutoff = (n-1)/2
    return i-cutoff
