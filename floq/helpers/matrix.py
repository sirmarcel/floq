import numpy as np


def is_unitary(u, tolerance=1e-10):
    unitary = np.eye(u.shape[0])
    umat = np.mat(u)
    product = umat.H * umat
    return np.allclose(product, unitary, atol=tolerance)


def adjoint(m):
    return np.transpose(np.conj(m))


def gram_schmidt(vecs):
    """Computes an orthonormal basis for the given set of vectors.

    Vectors are expected to be supplied as the columns of an array,
    i.e. the first vector should be vecs[:,0] etc.

    An array of the same form is returned.

    The algorithm implemented is the modified Gram-Schmidt procedure,
    given in Numerical Methods for Large Eigenvalue Problems: Revised Edition, algorithm 1.2.
    """

    result = np.zeros_like(vecs)
    n = vecs.shape[1]

    r = np.linalg.norm(vecs[:, 0])
    if r == 0.0:
        raise ArithmeticError("Vector with norm 0 occured.")
    else:
        result[:, 0] = vecs[:, 0]/r

    for j in xrange(1, n):
        q = vecs[:, j]

        for i in xrange(j):
            rij = np.dot(q, result[:, i])
            q = q-rij*result[:, i]

        rjj = np.linalg.norm(q)

        if rjj == 0.0:
            raise ArithmeticError("Vector with norm 0 occured.")
        else:
            result[:, j] = q/rjj

    return result
