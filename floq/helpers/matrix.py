import numpy as np


def is_unitary(u, tolerance=1e-10):
    unitary = np.eye(u.shape[0])
    umat = np.mat(u)
    product = umat.H * umat
    return np.allclose(product, unitary, atol=tolerance)


def adjoint(m):
    return np.transpose(np.conj(m))
