import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as la
import floq.helpers as h
import floq.blockmatrix as bm
import floq.fixed_system as fs
import floq.errors as errors
import itertools
import copy
from multiprocessing import Pool


def do_evolution(hf, params):
    """
    Calculate the time evolution operator U
    given a Fourier transformed Hamiltonian Hf
    """
    k = build_k(hf, params)

    vals, vecs = find_eigensystem(k, params)

    phi = calculate_phi(vecs)
    psi = calculate_psi(vecs, params)

    return calculate_u(phi, psi, vals, params)


def do_evolution_with_derivatives(hf, dhf, params):
    """
    Calculate the time evolution operator U
    given a Fourier transformed Hamiltonian Hf,
    as well as its derivative dU given dHf
    """
    k = build_k(hf, params)

    vals, vecs = find_eigensystem(k, params)

    phi = calculate_phi(vecs)
    psi = calculate_psi(vecs, params)

    u = calculate_u(phi, psi, vals, params)

    dk = build_dk(dhf, params)
    
    du = calculate_du(dk, psi, vals, vecs, params)

    return [u, du]



def build_k(hf, p):
    hf_max = (p.nc-1)/2
    nz = p.nz
    nc = p.nc
    dim = p.dim
    omega = p.omega



    k = np.zeros([p.k_dim, p.k_dim], dtype='complex128')

    # Assemble K by placing each component of Hf in turn, which
    # for a fixed Fourier index lie on diagonals, with 0 on the
    # main diagonal, positive numbers on the right and negative on the left
    #
    # The first row is therefore essentially Hf(0) Hf(1) ... Hf(hf_max) 0 0 0 ...
    # The last row is then ... 0 0 0 Hf(-hf_max) ... Hf(0)
    # Note that the main diagonal acquires a factor of omega*identity*(row/column number)

    for n in xrange(-hf_max, hf_max+1):
        start_row = max(0, n)  # if n < 0, start at row 0
        start_col = max(0, -n)  # if n > 0, start at col 0

        stop_row = min((nz-1)+n, nz-1)
        stop_col = min((nz-1)-n, nz-1)

        row = start_row
        col = start_col

        current_component = hf[h.n_to_i(n, nc)]

        while row <= stop_row and col <= stop_col:
            if n == 0:
                block = current_component + np.identity(dim)*omega*h.i_to_n(row, nz)
                bm.set_block_in_matrix(block, k, dim, nz, row, col)
            else:
                bm.set_block_in_matrix(current_component, k, dim, nz, row, col)

            row += 1
            col += 1

    return k


def build_dk(dhf, p):
    p2 = copy.copy(p)
    p2.omega = 0.0

    return np.array([build_k(dhf[i], p2) for i in xrange(0, p.np)])



def find_eigensystem(k, p):
    # Find eigenvalues and eigenvectors for k,
    # identify the dim unique ones,
    # return them in a segmented form
    vals, vecs = compute_eigensystem(k, p)

    unique_vals = find_unique_vals(vals, p)

    vals = vals.round(p.decimals)
    indices_unique_vals = [np.where(vals == eva)[0][0] for eva in unique_vals]

    unique_vecs = np.array([vecs[:, i] for i in indices_unique_vals])

    unique_vecs = separate_components(unique_vecs, p.nz)

    return [unique_vals, unique_vecs]


def compute_eigensystem(k, p):
    # Find eigenvalues and eigenvectors of k,
    # using the method specified in the parameters
    if p.sparse:
        k = sp.csc_matrix(k)

        number_of_eigs = min(2*p.dim, p.k_dim)
        vals, vecs = la.eigs(k, k=number_of_eigs, sigma=0.0)
    else:
        vals, vecs = np.linalg.eig(k)
        vals, vecs = trim_eigensystem(vals, vecs, p)

    vals = vals.real.astype(np.float64, copy=False)

    return vals, vecs


def trim_eigensystem(vals, vecs, p):
    # Trim eigenvalues and eigenvectors to only 2*dim ones
    # clustered around zero

    # Sort eigenvalues and -vectors in increasing order
    idx = vals.argsort()
    vals = vals[idx]
    vecs = vecs[:, idx]

    # Only keep values around 0
    middle = p.k_dim/2
    cutoff_left = max(0, middle - p.dim)
    cutoff_right = min(p.k_dim, cutoff_left + 2*p.dim)

    cut_vals = vals[cutoff_left:cutoff_right]
    cut_vecs = vecs[:, cutoff_left:cutoff_right]

    return cut_vals, cut_vecs


def find_unique_vals(vals, p):
    # In the list of values supplied, find the set of dim
    # e_i that fulfil (e_i - e_j) mod omega != 0 for all i,j,
    # and that lie closest to 0.

    mod_vals = np.mod(vals, p.omega)
    mod_vals = mod_vals.round(decimals=p.decimals)  # round to suppress floating point issues

    unique_vals = np.unique(mod_vals)

    # the unique_vals are ordered and >= 0, but we'd rather have them clustered around 0
    should_be_negative = np.where(unique_vals > p.omega/2.)
    unique_vals[should_be_negative] = (unique_vals[should_be_negative]-p.omega).round(p.decimals)

    if unique_vals.shape[0] != p.dim:
        raise errors.EigenvalueNumberError(vals, unique_vals)
    else:
        return np.sort(unique_vals)


def separate_components(vecs, n):
    # Given an array of vectors vecs,
    # return an array of each of the vectors split into n sub-arrays

    return np.array([np.split(eva, n) for eva in vecs])



def calculate_phi(vecs):
    # Given an array of eigenvectors vecs,
    # sum over all frequency components in each

    return np.array([np.sum(eva, axis=0) for eva in vecs])


def calculate_psi(vecs, p):
    # Given an array of eigenvectors vecs,
    # sum over all frequency components in each,
    # weighted by exp(- i omega t n), with n
    # being the Fourier index of the component

    psi = np.zeros([p.dim, p.dim], dtype='complex128')

    for k in xrange(0, p.dim):
        partial = np.zeros(p.dim, dtype='complex128')
        for i in xrange(0, p.nz):
            num = h.i_to_n(i, p.nz)
            partial += np.exp(1j*p.omega*p.t*num)*vecs[k][i]
        psi[k, :] = partial

    return psi



def calculate_u(phi, psi, energies, p):
    u = np.zeros([p.dim, p.dim], dtype='complex128')

    for k in xrange(0, p.dim):
        u += np.exp(-1j*p.t*energies[k])*np.outer(psi[k], np.conj(phi[k]))

    return u


def calculate_du(dk, psi, vals, vecs, p):
    dim = p.dim
    nz_max = p.nz_max
    nz = p.nz
    npm = p.np
    omega = p.omega
    t = p.t

    vecsstar = np.conj(vecs)

    du = np.zeros([npm, dim, dim], dtype='complex128')

    # (i1,n1) & (i2,n2) iterate over the full spectrum of k:
    # i1, i2: unique eigenvalues/-vectors in 0th Brillouin zone
    # n1, n2: related vals/vecs derived by shifting with those offsets (lying in the nth BZs)
    uniques = xrange(0, dim)
    offsets = xrange(-nz_max, nz_max+1)

    for n1 in offsets:
        phase = t*np.exp(1j*omega*t*n1)
        for n2 in offsets:
            for i1, i2 in itertools.product(uniques, uniques):
                e1 = vals[i1] + n1*omega
                e2 = vals[i2] + n2*omega

                v1 = np.roll(vecsstar[i1], n1, axis=0)
                v2 = np.roll(vecs[i2], n2, axis=0)

                factor = phase*integral_factors(e1, e2, t)
                product = np.outer(psi[i1], vecsstar[i2, h.n_to_i(-n2, nz)])

                for c in xrange(0, npm):
                    du[c, :, :] += expectation_value(dk[c], v1, v2)*factor*product

    return du


def integral_factors(e1, e2, t):
    if e1 == e2:
        return -1.0j*np.exp(-1j*t*e1)
    else:
        return (np.exp(-1j*t*e1)-np.exp(-1j*t*e2))/(t*(e1-e2))


def expectation_value(dk, v1, v2):
    a = np.transpose(v1.flatten())
    b = v2.flatten()

    return np.dot(np.dot(a, dk), b)
