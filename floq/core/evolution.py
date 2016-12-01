import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as la
from floq.helpers.index import n_to_i, i_to_n
from floq.helpers.numpy_replacements import numba_outer, numba_zeros
import floq.helpers.blockmatrix as bm
import floq.fixed_system as fs
import floq.errors as errors
import itertools
import cmath
from numba import autojit


def get_u(hf, params):
    """
    Calculate the time evolution operator U,
    given a Fourier transformed Hamiltonian Hf
    and the parameters of the problem
    """
    return get_u_and_eigensystem(hf, params)[0]


def get_u_and_du(hf, dhf, params):
    """
    Calculate the time evolution operator U
    given a Fourier transformed Hamiltonian Hf,
    as well as its derivative dU given dHf,
    and the parameters of the problem
    """
    u, vals, vecs, phi, psi = get_u_and_eigensystem(hf, params)

    du = get_du_from_eigensystem(dhf, psi, vals, vecs, params)

    return [u, du]


def get_u_and_eigensystem(hf, params):
    """
    Calculate the time evolution operator U,
    given a Fourier transformed Hamiltonian Hf
    and the parameters of the problem, and return
    it as well as the intermediary results
    """
    k = assemble_k(hf, params)

    vals, vecs = find_eigensystem(k, params)

    phi = calculate_phi(vecs)
    psi = calculate_psi(vecs, params)

    return [calculate_u(phi, psi, vals, params), vals, vecs, phi, psi]


def get_du_from_eigensystem(dhf, psi, vals, vecs, params):
    dk = assemble_dk(dhf, params)
    du = calculate_du(dk, psi, vals, vecs, params)

    return du


def assemble_k(hf, p):
    # assemble the Floquet Hamiltonian K from
    # the components of the Fourier-transformed Hamiltonian
    return numba_assemble_k(hf, p.dim, p.k_dim, p.nz, p.nc, p.omega)

@autojit(nopython=True)
def numba_assemble_k(hf, dim, k_dim, nz, nc, omega):
    hf_max = (nc-1)/2
    k = numba_zeros((k_dim, k_dim))

    # Assemble K by placing each component of Hf in turn, which
    # for a fixed Fourier index lie on diagonals, with 0 on the
    # main diagonal, positive numbers on the right and negative on the left
    #
    # The first row is therefore essentially Hf(0) Hf(1) ... Hf(hf_max) 0 0 0 ...
    # The last row is then ... 0 0 0 Hf(-hf_max) ... Hf(0)
    # Note that the main diagonal acquires a factor of omega*identity*(row/column number)

    for n in range(-hf_max, hf_max+1):
        start_row = max(0, n)  # if n < 0, start at row 0
        start_col = max(0, -n)  # if n > 0, start at col 0

        stop_row = min((nz-1)+n, nz-1)
        stop_col = min((nz-1)-n, nz-1)

        row = start_row
        col = start_col

        current_component = hf[n_to_i(n, nc)]

        while row <= stop_row and col <= stop_col:
            if n == 0:
                block = current_component + np.identity(dim)*omega*i_to_n(row, nz)
                bm.set_block_in_matrix(block, k, dim, nz, row, col)
            else:
                bm.set_block_in_matrix(current_component, k, dim, nz, row, col)

            row += 1
            col += 1

    return k


def assemble_dk(dhf, p):
    # assemble the derivative of the Floquet Hamiltonian K from
    # the components of the derivative of the Fourier-transformed Hamiltonian
    # This is equivalent to K, with Hf -> d HF and omega -> 0.
    return numba_assemble_dk(dhf, p.np, p.dim, p.k_dim, p.nz, p.nc)

@autojit(nopython=True)
def numba_assemble_dk(dhf, npm, dim, k_dim, nz, nc):
    dk = np.empty((npm, k_dim, k_dim), dtype=np.complex128)
    for c in range(npm):
        dk[c, :, :] = numba_assemble_k(dhf[c], dim, k_dim, nz, nc, 0.0)

    return dk



def find_eigensystem(k, p):
    # Find eigenvalues and eigenvectors for k,
    # identify the dim unique ones,
    # return them in a segmented form
    vals, vecs = compute_eigensystem(k, p)

    unique_vals = find_unique_vals(vals, p)

    # rounding is necessary, since full floating-point numbers are seldom equal
    vals = vals.round(p.decimals)
    indices_unique_vals = [np.where(vals == eva)[0][0] for eva in unique_vals]

    unique_vecs = np.array([vecs[:, i] for i in indices_unique_vals])
    unique_vecs = separate_components(unique_vecs, p.nz)

    return [unique_vals, unique_vecs]


def compute_eigensystem(k, p):
    # Find eigenvalues and eigenvectors of k,
    # using the method specified in the parameters
    # (sparse is almost always faster, and is the default)
    if p.sparse:
        k = sp.csc_matrix(k)

        number_of_eigs = min(2*p.dim, p.k_dim)

        # find number_of_eigs eigenvectors/-values around 0.0
        # -> trimming/sorting the eigensystem is NOT necessary
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
    # -> these sub-arrays are the Fourier components of the state
    return np.array([np.split(eva, n) for eva in vecs])



@autojit(nopython=True)
def calculate_phi(vecs):
    # Given an array of eigenvectors vecs,
    # sum over Fourier components in each
    dim = vecs.shape[0]
    phi = np.empty((dim, dim), dtype=np.complex128)
    for i in range(dim):
        phi[i] = numba_sum_components(vecs[i], dim)
    return phi

@autojit(nopython=True)
def numba_sum_components(vec, dim):
    n = vec.shape[0]
    result = numba_zeros(dim)
    for i in range(n):
        result += vec[i]
    return result



def calculate_psi(vecs, p):
    # Given an array of eigenvectors vecs,
    # sum over all Fourier components in each,
    # weighted by exp(- i omega t n), with n
    # being the Fourier index of the component

    return numba_calculate_psi(vecs, p.dim, p.nz, p.omega, p.t)

@autojit(nopython=True)
def numba_calculate_psi(vecs, dim, nz, omega, t):
    psi = numba_zeros((dim, dim))

    for k in range(0, dim):
        partial = numba_zeros(dim)
        for i in range(0, nz):
            num = i_to_n(i, nz)
            partial += np.exp(1j*omega*t*num)*vecs[k][i]
        psi[k, :] = partial

    return psi



def calculate_u(phi, psi, energies, p):
    u = np.zeros([p.dim, p.dim], dtype='complex128')
    t = p.t

    for k in xrange(0, p.dim):
        u += np.exp(-1j*t*energies[k])*np.outer(psi[k], np.conj(phi[k]))

    return u



def calculate_du(dk, psi, vals, vecs, p):
    # Given the eigensystem of K, and its derivative,
    # perform the computations to get dU.
    #
    # This routine is optimised and quite hard to read, I recommend
    # taking a look in the museum, which contains functionally equivalent,
    # but much more readable versions.

    dim = p.dim
    nz_max = p.nz_max
    nz = p.nz
    npm = p.np
    omega = p.omega
    t = p.t

    vecsstar = np.conj(vecs)
    factors = calculate_factors(dk, nz, nz_max, dim, npm, vals, vecs, vecsstar, omega, t)
    return assemble_du(nz, nz_max, dim, npm, factors, psi, vecsstar)


def calculate_factors(dk, nz, nz_max, dim, npm, vals, vecs, vecsstar, omega, t):
    # Factors in the sum for dU that only depend on dn=n1-n2, and therefore
    # can be computed more efficiently outside the "full" loop
    factors = np.empty([npm, 2*nz+1, dim, dim], dtype=np.complex128)

    for dn in xrange(-nz_max*2, 2*nz_max+1):
        idn = n_to_i(dn, 2*nz)
        for i1 in xrange(0, dim):
            for i2 in xrange(0, dim):
                v1 = np.roll(vecsstar[i1], dn, axis=0)  # not supported by numba!
                for c in xrange(0, npm):
                    factors[c, idn, i1, i2] = (integral_factors(vals[i1], vals[i2], dn, omega, t) *
                                               expectation_value(dk[c], v1, vecs[i2]))

    return factors


@autojit(nopython=True)
def assemble_du(nz, nz_max, dim, npm, alphas, psi, vecsstar):
    # Execute the sum defining dU, taking pre-computed factors into account
    du = numba_zeros((npm, dim, dim))

    for n2 in range(-nz_max, nz_max+1):
        for i1 in range(0, dim):
            for i2 in range(0, dim):
                product = numba_outer(psi[i1], vecsstar[i2, n_to_i(-n2, nz)])
                for n1 in range(-nz_max, nz_max+1):
                    idn = n_to_i(n1-n2, 2*nz)
                    for c in xrange(0, npm):
                        du[c] += alphas[c, idn, i1, i2]*product

    return du


@autojit(nopython=True)
def integral_factors(e1, e2, dn, omega, t):
    if e1 == e2 and dn == 0:
        return -1.0j*cmath.exp(-1j*t*e1)*t
    else:
        return (cmath.exp(-1j*t*e1)-cmath.exp(-1j*t*(e2-omega*dn)))/((e1-e2+omega*dn))


@autojit(nopython=True)
def expectation_value(dk, v1, v2):
    # Computes <v1|dk|v2>, assuming v1 is already conjugated

    # v1 and v2 are split into Fourier components,
    # we undo that here
    a = v1.flatten()
    b = v2.flatten()

    return np.dot(np.dot(a, dk), b)
