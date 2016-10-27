import numpy as np
import floq.helpers as h
import floq.blockmatrix as bm
import floq.dtos as dtos
import itertools
import copy

class EigenvalueNumberError(Exception):
    def __init__(self, all_vals, unique_vals):
        self.all_vals, self.unique_vals = all_vals, unique_vals
    def __str__(self):
        return "Number of eigenvalues of K does not match dimension of the the Hilbert space. \n All vals: " + repr(self.all_vals) + "\n 'Unique' vals: " + repr(self.unique_vals)


def do_evolution(hf,p):
    """
    Calculate the time evolution operator U
    given a Fourier transformed Hamiltonian Hf
    """
    k = build_k(hf,p)

    vals,vecs = find_eigensystem(k,p)

    phi = calculate_phi(vecs)
    psi = calculate_psi(vecs,p)

    return calculate_u(phi,psi,vals,p)

def do_evolution_with_derivatives(hf,dhf,p):
    """
    Calculate the time evolution operator U
    given a Fourier transformed Hamiltonian Hf,
    as well as its derivative dU given dHf
    """
    k = build_k(hf,p)

    vals,vecs = find_eigensystem(k,p)

    phi = calculate_phi(vecs)
    psi = calculate_psi(vecs,p)

    u = calculate_u(phi,psi,vals,p)
    du = calculate_du(dhf,psi,vals,vecs,p)

    return [u,du]


def build_k(hf,p):
    """
    Build the Floquet-Hamiltonian K 
    from the Fourier transform of the system Hamiltonian
    """
    
    hf_max = (p.nc-1)/2 # maximal frequency in Hf

    k = np.zeros([p.k_dim,p.k_dim])

    # Assemble K by placing each component of Hf in turn
    # The nc lie on diagonals, with Hf(0) on the main diagonal
    # The first row is therefore essentially Hf(0) Hf(1) ... Hf(hf_max) 0 0 0 ...
    # The last row is then ... 0 0 0 Hf(-hf_max) ... Hf(0)
    for n in xrange(-hf_max,hf_max+1):
        start_row = max(0,n) # n < 0, start at row 0
        start_col = max(0,-n) # n > 0, start at col 0
        
        stop_row = min((p.nz-1)+n,p.nz-1) # if n > 0, start from the last col
        stop_col = min((p.nz-1)-n,p.nz-1) # if n < 0, start from the last row

        row = start_row
        col = start_col

        hf_of_n = hf[h.num_to_i(n,p.nc)]

        while row <= stop_row and col <= stop_col:
            if n == 0:
                block = hf_of_n + np.identity(p.dim)*p.omega*h.i_to_num(row,p.nz)
                bm.set_block_in_matrix(block,k,p.dim,p.nz,row,col)
            else:
                bm.set_block_in_matrix(hf_of_n,k,p.dim,p.nz,row,col)

            row += 1
            col += 1

    return k

def build_dk(dhf,p):
    p2 = copy.copy(p)
    p2.omega = 0.0

    return np.array([build_k(dhf[i],p2) for i in xrange(0,p.np)])


def find_eigensystem(k,p):
    """
    Find eigenvalues and eigenvectors for k,
    identify the dim unique ones,
    return them in a segmented form
    """
    vals, vecs = np.linalg.eig(k)
    vals = vals.real.astype(np.float64,copy=False)

    unique_vals = find_unique_vals(vals,p)

    vals = vals.round(p.decimals)
    indices_unique_vals = [np.where(vals == eva)[0][0] for eva in unique_vals]
    
    unique_vecs = np.array([vecs[:,i] for i in indices_unique_vals],dtype='complex128')
    unique_vecs = separate_components(unique_vecs,p.nz)
    
    return [unique_vals,unique_vecs]

def find_unique_vals(vals,p):
    """
    In the list of values supplied, find the set of dim 
    e_i that fulfil (e_i - e_j) mod omega != 0 for all i,j,
    and that lie closest to 0.
    """

    # cut off the first and last zone to prevent finite-size effects
    if p.nz > 4:
        vals = np.delete(vals,np.s_[0:p.dim])
        vals = np.delete(vals,np.s_[-p.dim:])
    
    mod_vals = np.mod(vals,p.omega).round(decimals=p.decimals) # round to suppress floating point issues
    
    unique_vals = np.unique(mod_vals) 

    # the unique_vals are ordered and >= 0, but we'd rather have them clustered around 0
    should_be_negative = np.where(unique_vals>p.omega/2.)
    unique_vals[should_be_negative] = (unique_vals[should_be_negative]-p.omega).round(p.decimals)

    if unique_vals.shape[0] != p.dim:
        raise EigenvalueNumberError(vals,unique_vals)
    else:
        return np.sort(unique_vals)

def separate_components(vecs,n):
    """
    Separate each vector in vecs into n sub-arrays
    """
    return np.array([np.split(eva,n) for eva in vecs])


def calculate_phi(vecs):
    """
    For the p.dim eigenvectors indexed by k, find the sum
    over all frequency components:
    |phi_k> = \sum_nu <nu | xi_k> 
    """
    return np.array([np.sum(eva,axis=0) for eva in vecs])

def calculate_psi(vecs,p):
    """
    For the eigenvectors indexed by k,
    supplied in a split form,
    find the sum weighted with the Fourier 
    factors exp(- i num omega t)
    """
    psi = np.zeros([p.dim,p.dim],dtype='complex128')

    for k in xrange(0,p.dim):
        partial = np.zeros(p.dim,dtype='complex128')
        for i in xrange(0,p.nz):
            num = h.i_to_num(i,p.nz)
            partial += np.exp(1j*p.omega*p.t*num)*vecs[k][i]
        psi[k,:] = partial

    return psi


def calculate_u(phi,psi,energies,p):   
    u = np.zeros([p.dim,p.dim],dtype='complex128')

    for k in xrange(0,p.dim):
        u += np.exp(-1j*p.t*energies[k])*np.outer(psi[k],np.conj(phi[k]))

    return u


def calculate_du(dhf,psi,vals,vecs,p):
    du = np.zeros([p.np,p.dim,p.dim],dtype='complex128')
    dk = build_dk(dhf,p)

    # (i1,n1) & (i2,n2) iterate over the full spectrum of k:
    # i1, i2: unique eigenvalues/-vectors in 0th Brillouin zone
    # n1, n2: related vals/vecs derived by shifting with those offsets
    uniques = xrange(0,p.dim)
    offsets = xrange(p.nz_min,p.nz_max+1)

    for c in xrange(0,p.np):    
        for i1,i2 in itertools.product(uniques,uniques):
            for n1,n2 in itertools.product(offsets,offsets):
                e1 = vals[i1] + n1*p.omega
                e2 = vals[i2] + n2*p.omega            

                v1 = np.roll(vecs[i1],n1,axis=0)
                v2 = np.roll(vecs[i2],n2,axis=0)
                
                du[c,:,:] += p.t*np.exp(1j*p.omega*p.t*n1)*efacs(e1,e2,p)*expectation(dk[c],v1,v2,p)*np.outer(psi[i1],np.conj(vecs[i2,h.num_to_i(-n2,p.nz)]))

    return du
    
def efacs(e1,e2,p):
    if e1 == e2:
        return -1.0j*np.exp(-1j*p.t*e1)
    else:
        return (np.exp(-1j*p.t*e1)-np.exp(-1j*p.t*e2))/(p.t*(e1-e2))

def expectation(dk,v1,v2,p):
    a = np.conj(np.transpose(v1.flatten()))
    b = v2.flatten()

    return np.dot(np.dot(a,dk),b)