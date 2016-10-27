import numpy as np
import floq.helpers as h
import floq.blockmatrix as bm
import floq.dtos as dtos
import itertools
import copy

class EigenvalueNumberError(Exception):
    def __init__(self, all_evas, unique_evas):
        self.all_evas, self.unique_evas = all_evas, unique_evas
    def __str__(self):
        return "Number of eigenvalues of K does not match dimension of the the Hilbert space. \n All evas: " + repr(self.all_evas) + "\n 'Unique' evas: " + repr(self.unique_evas)


def do_evolution(hf,p):
    """
    Calculate the time evolution operator U
    given a Fourier transformed Hamiltonian Hf
    """
    k = build_k(hf,p)

    evas,eves = find_eigensystem(k,p)

    phi = calculate_phi(eves)
    psi = calculate_psi(eves,p)

    return calculate_u(phi,psi,evas,p)

def do_evolution_with_derivatives(hf,dhf,p):
    """
    Calculate the time evolution operator U
    given a Fourier transformed Hamiltonian Hf,
    as well as its derivative dU given dHf
    """
    k = build_k(hf,p)

    evas,eves = find_eigensystem(k,p)

    phi = calculate_phi(eves)
    psi = calculate_psi(eves,p)

    u = calculate_u(phi,psi,evas,p)
    du = calculate_du(dhf,psi,evas,eves,p)

    return du


def build_k(hf,p):
    """
    Build the Floquet-Hamiltonian K 
    from the Fourier transform of the system Hamiltonian
    """
    
    hf_cutoff = (p.components-1)/2

    k = np.zeros([p.k_dim,p.k_dim])

    # Assemble K by placing each component of Hf in turn
    # The components lie on diagonals, with Hf(0) on the main diagonal
    # The first row is therefore essentially Hf(0) Hf(1) ... Hf(hf_cutoff) 0 0 0 ...
    # The last row is then ... 0 0 0 Hf(-hf_cutoff) ... Hf(0)
    for num in xrange(-hf_cutoff,hf_cutoff+1):
        start_row = max(0,num) # num < 0, start at row 0
        start_col = max(0,-num) # num > 0, start at col 0
        
        stop_row = min((p.zones-1)+num,p.zones-1) # if num > 0, start from the last col
        stop_col = min((p.zones-1)-num,p.zones-1) # if num < 0, start from the last row

        row = start_row
        col = start_col

        hf_of_num = hf[h.num_to_i(num,p.components)]

        while row <= stop_row and col <= stop_col:
            if num == 0:
                block = hf_of_num + np.identity(p.dim)*p.omega*h.i_to_num(row,p.zones)
                bm.set_block_in_matrix(block,k,p.dim,p.zones,row,col)
            else:
                bm.set_block_in_matrix(hf_of_num,k,p.dim,p.zones,row,col)

            row += 1
            col += 1

    return k

def build_dk(dhf,p):
    p2 = copy.copy(p)
    p2.omega = 0.0

    return np.array([build_k(dhf[i],p2) for i in xrange(0,p.controls)])


def find_eigensystem(k,p):
    """
    Find eigenvalues and eigenvectors for k,
    identify the dim unique ones,
    return them in a segmented form
    """
    evas, eves = np.linalg.eig(k)
    evas = evas.real.astype(np.float64,copy=False)

    unique_evas = find_unique_evas(evas,p)

    evas = evas.round(p.decimals)
    indices_unique_evas = [np.where(evas == eva)[0][0] for eva in unique_evas]
    
    unique_eves = np.array([eves[:,i] for i in indices_unique_evas],dtype='complex128')
    unique_eves = separate_components(unique_eves,p.zones)
    
    return [unique_evas,unique_eves]

def find_unique_evas(evas,p):
    """
    In the list of values supplied, find the set of dim 
    e_i that fulfil (e_i - e_j) mod omega != 0 for all i,j,
    and that lie closest to 0.
    """

    # cut off the first and last zone to prevent finite-size effects
    if p.zones > 4:
        evas = np.delete(evas,np.s_[0:p.dim])
        evas = np.delete(evas,np.s_[-p.dim:])
    
    mod_evas = np.mod(evas,p.omega).round(decimals=p.decimals) # round to suppress floating point issues
    
    unique_evas = np.unique(mod_evas) 

    # the unique_evas are ordered and >= 0, but we'd rather have them clustered around 0
    should_be_negative = np.where(unique_evas>p.omega/2.)
    unique_evas[should_be_negative] = (unique_evas[should_be_negative]-p.omega).round(p.decimals)

    if unique_evas.shape[0] != p.dim:
        raise EigenvalueNumberError(evas,unique_evas)
    else:
        return np.sort(unique_evas)

def separate_components(eves,n):
    """
    Separate each vector in eves into n sub-arrays
    """
    return np.array([np.split(eva,n) for eva in eves])


def calculate_phi(eves):
    """
    For the p.dim eigenvectors indexed by k, find the sum
    over all frequency components:
    |phi_k> = \sum_nu <nu | xi_k> 
    """
    return np.array([np.sum(eva,axis=0) for eva in eves])

def calculate_psi(eves,p):
    """
    For the eigenvectors indexed by k,
    supplied in a split form,
    find the sum weighted with the Fourier 
    factors exp(- i num omega t)
    """
    psi = np.zeros([p.dim,p.dim],dtype='complex128')

    for k in xrange(0,p.dim):
        partial = np.zeros(p.dim,dtype='complex128')
        for i in xrange(0,p.zones):
            num = h.i_to_num(i,p.zones)
            partial += np.exp(1j*p.omega*p.t*num)*eves[k][i]
        psi[k,:] = partial

    return psi


def calculate_u(phi,psi,energies,p):
    """
    Given phi and psi,
    calculate U(t)
    """
    
    u = np.zeros([p.dim,p.dim],dtype='complex128')

    for k in xrange(0,p.dim):
        u += np.exp(-1j*p.t*energies[k])*np.outer(psi[k],np.conj(phi[k]))

    return u


def calculate_du(dhf,psi,evas,eves,p):
    du = np.zeros([p.controls,p.dim,p.dim],dtype='complex128')
    dk = build_dk(dhf,p)

    for control in xrange(0,p.controls):
        for i1,i2 in itertools.product(xrange(0,p.dim),xrange(0,p.dim)):
            for n1,n2 in itertools.product(xrange(-p.zones_cutoff,p.zones_cutoff+1),xrange(-p.zones_cutoff,p.zones_cutoff+1)):
                e1 = evas[i1] + n1*p.omega
                e2 = evas[i2] + n2*p.omega            

                v1 = np.roll(eves[i1],n1,axis=0)
                v2 = np.roll(eves[i2],n2,axis=0)

                temp = np.exp(1j*p.omega*p.t*n1)*e(e1,e2,p)*expectation(dk[control],v1,v2,p)*np.outer(psi[i1],np.conj(eves[i2,h.num_to_i(-n2,p.zones),:]))
                
                du[control,:,:] += temp

    return du
    
def e(e1,e2,p):
    if e1 == e2:
        return -1.0j*np.exp(-1j*p.t*e1)
    else:
        return (np.exp(-1j*p.t*e1)-np.exp(-1j*p.t*e2))/(e1-e2)

def expectation(dk,v1,v2,p):
    a = np.conj(np.transpose(v1.flatten()))
    b = v2.flatten()

    return np.dot(np.dot(a,dk),b)