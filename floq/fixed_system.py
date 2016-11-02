import numpy as np
import errors as er

class FixedSystem(object):
    """
    Class that defines one specific instance of a Floquet problem, i.e. some system for which the dynamics have to be calculated

    Has the following attributes:
    - hf, the Fourier transformed Hamiltonian (ndarray, square)
    - dhf, its derivative with respect to the controls (ndarray of square ndarrays)
    - parameters, an instance of FixedSystemParameters

    - omega: The frequency associated with the period T of the control pulse
    - t: Control duration
    - nz: Number of Fourier components taken into account for K, and of Brillouin zones in the resulting set of eigenvalues
    Derived from nz: nz_min/max, the cutoff for the integers labelling the Fourier components
    - decimals: The number of decimals used for rounding
    
    The following parameters are inferred from hf and dhf during initialisation:
    - dim, the size of the Hilbert space
    - nc, the number of components in Hf
    - np, the number of control parameters
    - k_kim = dim*nz: the size of K
    """

    def __init__(self,hf,dhf,nz,omega,t,decimals=10):
        self._nz = 0.0
        self._dim = 0.0

        self.hf = hf
        self.dhf = dhf

        self.nz = nz
        self.omega = omega
        self.t = t
        self.decimals = decimals

        # Inferred parameters
        self.dim = hf.shape[1]
        self.nc = hf.shape[0]
        self.np = dhf.shape[0]

    @property
    def nz(self):
        """
        Number of Brillouin zones in the extended space
        """
        return self._nz

    @nz.setter
    def nz(self, value):
        if value%2 == 0:
            raise er.UsageError("Number of Fourier components in the extended space (nz) cannot be even.")

        self._nz = value    
        self.k_dim = self.dim*value
        self.nz_max = (value-1)/2
        self.nz_min = -(value-1)/2

    @property
    def dim(self):
        """
        Physical size of Hilbert Space
        """
        return self._dim

    @dim.setter
    def dim(self, value):
        self._dim = value
        self.k_dim = self.nz*value


class FixedSystemParameters(object):
    """
    Class to pass to functions in evolution.py without having to
    specify a full FixedSystem -- retained mainly for testing
    """
    def __init__(self,dim=None,nz=None,nc=None,np=None,omega=None,t=None,decimals=10):
        
        self._dim = None
        self._nz = None

        self.dim = dim
        self.nz = nz
        
        self.nc = nc
        self.np = np

        self.omega = omega
        self.t = t
        
        self.decimals = decimals
        

    @property
    def nz(self):
        """
        Number of Brillouin zones in the extended space
        """
        return self._nz

    @nz.setter
    def nz(self, value):
        self._nz = value
        if self._dim != None:
            self.k_dim = self.dim*value
            self.nz_max = (value-1)/2
            self.nz_min = -(value-1)/2

    @property
    def dim(self):
        """
        Physical size of Hilbert Space
        """
        return self._dim

    @dim.setter
    def dim(self, value):
        self._dim = value
        if self.nz != None:
            self.k_dim = self.nz*value