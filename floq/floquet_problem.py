import numpy as np

class FloquetProblemParameters(object):
    """
    Data transfer object to hold the parameters of a Floquet problem.

    -> indicates quantities that are computed on initialisation if the required parameters have been passed

    dim: Size of the Hilbert space of the physical system
    nz: Number of Fourier components taken into account for K, and of Brillouin zones in the resulting set of eigenvalues
    -> nz_min/max: the cutoff for the integers labelling the Fourier components
    -> k_kim = dim*nz: the size of K

    nc: Number of Fourier components of Hf
    np: Number of control parameters in Hf

    omega: The frequency associated with the period T of the control pulse
    t: Control duration

    decimals: Number of decimals used for internal rounding (defaults to 10)

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
            