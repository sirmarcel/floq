import numpy as np
import errors as er


class FixedSystem(object):
    """
    Class that defines one specific instance of a Floquet problem, i.e.
    some system for which the dynamics have to be calculated

    Has the following attributes:
    - hf: the Fourier transformed Hamiltonian (ndarray, square)
    - dhf: its derivative with respect to the controls (ndarray of square ndarrays)
    - params: an instance of FixedSystemParameters

    - omega: The frequency associated with the period T of the control pulse
    - t: Control duration
    - nz: Number of Fourier components taken into account for K, and of Brillouin
          zones in the resulting set of eigenvalues

    Derived from nz: nz_min/max, the cutoff for the integers labelling the Fourier components
    - decimals: The number of decimals used for rounding

    The following parameters are inferred from hf and dhf during initialisation:
    - dim: the size of the Hilbert space
    - nc: the number of components in Hf
    - np: the number of control parameters
    """

    def __init__(self, hf, dhf, nz, omega, t, decimals=10, sparse=True):
        self.hf = hf
        self.dhf = dhf

        # Inferred parameters
        dim = hf.shape[1]
        nc = hf.shape[0]
        np = dhf.shape[0]

        self.params = FixedSystemParameters(dim, nz, nc, np, omega, t, decimals)

    def __eq__(self, other):
        assert isinstance(other, FixedSystem)
        hf_same = np.array_equal(self.hf, other.hf)
        dhf_same = np.array_equal(self.dhf, other.dhf)
        params_same = (self.params.nz, self.params.omega, self.params.t) \
            == (other.params.nz, other.params.omega, other.params.t)

        return hf_same and dhf_same and params_same



class DummyFixedSystem(FixedSystem):
    """
    A dummy FixedSystem that can be initialised with arbitrary dimensions
    without specifying hf and dhf (mainly for testing)
    """

    def __init__(self, **kwargs):
        self.params = FixedSystemParameters.optional(**kwargs)


class FixedSystemParameters(object):
    """
    Hold parameters for a FixedSystem

    - dim: the size of the Hilbert space
    - nz: Number of Fourier components taken into account for K, and of Brillouin 
          zones in the resulting set of eigenvalues

    Derived from nz:
        - nz_min/max, the cutoff for the integers labelling the Fourier components
        - k_kim = dim*nz: the size of K

    - nc: number of components in Hf
    - np: number of control parameters
    - omega: The frequency associated with the period T of the control pulse
    - t: Control duration
    - decimals: The number of decimals used for rounding when finding unique eigenvalues
    - sparse: If True, a sparse eigensolver will be used
              -- unless working with very small systems with < 15 zones, this should be True
    """

    def __init__(self, dim, nz, nc, np, omega, t, decimals, sparse=True):
        self._dim = 0
        self._nz = 0
        self._nc = 0

        self.dim = dim
        self.nz = nz

        self.nc = nc
        self.np = np

        self.omega = omega
        self.t = t

        self.decimals = decimals
        self.sparse = sparse

    @classmethod
    def optional(self, dim=0, nz=1, nc=1, np=0, omega=1, t=1, decimals=10, sparse=True):
        """
        Class method to instantiate FixedSystemParameters without specifying
        the full set of parameters -- only needed for testing!
        """
        return FixedSystemParameters(dim, nz, nc, np, omega, t, decimals, sparse)


    @property
    def nz(self):
        return self._nz

    @nz.setter
    def nz(self, value):
        if value % 2 == 0:
            raise er.UsageError("Number of Fourier components in the \
                  extended space (nz) cannot be even.")

        self._nz = value
        self.k_dim = self.dim*value
        self.nz_max = (value-1)/2
        self.nz_min = -(value-1)/2


    @property
    def nc(self):
        return self._nc

    @nc.setter
    def nc(self, value):
        if value % 2 == 0:
            raise er.UsageError("Number of Fourier components of H \
                  cannot be even.")

        self._nc = value
        self.nc_max = (value-1)/2
        self.nc_min = -(value-1)/2

    @property
    def dim(self):
        return self._dim

    @dim.setter
    def dim(self, value):
        self._dim = value
        self.k_dim = self.nz*value
