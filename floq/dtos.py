import numpy as np

class FloquetProblemParameters(object):
    """Data transfer object to hold the parameters of a Floquet problem"""
    def __init__(self,dim=None,nz=None,nc=None,np=None,omega=None,t=None,decimals=10):
        self.dim = dim # Size of the Hilbert space of the physical system
        self.nz = nz # Number of Brillouin zones in the extended space
        self.nc = nc # Number of frequency components of Hf
        self.np = np # Number of control parameters

        self.omega = omega # The frequency associated with the period of the control signal, T = 2 Pi / omega
        self.t = t # Control duration
        
        self.decimals = decimals # Number of decimals used for internal rounding
        
        if dim != None and nz != None:
            self.k_dim = dim*nz # Size of the extended Hilbert + Fourier space

        if nz != None:
            self.nz_max = (nz-1)/2
            self.nz_min = -(nz-1)/2