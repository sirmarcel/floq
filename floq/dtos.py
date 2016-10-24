import numpy as np

class FloquetProblemParameters(object):
    """Data transfer object to hold the parameters of a Floquet problem"""
    def __init__(self,dim=None,zones=None,components=None,controls=None,omega=None,t=None,decimals=10):
        self.dim = dim # Size of the Hilbert space of the physical system
        self.zones = zones # Number of Brillouin zones in the extended space
        self.components = components # Number of frequency components of Hf
        self.controls = controls # Number of control parameters

        self.omega = omega # The frequency associated with the period of the control signal, T = 2 Pi / omega
        self.t = t # Control duration
        
        self.decimals = decimals # Number of decimals used for internal rounding
        
        self.k_dim = dim*zones # Size of the extended Hilbert + Fourier space