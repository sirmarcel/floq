import numpy as np

class FloquetProblemParameters(object):
    """Data transfer object to hold the parameters of a Floquet problem"""
    def __init__(self,dim=None,zones=None,components=None,omega=None,t=None,decimals=10):
        self.dim = dim
        self.zones = zones
        self.components = components
        self.omega = omega
        self.t = t
        self.decimals = decimals
        
        self.k_dim = dim*zones