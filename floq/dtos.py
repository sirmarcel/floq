import numpy as np

class FloquetProblemParameters(object):
    """Data transfer object to hold the parameters of a Floquet problem"""
    def __init__(self,dim,zones,omega,tgst):
        self.dim = dim
        self.zones = zones
        self.omega = omega
        self.t = t
        
        self.k_dim = dim*zones