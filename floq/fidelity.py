import numpy as np

def trace_overlap(system,u,target):
    hs = hilbert_schmidt_product(u,target)
    return hs.real/system.params.dim

def hilbert_schmidt_product(a,b):
    a = np.matrix(a)
    return np.trace(np.dot(a.getH(),b))