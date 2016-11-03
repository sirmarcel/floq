import numpy as np

def trace_overlap(system,u,target):
    hs = hilbert_schmidt_product(target,u)
    return hs.real/system.params.dim

def d_trace_overlap(system,u,dus,target):
    return np.array([trace_overlap(system,du,target) for du in dus])

def hilbert_schmidt_product(a,b):
    a = np.matrix(a)
    return np.trace(np.dot(a.getH(),b))