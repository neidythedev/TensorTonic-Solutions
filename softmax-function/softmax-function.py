import numpy as np

def softmax(x):
    """
    Compute the softmax of input x.
    Works for 1D or 2D NumPy arrays.
    For 2D, compute row-wise softmax.
    """
    
    z = np.asarray(x)
    #1d
    if (np.ndim(z) == 1):
        z = z - np.max(z)
        p = np.exp(z)
        return p/np.sum(p)
        
    #2d
    z = z - np.max(z, axis =1, keepdims=True)
    p = np.exp(z)
    return p/np.sum(p, axis=1, keepdims=True)
    