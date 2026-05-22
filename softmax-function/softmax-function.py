import numpy as np

def softmax(x):
    """
    Compute the softmax of input x.
    Works for 1D or 2D NumPy arrays.
    For 2D, compute row-wise softmax.
    """
    x = np.asarray(x)
    if (np.ndim(x) == 1):
        x_max = np.max(x)
        x = x - x_max
        x = np.exp(x)
        return x / np.sum(x)

    x = x - np.max(x, axis=1,keepdims = True)
    x = np.exp(x)
    return x / np.sum(x,axis=1,keepdims=True)
    
   
    