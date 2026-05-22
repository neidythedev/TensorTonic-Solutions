import numpy as np

def tanh(x):
    """
    Implement Tanh activation function.
    """
    # Write code here
    x = np.asarray(x)
   
    return (np.exp(x) - np.exp(-1 * x))/(np.exp(x)+np.exp(x*-1))
    
    pass