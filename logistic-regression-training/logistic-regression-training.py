import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    N,n = np.shape(X)

    w = np.zeros(n,)
    b = 0.0
    
    for i in range(steps):
        z = X@w + b
        p = _sigmoid(z)
        error = p - y
    
        dw = 1/N * (X.T @ error) 
        db = 1/N * np.sum(error)

        w = w - lr * dw
        b = b - lr * db
    
    # Write code here
    return w, float(b)