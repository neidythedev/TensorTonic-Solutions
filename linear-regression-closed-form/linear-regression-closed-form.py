import numpy as np

def linear_regression_closed_form(X, y):
    """
    Compute the optimal weight vector using the normal equation.
    """
    # Write code here
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    trans_X = np.transpose(X)
    A = np.linalg.inv(np.dot(trans_X,X))
    B = np.dot(trans_X,y)

    return np.dot(A,B)
    