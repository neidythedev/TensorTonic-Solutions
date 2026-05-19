import numpy as np
import math
def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.
    """
    #n = number of class
    m, n = np.shape(y_pred)
    y_pred, y_true = np.asarray(y_pred), np.asarray(y_true)

    correct_probs = y_pred[np.arange(m), y_true]

    
    return -np.mean(np.log(correct_probs + 1e-15))