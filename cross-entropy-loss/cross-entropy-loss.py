import numpy as np
import math
def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.
    """
    #n = number of class
    m, n = np.shape(y_pred)
    y_pred, y_true = np.asarray(y_pred), np.asarray(y_true)
    loss_total = 0
    for i in range(m):
        
        
        loss_total = loss_total - math.log(y_pred[i,y_true[i]])

    
    return loss_total/m