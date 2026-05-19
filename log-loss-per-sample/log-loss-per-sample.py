import math

def log_loss(y_true, y_pred, eps=1e-15):
    """
    Compute per-sample log loss.
    """
    loss_list = []
    
    for i in range(len(y_pred)):
        y_pred[i] = max(eps, min(1-eps,y_pred[i]))
        loss =  -(y_true[i] * math.log(y_pred[i]) + (1-y_true[i])* math.log(1-y_pred[i]))
        loss_list.append(loss)
    return loss_list