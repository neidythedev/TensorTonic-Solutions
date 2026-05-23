def linear_layer_forward(X, W, b):
    """
    Compute the forward pass of a linear (fully connected) layer.
    """

    n_sample = len(X)
    n_feature = len(X[0])
    n_output = len(b)
    Y = []
    for i in range (n_sample):
        row = []
        for j in range (n_output):
            temp = 0
            for k in range(n_feature):
                temp += X[i][k]*W[k][j]
            temp += b[j]
            row.append(temp)
        Y.append(row)
    return Y
    