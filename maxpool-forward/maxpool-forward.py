def maxpool_forward(X, pool_size, stride):
    """
    Compute the forward pass of 2D max pooling.
    """
    row = len(X)
    col = len(X[0])
    out_h =  (row - pool_size) // stride + 1
    out_w =  (col - pool_size) // stride + 1
    output =[]
    for i in range(out_h):
        row = []
        for j in range(out_w):
            current = -1e15
            for a in range(pool_size):
                for b in range(pool_size):
                  current = max(current,X[i* stride + a][j*stride + b])
            row.append(current)
        output.append(row)
    return output