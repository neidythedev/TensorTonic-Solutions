import numpy as np

def rnn_step_backward(dh, cache):
    """
    Returns:
        dx_t: gradient wrt input x_t      (shape: D,)
        dh_prev: gradient wrt previous h (shape: H,)
        dW: gradient wrt W               (shape: H x D)
        dU: gradient wrt U               (shape: H x H)
        db: gradient wrt bias            (shape: H,)
    """
   # 1. Chuyển list thành numpy array
    dh = np.asarray(dh)
    x_t, h_prev, h_t, W, U, b = [np.asarray(item) for item in cache]
    
    # 2. Đưa về ma trận 2D (cực kỳ quan trọng để dùng phép @)
    dh_2d = dh.reshape(-1, 1)
    h_t_2d = h_t.reshape(-1, 1)
    x_t_2d = x_t.reshape(-1, 1)
    h_prev_2d = h_prev.reshape(-1, 1)
    
    # 3. Nhân từng phần tử (*) để tính đạo hàm qua activation tanh
    dz_2d = dh_2d * (1 - np.square(h_t_2d))  # Shape: (H, 1)
    
    # 4. Nhân ma trận (@) để tính gradient cho các trọng số
    dx_t_2d = W.T @ dz_2d              # Shape: (D, 1)
    dh_prev_2d = U.T @ dz_2d          # Shape: (H, 1)
    dW = dz_2d @ x_t_2d.T              # Shape: (H, D)
    dU = dz_2d @ h_prev_2d.T          # Shape: (H, H)
    db_2d = dz_2d                      # Shape: (H, 1)
    
    # 5. Làm phẳng (flatten) kết quả về lại 1D để khớp với yêu cầu đầu ra
    dx_t = dx_t_2d.ravel()
    dh_prev = dh_prev_2d.ravel()
    db = db_2d.ravel()
    return (dx_t, dh_prev, dW, dU, db)
