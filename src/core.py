import numpy as np
from sklearn.linear_model import LinearRegression
from typing import Tuple


def estimate_jac(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Function for estimating locally linear maps from data matrix: x_tpl = J_t @ x_t + c_t

    Args:
        X (np.ndarray): Data matrix. Assumed to be batch x time x features

    Returns:
        Tuple[np.ndarray, np.ndarray]: jacobian coefficient matrices (J_t), intercepts (c_t)
    """

    B, T, N = X.shape

    # pre-allocate memory for storing Jacobians and intercepts
    js = np.zeros(shape=(T - 1, N, N))
    cs = np.zeros(shape=(T - 1, N))

    for t in range(T - 1):

        # get neural state vectors at time t and t + 1
        X_t = X[:, t]
        X_tp1 = X[:, t + 1]

        # find the affine map that takes X_t to X_tp1
        reg = LinearRegression().fit(X_t, X_tp1)

        # extract J_t and c_t from the affine map
        js[t] = reg.coef_
        cs[t] = reg.intercept_

    return js, cs
