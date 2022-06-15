import numpy as np
from sklearn.linear_model import LinearRegression
from typing import Tuple
import matplotlib.pyplot as plt


def estimate_jacobian(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Function for estimating locally linear maps from neural data tensor: x_tpl = J_t @ x_t + c_t

    Args:
        X (np.ndarray): Data tensor. Assumed to be batch-size x time x features

    Returns:
        Tuple[np.ndarray, np.ndarray]: jacobian coefficient matrices (J_t), intercepts (c_t)
    """

    B, T, N = X.shape

    # pre-allocate memory for storing Jacobians and intercepts
    js = np.zeros(shape=(1, T - 1, N, N))
    cs = np.zeros(shape=(1, T - 1, N))

    for t in range(T - 1):

        # get neural state vectors at time t and t + 1
        X_t = X[:, t, :]
        X_tp1 = X[:, t + 1, :]

        # find the affine map that takes X_t to X_tp1
        reg = LinearRegression(fit_intercept=False).fit(X_t, X_tp1)

        # extract J_t and c_t from the affine map
        js[:, t, :, :] = reg.coef_
        cs[:, t, :] = reg.intercept_

    return js, cs


def estimate_LLE(js: np.ndarray, p: int) -> np.ndarray:

    """Estimate Largest Lyapunov Exponent (LLE) using the technique presented in Benettin, G., et al "Lyapunov Characteristic Exponents...Numerical Application". Meccanica 15, 21–30 (1980).
    
    The idea is to push a random unit vector through the sequence of Jacobians, measure the deformation at every time step, and then average the deformations over all time-steps to get an estimate for the LLE."

    Args:
        js (np.ndarray): Sequence of Jacobians, stored in a multi-dimensional array.
        p (int): Number of random unit vectors to use.

    Returns:
        lams: p-dimensional array containing estimates of maximal Lyapunov exponent.
    """

    # preallocate memory for lyapunov exponents
    num_seqs, K, N = js.shape[0], js.shape[1], js.shape[2]
    lams = np.zeros((num_seqs, p))

    # loop over the sequences of jacobians
    for s in range(num_seqs):

        # generate p vectors on the unit sphere in R^n
        U = np.random.randn(N, p)
        U /= np.linalg.norm(U, axis=0)

        # loop through each sequences
        for k in range(K):

            # push U through jacobian at time t
            U = js[s, k, :, :] @ U

            # measure deformations and store
            deformations = np.log(np.linalg.norm(U, axis=0))

            lams[s, :] += deformations

            # renormalize U
            U /= np.linalg.norm(U, axis=0)

        # average by number time steps to get lyapunov exponent estimates
    lams /= K

    return lams
