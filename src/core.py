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
        reg = LinearRegression(fit_intercept=False).fit(X_t, X_tp1)

        # extract J_t and c_t from the affine map
        js[t] = reg.coef_
        cs[t] = reg.intercept_

    return js, cs


def estimate_stability_using_particle(js: np.ndarray, p: int) -> np.ndarray:
    """Estimate maximal lyapunov exponent given a sequence of Jacobians using the technique of ___.
    Push a random unit vector through the sequence and measure the deformation."

    Args:
        js (np.ndarray): Sequence of Jacobians, stored in a multi-dimensional array.
        p (int): Number of random unit vectors to  use.

    Returns:
        lams: p-dimensional array containing estimates of maximal Lyapunov exponent.
    """

    T, N = js.shape[0], js.shape[1]

    # generate p vectors on the unit sphere in R^n
    U = np.random.randn(N, p)
    U /= np.linalg.norm(U, axis=0)

    # preallocate memory for lyapunov exponents
    lams = np.zeros(p)

    for t in range(T):

        # push U through jacobian at time t
        U = js[t] @ U

        # measure deformation and store log
        lams += np.log(np.linalg.norm(U, axis=0))

        # renormalize U
        U /= np.linalg.norm(U, axis=0)

    # average by number time steps to get lyapunov exponent estimates
    lams /= T

    return lams
