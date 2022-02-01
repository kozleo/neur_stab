import numpy as np


def estimate_jac(X: np.ndarray):
    """Function to estimate locally linear maps (Jacobians) from provided data.

    Args:
        X (np.ndarray): Three-dimensional array of neural 'batch x time x neuron'.

    Returns:
        [type]: [description]
    """
    T = X.shape[0]

    for t in range(T):
        pass

    return X
