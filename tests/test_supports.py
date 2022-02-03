import numpy as np


def generate_random_stable_linear_RNN(N, max_abs_eig):
    W = 0
    b = 0
    return W, b


def linear_RNN_update(x, W, b):
    return W @ x + b


def run_linear_RNN(W, b, B, T):
    N = W.shape[0]
    x = np.random.normal(0, 1, shape=(B, N))

    xs = np.zeros((B, T, N))

    for t in range(T):
        x = linear_RNN_update(x, W, b)
        xs[:, t, :] = x

    return xs
