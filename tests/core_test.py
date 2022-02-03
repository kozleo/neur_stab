from src.core import estimate_jac
import numpy as np

if __name__ == "__main__":
    batch, time, neuron = 10, 100, 4
    X_test = np.random.rand(batch, time, neuron)

    js, cs = estimate_jac(X_test)

    # make sure jacobians and intercepts are the correct shape
    assert js.shape == (time - 1, neuron, neuron), "js is the wrong shape."
    assert cs.shape == (time - 1, neuron), "cs is the wrong shape."

