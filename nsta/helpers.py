import numpy as np
import matplotlib.pyplot as plt


def linear_vRNN_update(x, W, t, baseline_FR):
    # update for linear RNN
    return (
        W @ x
    )  # +  baseline_FR*np.sin(t/10) + 0.01*np.random.normal(0,1,W.shape[0])#+ 0.1*np.sin(t/10)


def linear_vRNN_jac(x, W):
    # jacobian of linear RNN
    return W


def get_max_singular_value(W):
    _, s, _ = np.linalg.svd(W)
    return np.max(s)


def get_max_abs_eig_value(W):
    eig_vals, eigvecs = np.linalg.eig(W)
    ind_max_eig = np.argmax(np.abs(eig_vals))

    # get condition number of coordinate transform
    _, s_V, _ = np.linalg.svd(eigvecs)
    cond = np.max(s_V) / np.min(s_V)

    return eig_vals[ind_max_eig], cond


def run_random_vRNN_sim(n, g, T, p):
    # run an n dimensional linear RNN for T timesteps, starting from p different initial conditions.

    # n: number of neurons

    # g: chaos dial

    # T: length of sim

    # p: number of trials

    W = np.random.normal(0, g / np.sqrt(n), (n, n))
    baseline_FR = np.random.normal(0, g, n)

    xsAllTrials = []
    JsAllTrials = []

    for trial in range(p):

        x = np.random.normal(0, 1, n)
        xsOneTrial = []
        JsOneTrial = []
        for time in range(T):

            x = linear_vRNN_update(x, W, time, baseline_FR)
            xsOneTrial.append(x)
            JsOneTrial.append(linear_vRNN_jac(x, W))

        xsAllTrials.append(xsOneTrial)
        JsAllTrials.append(JsOneTrial)

    xsAllTrials = np.stack(xsAllTrials)
    JsAllTrials = np.stack(JsAllTrials)

    return xsAllTrials, JsAllTrials, W


def plot_linear_RNN_summary(W, states):
    # plots eigenvalues and example activations

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    evals, evecs = np.linalg.eig(W)
    max_eig_pos = np.argmax(np.abs(evals))
    t_circle = np.linspace(0, 2 * np.pi, 100)

    ax[0].scatter(evals.real, evals.imag, s=100)
    ax[0].scatter(
        evals[max_eig_pos].real,
        evals[max_eig_pos].imag,
        color="r",
        s=200,
        label="Largest Eigenvalue (Abs Val)",
    )

    ax[0].plot(
        np.cos(t_circle),
        np.sin(t_circle),
        linewidth=2,
        color="k",
        label="Stability Boundry (Inside = Stable)",
    )
    ax[0].legend(loc="upper right")
    ax[0].set_title("Eigenvalues of Weight Matrix")
    ax[0].set_xlabel(r"$Re(\lambda_i)$")
    ax[0].set_ylabel(r"$Im(\lambda_i)$")

    ax[1].plot(states[20, :], color="k", alpha=0.2)
    ax[1].set_title("RNN Activations on A Random Trial")
    ax[1].set_xlabel("Timestep")
    ax[1].set_ylabel("Neural Activation")

    max_eig = np.abs(evals[max_eig_pos])

    plt.tight_layout()
    return max_eig

