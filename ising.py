import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from stqdm import stqdm


def compute_roh(
    roh: np.ndarray, alpha: np.ndarray, preference: np.ndarray, slow: bool = False
) -> np.ndarray:
    if slow:
        new_roh = np.zeros(roh.shape)
        for k in range(roh.shape[0]):
            for a in range(roh.shape[1]):
                alp = alpha[:, a]
                alp = alp + preference[:, a] * 0.5
                alp[k] = -np.inf
                new_roh[k, a] = 0.5 * preference[k, a] - max(alp)
        return new_roh

    alpha_perf = 0.5 * preference + alpha
    cols = np.arange(alpha_perf.shape[1])  # k

    idx_max = np.argmax(alpha_perf, axis=0)
    first_max = alpha_perf[idx_max, cols]

    alpha_perf[idx_max, cols] = -np.inf
    second_max = alpha_perf[np.argmax(alpha_perf, axis=0), cols]

    max_matrix = np.zeros_like(roh) + first_max[None, :]
    max_matrix[idx_max, cols] = second_max

    return 0.5 * preference - max_matrix


def compute_alpha(
    roh: np.ndarray,
    alpha: np.ndarray,
    preference: np.ndarray,
    slow: bool = False,
) -> np.ndarray:
    if slow:
        new_alpha = np.zeros(roh.shape)
        for k in range(roh.shape[0]):
            for a in range(roh.shape[1]):
                r = roh[k, :]
                r = r + preference[k, :] * 0.5
                r[a] = -np.inf
                new_alpha[k, a] = 0.5 * preference[k, a] - max(r)
        return new_alpha

    roh_perf = 0.5 * preference + roh

    rows = np.arange(roh_perf.shape[0])  # a

    idx_max = np.argmax(roh_perf, axis=1)
    first_max = roh_perf[rows, idx_max]

    roh_perf[rows, idx_max] = -np.inf
    second_max = roh_perf[rows, np.argmax(roh_perf, axis=1)]

    max_matrix = np.zeros_like(alpha) + first_max[:, None]
    max_matrix[rows, idx_max] = second_max

    return 0.5 * preference - max_matrix


def compute_eta(eta: np.ndarray, xi: np.ndarray) -> np.ndarray:
    new_eta = np.zeros(eta.shape)
    for k in range(eta.shape[0]):
        for a in range(eta.shape[1]):
            x = xi[k, :]
            x = np.exp(x)
            x[a] = 0
            new_eta[k, a] = -np.log10(x.sum())
    return new_eta


def compute_xi(eta: np.ndarray, xi: np.ndarray) -> np.ndarray:
    new_xi = np.zeros(xi.shape)
    for k in range(xi.shape[0]):
        for a in range(xi.shape[0]):
            e = eta[:, a]
            e = np.exp(e)
            e[k] = 0
            new_xi[k, a] = -np.log10(e.sum())
    return new_xi


def preference(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    s = np.zeros((u.shape[0], v.shape[0]))
    for i in range(s.shape[0]):
        for j in range(s.shape[1]):
            s[i, j] = -np.linalg.norm(u[i] - v[j])
    return s


def ising(
    X: np.ndarray,
    X_labels: np.ndarray,
    maxiter: int = 1000,
    local_thresh: int = 10,
    draw_every_n_iterations: int = 5,
    data_plot=None,
    fig=None,
    ax=None,
    c: st.empty = None,
):
    log = ""
    count_equal = 0

    users = X[X_labels == 0]
    resources = X[X_labels == 1]
    roh, alpha = np.zeros((users.shape[0], resources.shape[0])), np.zeros(
        (users.shape[0], resources.shape[0])
    )
    eta, xi = np.zeros((users.shape[0], resources.shape[0])), np.zeros(
        (users.shape[0], resources.shape[0])
    )

    preference_matrix = preference(users, resources)
    roh = compute_roh(roh, alpha, preference_matrix)
    alpha = compute_alpha(roh, alpha, preference_matrix)
    # eta = compute_eta(eta, xi)
    # xi = compute_xi(eta, xi)
    converged = False

    for i in stqdm(range(maxiter), st_container=st.sidebar):
        prev_x = np.argmax(roh + alpha, axis=1)
        roh = compute_roh(roh, alpha, preference_matrix)
        alpha = compute_alpha(roh, alpha, preference_matrix)
        # eta = compute_eta(eta, xi)
        # xi = compute_xi(eta, xi)
        x = np.argmax(roh + alpha, axis=1)
        s = np.argmax(eta + xi, axis=1)

        if i % draw_every_n_iterations == 0:
            log = f"Iteration: {i} : Links {x}  \n"
            c.write(log)
            plot_iteration(users, resources, x, s, data_plot, fig, ax, c)

        if np.all(prev_x == x) and len(set(x)) == len(x):
            count_equal += 1

        if count_equal > local_thresh:
            converged = True
            break

    plot_iteration(users, resources, x, s, data_plot, fig, ax, c)
    if converged:
        log += f"Converged after {i} iterations \n"
    else:
        log += f"Did not converge after {i} iterations \n"

    c.write(log)
    ising = np.zeros((users.shape[0], resources.shape[0]))
    for i in range(len(x)):
        ising[i, x[i]] = 1
    return ising


def plot_iteration(
    user: np.ndarray,
    resource: np.ndarray,
    x: np.ndarray,
    s: np.ndarray,
    data_plot=None,
    fig=None,
    ax: plt.Axes = None,
    c=None,
):
    ax.clear() # clear plot from before
    for i in range(len(user)):
        ax.plot(user[i][0], user[i][1], "o", color="blue")
        ax.plot(resource[i][0], resource[i][1], "o", color="red")
        ax.plot(
            [user[i][0], resource[x[i]][0]],
            [user[i][1], resource[x[i]][1]],
            color="black",
        )

    data_plot.pyplot(fig)
