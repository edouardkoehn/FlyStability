import copy
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch


def get_root():
    """Methods for getting the root of the repository"""
    return str(Path(__file__).resolve().parent.parent)


def construct_Random_Matrix_simple(
    n_neurons=100, coupling=1, showplot=False, seed=None
):
    """Method to construct a random matrix, N(0, g)"""
    if not seed is None:
        np.random.seed(seed)

    W = np.random.normal(
        loc=0, scale=coupling / np.sqrt(n_neurons), size=(n_neurons, n_neurons)
    )
    np.fill_diagonal(W, 0)
    C = np.where(W > 0, 1, np.where(W < 0, -1, 0))
    W = np.abs(W)
    if showplot:
        fig, axs = plt.subplots(1, 2, figsize=(10, 10))
        axs[0] = sns.heatmap(
            W,
            center=0,
            square=True,
            cmap="bwr",
            cbar_kws={"fraction": 0.046},
            annot=False,
            ax=axs[0],
        )
        axs[0].set_title("Random weights matrix", fontsize=12)
        axs[0].set_xticks([])
        axs[0].set_yticks([])
        axs[1] = sns.heatmap(
            C,
            center=0,
            square=True,
            cmap="bwr",
            cbar_kws={"fraction": 0.046},
            ax=axs[1],
        )
        axs[1].set_title("Random connectivity matrix", fontsize=12)
        axs[1].set_xticks([])
        axs[1].set_yticks([])
        # plt.xticks([0,n_neurons-1],[1,n_neurons],fontsize=10)
        # plt.yticks([0,n_neurons-1],[1,n_neurons],fontsize=10)
        plt.show()
    return W, C


def display_trajectory(X, ax1, ax2, ind):
    sns.heatmap(
        X,
        square=False,
        cmap="bwr",
        cbar_kws={"fraction": 0.046},
        annot=False,
        ax=ax1,
        vmin=-5,
        vmax=5,
    )
    ax1.set_title("Trajectory evolution", fontsize=12)
    ax1.set_ylabel("Unit", fontsize=12)
    ax1.set_xlabel("Time", fontsize=12)
    ax1.autoscale()

    random_neurons = ind

    for i in random_neurons:
        ax2.plot(np.arange(0, X.shape[1]), X[i, :], label=f"neuron_{i}")
    ax2.set_title("Sampled Trajectories", fontsize=12)
    ax2.set_xlabel("Time", fontsize=12)
    ax2.legend()
    ax2.set_xticks(np.arange(0, X.shape[1], step=10))
    # ax2.set_ylim([-3,3])
    plt.xticks(rotation=90)


def display_activity(A, ax1, ax2, ind):
    sns.heatmap(
        A,
        square=False,
        cmap="bwr",
        cbar_kws={"fraction": 0.046},
        annot=False,
        ax=ax1,
        vmin=0,
        vmax=1,
    )
    ax1.set_title("Network activity evolution", fontsize=12)
    ax1.set_ylabel("Unit", fontsize=12)
    ax1.set_xlabel("Time", fontsize=12)
    ax1.autoscale()
    random_neurons = ind

    for i in random_neurons:
        ax2.plot(np.arange(0, A.shape[1]), A[i, :], label=f"neuron_{i}")
    ax2.set_title("Sampled Network activity", fontsize=12)
    ax2.set_xlabel("Time", fontsize=12)
    ax2.legend()
    ax2.set_xticks(np.arange(0, A.shape[1], step=10))
    plt.xticks(rotation=90)


def run_plot_rnn(model, duration, driving_force=None, ind=None):
    if driving_force is None:
        driving_force = torch.zeros(model.H.shape[0], duration)
    X = np.zeros((model.H.shape[0], duration))
    A = np.zeros((model.H.shape[0], duration))
    model.set_states(np.random.normal(0, 1, model.H.shape[0]))
    for t in range(duration):
        X[:, t] = model.H.detach().numpy()
        A[:, t] = model.A.detach().numpy()
        model(ext_inputs=driving_force[:, t], update=True)
    # print(X[0,:])
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=False, sharey=False)
    if ind is None:
        traj_ind = np.random.choice(np.arange(0, A.shape[0]), 3)
    else:
        traj_ind = ind
    display_trajectory(X, axs[0, 0], axs[0, 1], traj_ind)
    display_activity(A, axs[1, 0], axs[1, 1], traj_ind)

    plt.tight_layout()
    # plt.show()
    return fig


""" def construct_path(main_name):
    ROOT_PATH = get_root()
    output_folder_figs = os.path.join(ROOT_PATH, "data", "fig", main_name)
    output_folder_logs = os.path.join(ROOT_PATH, "data", "logs", main_name)
    output_folder_models = os.path.join(ROOT_PATH, "data", "models", main_name)

    if not os.path.exists(output_folder_figs):
        os.mkdir(output_folder_figs)
    if not os.path.exists(output_folder_logs):
        os.mkdir(output_folder_logs)
    if not os.path.exists(output_folder_models):
        os.mkdir(output_folder_models)

    return (output_folder_figs, output_folder_logs, output_folder_models)
 """


def load_flybrain(ROI: str = "ellipsoid_body"):
    if ROI == "ellipsoid_body":
        general_path = os.path.join(
            get_root(), "data", "connectomics", "ellispoid_body"
        )
    else:
        print("Brain ROI not found")

    W = np.load(os.path.join(general_path, "adjacency.npy")).T
    C = np.where(W > 0, 1, np.where(W < 0, 0, 0))

    Ei = np.load(os.path.join(general_path, "ei_neuron_types.npy"))
    Ei = np.diag(2 * Ei - 1)
    C = np.matmul(Ei, C)
    return W, C


def sinkhorn_knopp(A, max_iter=10, tol=1e-6, epsilon=1e-10):
    """
    Sinkhorn-Knopp algorithm for matrix normalization to get a doubly stochastic matrix
    Such that the excit/inhib are balanced and the norm of the columns row equal to one
    """
    # Ensure the matrix is non-negative
    A = np.abs(A)

    for _ in range(max_iter):
        # Normalize rows
        row_sums = 2 * A.sum(axis=1, keepdims=True) + epsilon
        A = A / row_sums

        # Normalize columns
        col_sums = 2 * A.sum(axis=0, keepdims=True) + epsilon
        A = A / col_sums

        # Check for convergence
        if np.allclose(A.sum(axis=1), 1, atol=tol) and np.allclose(
            A.sum(axis=0), 1, atol=tol
        ):
            break
    return A


def get_Normalize_Connectivity(W, C):
    C_inhib = copy.deepcopy(C)
    C_inhib[np.where(C > 0)] = 0
    C_inhib = C_inhib * -1
    W_inhib = W * C_inhib

    C_exci = copy.deepcopy(C)
    C_exci[np.where(C < 0)] = 0
    W_exci = W * C_exci

    W_exci_N = sinkhorn_knopp(W_exci)
    W_inhib_N = sinkhorn_knopp(W_inhib)

    return (W_inhib_N + W_exci_N), C


import json

import matplotlib.pyplot as plt
import numpy as np


def plot_spectrum(file_path, spectrum):
    """
    Plots the full Lyapunov spectrum and saves the plot to a file.

    Parameters:
    - file_path (str): Path to save the spectrum plot image.
    - spectrum (list of np.ndarray): List of Lyapunov spectra, each as an array.

    Each element in the spectrum list represents a different sample's Lyapunov spectrum,
    and points are plotted with markers and transparency to distinguish multiple spectra.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    for i, spec in enumerate(spectrum):
        ax.scatter(np.arange(spec.shape[0]), spec, marker="o", s=2.5, alpha=0.4)
    ax.hlines(
        y=0,
        xmin=0,
        xmax=spectrum[0].shape[0],
        linestyles="--",
        color="black",
        alpha=0.5,
    )
    ax.set_title("Lyapunov Spectrum")
    ax.set_ylabel(r"$\lambda_i$")
    ax.set_xlabel(r"$i$")
    plt.tight_layout()
    plt.savefig(f"{file_path}_spectrum_logs.png")


def plot_losses(file_path, losses, maxlambda):
    """
    Plots training loss and maximum Lyapunov exponent over epochs, saving the plot to a file.

    Parameters:
    - file_path (str): Path to save the training plot image.
    - losses (list of np.ndarray): List of arrays representing training loss over epochs.
    - maxlambda (list of np.ndarray): List of arrays representing maximum Lyapunov exponents.

    Plots the evolution of training loss and the maximum Lyapunov exponent for each sample.
    """
    fig, axs = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for loss in losses:
        axs[0].plot(np.arange(loss.shape[0]), loss, alpha=0.5)
    for maxl in maxlambda:
        axs[1].scatter(np.arange(maxl.shape[0]), maxl, alpha=0.5)

    axs[0].set_title("Training Loss")
    axs[0].set_ylabel(r"$|\lambda|$")
    axs[1].set_title(r"Training $\lambda_{max}$")
    axs[1].set_ylabel(r"$\lambda_{max}$")
    axs[1].set_xlabel(r"$Epoch$")
    plt.tight_layout()
    plt.savefig(f"{file_path}_training_logs.png")


def load_logs(file_path):
    """
    Loads and converts log data from a JSON file to a dictionary of NumPy arrays.

    Parameters:
    - file_path (str): Path to the JSON file containing training logs.

    Returns:
    - dict: Dictionary containing NumPy arrays for training losses, spectrum,
      maximum Lyapunov exponents, and gradient norms for weights, gains, and shifts.
    """
    with open(file_path, "r") as f:
        data = json.load(f)
    return {
        "training_loss": np.array(data["training_loss"]),
        "spectrum": np.array(data["spectrum"]),
        "training_lambda_max": np.array(data["training_lambda_max"]),
        "grad_gains": np.array(data["grad_gains"]),
        "grad_shifts": np.array(data["grad_shifts"]),
        "grad_weights": np.array(data["grad_weights"]),
    }
