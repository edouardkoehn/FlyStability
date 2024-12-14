import json
import os

import click
import matplotlib.pyplot as plt
import numpy as np

import flybrain.functional as functional
import flybrain.utils as utils
from flybrain.lyapunov import Lyapunov
from flybrain.model import RNN


@click.command()
@click.option("--tons", type=float, required=True, help="tons used in this experiment")
@click.option(
    "--activation",
    type=click.Choice(["tanh", "tanh_pos", "tanh_streched"]),
    required=True,
    help="Which loss we want to use for the optimisation",
)
@click.option("--g", type=float, required=False, default=1.5, help="Syn dist")
@click.option(
    "--n_samples", type=int, required=False, default=5, help="Amount of sample"
)
@click.option(
    "--n",
    type=int,
    required=False,
    default=100,
    help="Size of the model, number of neurons used",
)
@click.option(
    "--nle", type=int, required=False, default=20, help="number of lyapunoc computed"
)
def run_convergence_lyapunov(
    tons: float = 0.1,
    activation: str = "tanh",
    g: float = 1.5,
    n_samples: int = 5,
    n: int = 100,
    nle: int = 20,
    dt: float = 0.1,
    tsim: int = 700,
):
    """
    Simulates the convergence of Lyapunov exponents in a recurrent neural network (RNN) model.

    Parameters:
        tons (float): Initial transient time before recording starts.
        n_samples (int): Number of samples to test.
        g (float): Coupling strength for connectivity matrix.
        n (int): Number of neurons in the network.
        nle (int): Number of Lyapunov exponents to compute.
        dt (float): Time step size for integration.
        tsim (int): Total simulation time.
        activation (str): Type of activation function ('std', 'pos', 'strech').
    """
    np.random.seed(10)  # Set random seed for reproducibility
    # Set the root path for outputs to avoid setting it repeatedly
    #ROOT_PATH = utils.get_root()
    ROOT_PATH="/pscratch/sd/e/ekoehn/FlyStability"
    OUTPUT_FIG_PATH = os.path.join(ROOT_PATH, "data", "fig", "lyapunov_convergence")
    OUTPUT_LOGS_PATH = os.path.join(ROOT_PATH, "data", "logs", "lyapunov_convergence")
    os.makedirs(OUTPUT_FIG_PATH, exist_ok=True)
    os.makedirs(OUTPUT_LOGS_PATH, exist_ok=True)
    # Choose activation function
    activation_functions = {
        "tanh": functional.tanh,
        "tanh_pos": functional.tanh_positive,
        "tanh_strech": functional.tanh_strech,
    }
    activation_function = activation_functions.get(activation, functional.tanh)()

    # Initialize the model
    W, C = utils.construct_Random_Matrix_simple(n_neurons=n, coupling=g, showplot=False)
    initial_conditions = [np.random.normal(0, 1, n) for _ in range(n_samples)]
    rnn = RNN(
        connectivity_matrix=C,
        weights_matrix=W,
        initial_condition=initial_conditions[0],
        activation_function=activation_function,
        dt=dt,
    )
    rnn.eval()
    experiment_name = (
        f"{rnn.name()}_N{rnn.N}_NLE{nle}_g{g}_Tons{tons}_Tsim{tsim}_dt{dt}"
    )

    # Allocate memory for results
    spectrums = []
    times = []
    logs = {}
    for sample in range(n_samples):
        rnn.set_states(initial_conditions[sample])
        lyapunov_metric = Lyapunov()
        lyapunov_metric.compute_spectrum(
            model=rnn, dt=dt, tSim=tsim, nLE=nle, tONS=tons, logs=True
        )
        spectrums.append(lyapunov_metric.spectrum_history)
        times.append(lyapunov_metric.time_history)
        logs[f"sample_{sample}"] = {
            "times": [
                float(time) for time in lyapunov_metric.time_history
            ],  # Ensures native float types
            "tons": [
                int(ton) for ton in lyapunov_metric.stepON_history
            ],  # Ensures native int types
            "spectrum": [
                i.tolist() for i in lyapunov_metric.spectrum_history
            ],  # Converts arrays to lists
        }

    # Save results

    with open(os.path.join(OUTPUT_LOGS_PATH, f"{experiment_name}.json"), "w") as f:
        json.dump(logs, f)
    plot_convergence(
        spectrums, times, nle, os.path.join(OUTPUT_FIG_PATH, f"{experiment_name}.png")
    )


def plot_convergence(spectrums, times, nLE, output_path=None):
    """
    Plots the convergence of Lyapunov exponents over time.

    Parameters:
        spectrums (list): List of spectrum histories for each sample.
        times (list): List of time histories for each sample.
        nLE (int): Number of Lyapunov exponents computed.
        output_path (str, optional): File path to save the plot. If None, the plot is displayed.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["red", "blue", "green", "orange"]
    spectrums = [np.array(spectrums[sample]) for sample in range(len(spectrums))]

    for sample in range(len(spectrums)):
        for exponent, color in zip(range(0, nLE, 2), colors):
            ax.plot(
                times[sample][1:],
                spectrums[sample][:, exponent],
                alpha=0.2,
                color=color,
                marker="D",
                markersize=1,
                linewidth=0.5,
            )

    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")
    ax.set_title("Lyapunov Convergence")
    ax.set_ylabel(r"$\lambda_{i}$")
    ax.set_xlabel(r"$time[\tau]$")
    ax.legend([r"$\lambda_0$", r"$\lambda_2$", r"$\lambda_4$", r"$\lambda_8$"])

    plt.tight_layout()
    plt.savefig(output_path)


if __name__ == "__main__":
    run_convergence_lyapunov()
