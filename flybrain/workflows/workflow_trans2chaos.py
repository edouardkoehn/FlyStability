import os

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import flybrain.functional as functional
import flybrain.model as model
import flybrain.utils as utils
from flybrain.lyapunov import Lyapunov


@click.command()
@click.option(
    "--gmin", type=float, required=True, help="Minimum value for coupling strength g"
)
@click.option(
    "--gmax", type=float, required=True, help="Maximum value for coupling strength g"
)
@click.option(
    "--m_g",
    type=int,
    default=7,
    help="Number of g values between gmin and gmax (default: 5)",
)
@click.option(
    "--n_samples", type=int, required=True, help="Number of samples for each g value"
)
@click.option(
    "--parameter",
    type=click.Choice(["weights", "shifts", "gains"]),
    multiple=True,
    required=False,
    default=["weights"],
    help="Which parameter to vary ",
)
@click.option(
    "--activation",
    type=click.Choice(["std", "pos", "strech"], case_sensitive=False),
    default="std",
    help="Activation function used in the model (default: std)",
)
@click.option(
    "--save",
    type=bool,
    default=True,
    help="Options to save the output",
)
def find_transition_2_chaos(
    gmin: float,
    gmax: float,
    m_g: int,
    N: int = 1000,
    n_samples: int = 10,
    tSim: int = 200,
    dt: float = 0.1,
    tOns: float = 0.2,
    activation: str = "pos",
    parameter: list = ["weights"],
    save: bool = True,
):
    """
    Find the transition to chaos in a recurrent neural network (RNN) model.

    Parameters:
        gmin (float): Minimum value for coupling strength g.
        gmax (float): Maximum value for coupling strength g.
        m_g (int): Number of g values between gmin and gmax.
        N (int): Size of the network.
        n_samples (int): Number of samples for each g value.
        tSim (int): Total simulation time.
        dt (float): Time step size for integration.
        tOns (float): Initial transient time before recording starts.
        activation (str): Activation function used in the model ('std', 'pos', 'strech').
        parameter (list): List of parameters to vary ('weights', 'shifts', 'gains').
        save (bool): Option to save the output.

    Returns:
        None
    """

    parameter = list(parameter)
    # Define output paths and create directory if it doesn't exist
    ROOT_PATH = utils.get_root()
    output_fig_path = os.path.join(ROOT_PATH, "data", "fig", "transition_2_chaos")
    os.makedirs(output_fig_path, exist_ok=True)

    # Select activation function
    activation_function = {
        "std": functional.tanh(),
        "pos": functional.tanh_positive(),
        "strech": functional.tanh_strech(),
    }[activation]
    # Prepare values of g and data storage
    gs = np.linspace(gmin, gmax, num=m_g)
    lambdas = np.ones((n_samples, len(gs)))

    W_default, C_default = utils.construct_Random_Matrix_simple(
        n_neurons=N, coupling=1.0, showplot=False
    )

    # Loop over each g value and generate n_samples models
    for i, g in enumerate(gs):
        for sample in range(n_samples):
            initial_condition = np.random.normal(0, 1, N)
            if "weights" in parameter:
                W, C = utils.construct_Random_Matrix_simple(
                    n_neurons=N, coupling=g, showplot=False
                )
            else:
                W, C = W_default, C_default

            # Handle gains and shifts
            gains = (
                np.random.normal(loc=1, scale=g / np.sqrt(N), size=N)
                if "gains" in parameter
                else np.ones(N)
            )

            shifts = (
                np.random.normal(loc=1, scale=g / np.sqrt(N), size=N)
                if "shifts" in parameter
                else np.zeros(N)
            )

            RNN = model.RNN(
                weights_matrix=W,
                connectivity_matrix=C,
                initial_condition=initial_condition,
                activation_function=activation_function,
                gains_vector=gains,
                shifts_vector=shifts,
            )

            RNN.eval()
            # Compute the Lyapunov spectrum and store it
            lambdas[sample, i] = Lyapunov().compute_spectrum(
                model=RNN, dt=dt, tSim=tSim, nLE=1, tONS=tOns, logs=True
            )

    # Generate run name and save results to DataFrame
    run_name = (
        f"{RNN.name()}_N{RNN.N}_nSample{n_samples}_tSim{tSim}_dt{dt}_tOns{tOns}"
        + f"_{'_'.join(parameter)}"
    )
    data = pd.DataFrame(data=lambdas, columns=[f"{g:.2f}" for g in gs])

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 5))
    palette = sns.color_palette("Set2", n_colors=len(gs))
    sns.stripplot(data=data, ax=ax, alpha=0.4, palette=palette)
    medians = data.median()
    for i, median in enumerate(medians):
        ax.scatter(
            i,
            median,
            marker="D",
            color=palette[i],
            edgecolor="black",
            alpha=1,
        )  # Mark median points

    # Add threshold line at 0
    ax.hlines(
        0, xmin=-0.5, xmax=len(gs) - 0.5, colors="black", linestyles="--", alpha=0.5
    )

    # Customize labels and title
    ax.set_xticks(range(len(gs)))
    ax.set_xticklabels([f"{g:.1f}" for g in gs])
    ax.set_ylabel(r"$\lambda_{max}$")
    ax.set_xlabel(r"$\sigma$")
    ax.set_title(
        f"Transition to Chaos\n{RNN.name()}, {'_'.join(parameter)} drawn from $N(0, \sigma/\\sqrt{{N}})$",
        fontsize=12,  # Adjust fontsize as needed
        wrap=True,  # Ensures wrapping if layout constraints exist
    )
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")
    # Save figure
    if save:
        plt.tight_layout()
        plt.savefig(os.path.join(output_fig_path, f"{run_name}.pdf"), format="pdf")
        plt.savefig(os.path.join(output_fig_path, f"{run_name}.svg"), format="svg")
    return


if __name__ == "__main__":
    find_transition_2_chaos()
