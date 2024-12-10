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
    default=5,
    help="Number of g values between gmin and gmax (default: 5)",
)
@click.option(
    "--n_samples", type=int, required=True, help="Number of samples for each g value"
)
@click.option(
    "--parameter",
    type=click.Choice(["weights", "activation"]),
    required=False,
    default="weights",
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
    N: int = 400,
    n_samples: int = 10,
    tSim: int = 200,
    dt: float = 0.1,
    tOns: float = 0.2,
    activation: str = "tanh",
    parameter: str = "weights",
    save: bool = True,
):
    """Generates multiple RNN models with varying coupling values and calculates Lyapunov spectrum."""

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

    if parameter == "weights":
        # Prepare values of g and data storage
        gs = np.linspace(gmin, gmax, num=m_g)
        lambdas = np.ones((n_samples, len(gs)))

        # Loop over each g value and generate n_samples models
        for i, g in enumerate(gs):
            for sample in range(n_samples):
                W, C = utils.construct_Random_Matrix_simple(
                    n_neurons=N, coupling=g, showplot=False
                )
                initial_condition = np.random.normal(0, 1, N)
                RNN = model.RNN(
                    weights_matrix=W,
                    connectivity_matrix=C,
                    initial_condition=initial_condition,
                    activation_function=activation_function,
                )
                RNN.eval()
                # Compute the Lyapunov spectrum and store it
                lambdas[sample, i] = Lyapunov().compute_spectrum(
                    model=RNN, dt=dt, tSim=tSim, nLE=1, tONS=tOns, logs=True
                )
    elif parameter == "activation":
        # Prepare values of g and data storage
        gs = np.linspace(gmin, gmax, num=m_g)
        lambdas = np.ones((n_samples, len(gs)))
        W, C = utils.construct_Random_Matrix_simple(
            n_neurons=N, coupling=1, showplot=False
        )
        # Loop over each g value and generate n_samples models
        for i, g in enumerate(gs):
            for sample in range(n_samples):
                gains = np.random.normal(loc=1, scale=g / np.sqrt(N), size=(N))
                shifts = np.random.normal(loc=0, scale=g / np.sqrt(N), size=(N))
                initial_condition = np.random.normal(0, 1, N)
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
    run_name = f"{RNN.name()}_N{RNN.N}_nSample{n_samples}_tSim{tSim}_dt{dt}_tOns{tOns}_{parameter}"
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
    ax.set_xlabel(r"$g$")
    ax.set_title(
        f"Transition to Chaos\n{RNN.name()}, {parameter} drawn from $N(0, g/\\sqrt{{N}})$",
        fontsize=12,  # Adjust fontsize as needed
        wrap=True,  # Ensures wrapping if layout constraints exist
    )
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")
    # Save figure
    if save:
        plt.tight_layout()
        plt.savefig(os.path.join(output_fig_path, f"{run_name}.pdf"))
    return


if __name__ == "__main__":
    find_transition_2_chaos()
