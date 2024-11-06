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
    "--gmin",
    type=float,
    multiple=False,
    required=True,
    help="""minimum value for g""",
)
@click.option(
    "--gmax",
    type=float,
    multiple=False,
    required=True,
    help="""max value for g""",
)
@click.option(
    "--m_g",
    type=int,
    multiple=False,
    required=False,
    default=5,
    help="""Number of g to pick between the two bounds. 5 by default""",
)
@click.option(
    "--n_samples",
    type=int,
    multiple=False,
    required=False,
    help="""Number of sample to generate from each g values""",
)
@click.option(
    "--activation",
    type=click.Choice(["std", "pos", "strech"], case_sensitive=False),
    required=False,
    default="std",
    help="""The activation function used with the model""",
)
def find_transition_2_chaos(
    gmin: float,
    gmax: float,
    m_g=5,
    N: int = 100,
    n_samples: int = 10,
    tSim: int = 200,
    dt: float = 0.1,
    tOns: float = 0.2,
    activation: str = "tanh",
):
    """Script to generate n_samples model where the coupling values of each model
    varies from g_min to g_max"""
    ROOT_PATH = utils.get_root()
    script_name = "transition_2_chaos"
    output_fig_path = os.path.join(ROOT_PATH, "data", "fig", script_name)
    if not os.path.exists(output_fig_path):
        os.mkdir(output_fig_path)

    if activation == "std":
        activation = functional.tanh()
    elif activation == "pos":
        activation = functional.tanh_positive()
    elif activation == "strech":
        activation = functional.tanh_strech()

    gs = np.linspace(gmin, gmax, num=m_g)
    lambdas = np.ones((n_samples, len(gs)))

    for g, i in zip(gs, range(len(gs))):
        for sample in range(n_samples):
            W, C = utils.construct_Random_Matrix_simple(
                n_neurons=N, coupling=g, showplot=False
            )
            int_cond = np.random.normal(0, 1, N)
            RNN = model.RNN(
                weights_matrix=W,
                connectivity_matrix=C,
                initial_condition=int_cond,
                activation_function=activation,
            )
            RNN.eval()
            lambdas[sample, i] = Lyapunov().compute_spectrum(
                model=RNN, dt=dt, tSim=tSim, nLE=1, tONS=tOns, logs=True
            )
    run_name = f"{RNN.name()}_N{RNN.N}_nSample{n_samples}_tSim{tSim}_dt{dt}_tOns{tOns}"
    data = pd.DataFrame(columns=[f"{g}" for g in gs], data=lambdas)

    fig, axs = plt.subplots(1, 1, figsize=(10, 5))
    palette = sns.color_palette(
        "Set2", n_colors=len(gs)
    )  # Adjust this based on the number of groups (gs)
    g = sns.stripplot(data=data, ax=axs, alpha=0.4, palette=palette)
    medians = data.median(axis=0, skipna=True)

    for i, median in enumerate(medians):
        axs.scatter(
            i, median, marker="D", color=palette[i], alpha=1, edgecolor="black"
        )  # Add edgecolor for clarity
    axs.hlines(
        y=0,
        xmin=-int(len(gs) * 0.1),
        xmax=(len(gs) - 1) + int(len(gs) * 0.1),
        ls="--",
        color="black",
        alpha=0.5,
    )

    axs.set_xticks(np.arange(len(gs)), [f"{g:.1f}" for g in gs])
    axs.set_ylabel(r"$\lambda_{max}$")
    axs.set_xlabel(r"$g$")
    axs.set_title(f"Transition to Chaos for {RNN.name()} random network")
    plt.tight_layout()
    plt.savefig(os.path.join(output_fig_path, run_name + ".png"))
    return


if __name__ == "__main__":
    find_transition_2_chaos()
