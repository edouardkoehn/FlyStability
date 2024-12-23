import json
import os

import click
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

import flybrain.functional as functional
import flybrain.lyapunov as lyapu
import flybrain.model as model
import flybrain.utils as utils


@click.command()
@click.option(
    "--n",
    type=int,
    required=True,
    default=100,
    help="Size of the network ",
)
@click.option(
    "--n_samples", type=int, required=True, help="Number of samples for each g value"
)
@click.option(
    "--variance_max",
    type=int,
    default=12,
    required=False,
    help="Number of samples for each g value",
)
@click.option(
    "--parameter",
    type=click.Choice(["weights", "shifts", "gains"]),
    multiple=True,
    required=True,
    default=["weights"],
    help="Which parameter to vary ",
)
@click.option(
    "--activation",
    type=click.Choice(["std", "pos", "strech"], case_sensitive=False),
    default="std",
    required=False,
    help="Activation function used in the model (default: std)",
)
@click.option(
    "--save",
    type=bool,
    default=True,
    required=False,
    help="Options to save the output",
)
def spectrum_characterization(
    n,
    n_samples,
    variance_max: int = 15,
    activation: str = "std",
    parameter: list = ["weights"],
    tSim: int = 200,
    dt: float = 0.1,
    tOns: float = 0.2,
    save: bool = True,
):
    """Generates multiple RNN models with varying coupling values and calculates Lyapunov spectrum."""
    parameter = list(parameter)
    N = n
    # Define output paths and create directory if it doesn't exist
    ROOT_PATH = utils.get_root()
    output_fig_path = os.path.join(
        ROOT_PATH, "data", "fig", "spectrum_characterization"
    )
    output_log_path = os.path.join(
        ROOT_PATH, "data", "logs", "spectrum_characterization"
    )
    os.makedirs(output_fig_path, exist_ok=True)
    os.makedirs(output_log_path, exist_ok=True)

    # Select activation function
    activation_function = {
        "std": functional.tanh(),
        "pos": functional.tanh_positive(),
        "strech": functional.tanh_strech(),
    }[activation]
    # Prepare values of g and data storage
    sigmas = [2**n for n in range(0, variance_max)]
    models = {str(g): [] for g in sigmas}
    spectrums = {str(g): [] for g in sigmas}
    entropies = {str(g): [] for g in sigmas}
    dimensionality = {str(g): [] for g in sigmas}
    cmap = cm.get_cmap("nipy_spectral", len(sigmas))
    colors = {str(g): cmap(i) for g, i in zip(sigmas, range(len(sigmas)))}

    W_default, C_default = utils.construct_Random_Matrix_simple(
        n_neurons=N, coupling=1.0, showplot=False
    )
    # Loop over each g value and generate n_samples models
    for i, sigma in enumerate(sigmas):
        for sample in range(n_samples):
            initial_condition = np.random.normal(0, 1, N)
            if "weights" in parameter:
                W, C = utils.construct_Random_Matrix_simple(
                    n_neurons=N, coupling=sigma, showplot=False
                )
            else:
                W, C = W_default, C_default

            # Handle gains and shifts
            gains = (
                np.random.normal(loc=1, scale=sigma / np.sqrt(N), size=N)
                if "gains" in parameter
                else np.ones(N)
            )

            shifts = (
                np.random.normal(loc=1, scale=sigma / np.sqrt(N), size=N)
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
            spectrum = lyapu.Lyapunov().compute_spectrum(
                model=RNN, dt=dt, tSim=tSim, nLE=N, tONS=tOns, logs=True
            )
            models[f"{sigma}"].append(RNN)
            spectrums[f"{sigma}"].append(spectrum)

    # Generate run name and save results to DataFrame
    run_name = (
        f"{RNN.name()}_N{RNN.N}_nSample{n_samples}_tSim{tSim}_dt{dt}_tOns{tOns}"
        + f"_{'_'.join(parameter)}"
    )
    choosen = ["1", "8", "64", "256", "1000"]
    for keys in spectrums.keys():
        for sample in spectrums[keys]:
            entropies[keys].append(lyapu.get_entropy(sample))
            dimensionality[keys].append(lyapu.get_attractor_dimensionality(sample))

    plot_dimensionality(dimensionality, sigmas, colors)
    if save:
        plt.tight_layout()
        plt.savefig(os.path.join(output_fig_path, f"{run_name}_dim.pdf"), format="pdf")
        plt.savefig(os.path.join(output_fig_path, f"{run_name}_dim.svg"), format="svg")

    plot_entropy(entropies, sigmas, colors)
    if save:
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_fig_path, f"{run_name}_entro.pdf"), format="pdf"
        )
        plt.savefig(
            os.path.join(output_fig_path, f"{run_name}_entro.svg"), format="svg"
        )

    plot_spectrum(spectrums, choosen, sigmas, colors)
    if save:
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_fig_path, f"{run_name}_spect.pdf"), format="pdf"
        )
        plt.savefig(
            os.path.join(output_fig_path, f"{run_name}_spect.svg"), format="svg"
        )
        for key in spectrums.keys():
            for sample in range(0, len(spectrums[key])):
                spectrums[key][sample] = spectrums[key][sample].tolist()
                entropies[key][sample] = entropies[key][sample].tolist()

        with open(os.path.join(output_log_path, run_name + "_spectrum.json"), "w") as f:
            json.dump(
                spectrums,
                f,
            )
        with open(
            os.path.join(output_log_path, run_name + "_entropies.json"), "w"
        ) as f:
            json.dump(
                entropies,
                f,
            )
        with open(os.path.join(output_log_path, run_name + "_dim.json"), "w") as f:
            json.dump(
                dimensionality,
                f,
            )

    return


def plot_dimensionality(dim_dict, var_list, color):
    fig, axs = plt.subplots(1, 1)
    for key, i in zip(dim_dict.keys(), range(len(var_list))):
        for samples in dim_dict[key]:
            axs.scatter(var_list[i], samples, color=color[key], alpha=0.1)
        axs.scatter(
            var_list[i],
            np.array(dim_dict[key]).mean(axis=0),
            alpha=0.8,
            label=f"sigma:{key}",
            color=color[key],
        )
    axs.set_ylabel("D")
    axs.set_xlabel(r"$\sigma$")
    axs.set_xscale("log")
    axs.spines["right"].set_color("none")
    axs.spines["top"].set_color("none")
    axs.set_title("Attractor dimensionality")
    plt.tight_layout()
    return


def plot_entropy(entropy_dict, var_list, color):
    fig, axs = plt.subplots(1, 1)
    for key, i in zip(entropy_dict.keys(), range(len(var_list))):
        for samples in entropy_dict[key]:
            axs.scatter(var_list[i], samples, color=color[key], alpha=0.1)
        axs.scatter(
            var_list[i],
            np.array(entropy_dict[key]).mean(axis=0),
            alpha=0.8,
            label=f"sigma:{key}",
            color=color[key],
        )
    axs.set_ylabel("H")
    axs.set_xlabel(r"$\sigma$")
    axs.set_xscale("log")
    axs.spines["right"].set_color("none")
    axs.spines["top"].set_color("none")
    axs.set_title("Sinai-Kolmogorov Entropy")
    plt.tight_layout()
    return


def plot_spectrum(spectrum_dict, choosen_key, var_list, colors):
    fig, axs = plt.subplots(1, 1, figsize=(10, 5))
    # Plot a horizontal line
    axs.hlines(y=0, xmin=0, xmax=100, ls="--", color="black")
    # Loop through spectrums and plot
    for key, i in zip(spectrum_dict.keys(), range(len(var_list))):
        all_samples = np.array(
            spectrum_dict[key]
        )  # Convert to a NumPy array for mean calculation
        if key in choosen_key:
            # Plot individual samples
            for j, sample in enumerate(all_samples):
                label = (
                    key if j == 0 else None
                )  # Only add the label for the first sample
                axs.plot(
                    np.linspace(0, 100, len(sample)),
                    sample,
                    alpha=0.1,
                    color=colors[key],
                    lw=1,
                )

            # Plot the mean of the samples
            mean_sample = all_samples.mean(axis=0)
            axs.plot(
                np.linspace(0, 100, len(sample)),
                mean_sample,
                color=colors[key],
                alpha=0.5,
                lw=1,
                label=f"g:{key}",
            )
        else:
            pass
    # Add legend
    axs.legend()
    axs.set_title("Lyapunov spectrum")
    axs.set_ylim([-2.5, 2.5])
    axs.spines["right"].set_color("none")
    axs.spines["top"].set_color("none")
    axs.set_ylabel(r"$\lambda_{i}$")
    axs.set_xlabel(r"$i$")
    plt.tight_layout()
    return


if __name__ == "__main__":
    spectrum_characterization()
