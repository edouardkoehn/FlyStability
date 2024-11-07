import os

import click
import numpy as np

import flybrain.functional as functional
import flybrain.model as model
import flybrain.utils as utils
from flybrain.training import train_RD_RNN


@click.command()
@click.option(
    "--n", type=int, required=True, help="Size of the model, number of neurons used"
)
@click.option(
    "--n_samples",
    "--n_samples",
    type=int,
    required=True,
    help="Number of sample used, (default:1)",
)
@click.option("--nLE", type=int, required=True, help="Number of Lyapunov exponent used")
@click.option(
    "--loss",
    type=click.Choice(["l2", "MSE"]),
    required=True,
    help="Which loss we want to use for the optimisation",
)
@click.option(
    "--activation",
    type=click.Choice(["tanh", "tanh_pos", "tanh_streched"]),
    required=True,
    help="Which loss we want to use for the optimisation",
)
@click.option(
    "--target", type=float, required=False, default=0.0, help="Target lyapunov vector"
)
@click.option(
    "--tOns",
    type=float,
    required=False,
    default=0.2,
    help="Step size between two consecutive QR facto",
)
@click.option(
    "--tSim",
    type=int,
    required=False,
    default=200,
    help="Length of the simulation [tau]",
)
@click.option("--g", type=float, required=True, help="Synaptic distribution parameter")
@click.option(
    "--n_epochs", type=int, required=False, default=10, help="Number of epochs used"
)
@click.option(
    "--lr", type=int, required=False, default=0.001, help="Learning rate used"
)
@click.option(
    "--train_weights",
    type=bool,
    required=False,
    default=True,
    help="Optimizition on the weights",
)
@click.option(
    "--train_shifts",
    type=bool,
    required=False,
    default=False,
    help="Optimizition on the shitfs",
)
@click.option(
    "--train_gains",
    type=bool,
    required=False,
    default=False,
    help="Optimizition on the gains",
)
def run_training_RD_RNN(
    n: int = 100,
    n_samples: int = 1,
    nle: int = 1,
    loss: str = "l2",
    target: float = 0.0,
    tons: float = 0.2,
    tsim: int = 200,
    g: float = 1,
    n_epochs: int = 100,
    lr: float = 0.001,
    train_weights: bool = True,
    train_shifts: bool = False,
    train_gains: bool = False,
    activation: str = "tanh",
    dt: float = 0.1,
):
    # Set up paths
    np.random.seed(30)
    ROOT = utils.get_root()
    output_paths = {
        "logs": os.path.join(ROOT, "data", "logs", "rd_RNN"),
        "fig": os.path.join(ROOT, "data", "fig", "train_model_weight"),
        "model": os.path.join(ROOT, "data", "model", "rd_RNN"),
    }
    for path in output_paths.values():
        os.makedirs(path, exist_ok=True)

    # Experiment parameters
    loss_func = {"l2": functional.l2_norm(target), "MSE": functional.mse(target)}[loss]
    activation_func = {
        "tanh": functional.tanh(),
        "tanh_pos": functional.tanh_positive(),
        "tanh_strech": functional.tanh_strech(),
    }[activation]

    experiment_name = (
        f"{activation_func.name()}_Weights{train_weights}_Shifts{train_shifts}_"
        f"Gains{train_gains}_N{n}_lr{lr}_NLE{nle}_Epochs{n_epochs}_{loss_func.name()}_"
        f"g{g}_Tons{tons}_Tsim{tsim}_dt{dt}"
    )

    training_loss, training_maxlambda, spectrum = [], [], []

    for sample in range(n_samples):
        # Create model
        W, C = utils.construct_Random_Matrix_simple(
            n_neurons=n, coupling=g, showplot=False
        )
        c0 = np.random.normal(0.0, 1.0, n)
        rnn = model.RNN(
            connectivity_matrix=C,
            weights_matrix=W,
            initial_condition=c0,
            activation_function=activation_func,
        )

        run_name = f"{experiment_name}_Sample{sample}"

        # Train model
        train_RD_RNN(
            rnn_model=rnn,
            loss=loss_func,
            nLE=nle,
            N_epoch=n_epochs,
            tSim=tsim,
            tONs=tons,
            dt=dt,
            train_weights=train_weights,
            train_shifts=train_shifts,
            train_gains=train_gains,
            lr=lr,
            run_name=run_name,
        )

        # Load logs and store results
        data = utils.load_logs(
            os.path.join(output_paths["logs"], f"{run_name}_logs.json")
        )
        training_loss.append(data["training_loss"])
        training_maxlambda.append(data["training_lambda_max"])
        spectrum.append(data["spectrum"])
        # Plot results
        utils.plot_losses(
            os.path.join(output_paths["fig"], experiment_name),
            training_loss,
            training_maxlambda,
        )
        utils.plot_spectrum(
            os.path.join(output_paths["fig"], experiment_name), spectrum
        )

    # Plot results
    utils.plot_losses(
        os.path.join(output_paths["fig"], experiment_name),
        training_loss,
        training_maxlambda,
    )
    utils.plot_spectrum(os.path.join(output_paths["fig"], experiment_name), spectrum)

    return


if __name__ == "__main__":
    run_training_RD_RNN()
