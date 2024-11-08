import os

import click
import numpy as np

import flybrain.functional as functional
import flybrain.model as model
import flybrain.utils as utils
from flybrain.training import train_RD_RNN


@click.command()
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
    "--ROI",
    type=str,
    required=True,
    default="EB",
    help="Which ROI, we would like to use",
)
@click.option(
    "--activation",
    type=click.Choice(["tanh_pos", "tanh_streched"]),
    required=True,
    help="Which loss we want to use for the optimisation",
)
@click.option(
    "--subpopulation",
    type=click.Choice(["cell_fiber", "neurotransmitter"]),
    required=True,
    help="Which features would you like to use to define the subpopulation",
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
@click.option(
    "--n_epochs", type=int, required=False, default=10, help="Number of epochs used"
)
@click.option(
    "--lr", type=float, required=False, default=0.01, help="Learning rate used"
)
def run_training_flybrain_RNN(
    n_samples: int = 1,
    subpopulation: str = "neurotransmitter",
    nle: int = 1,
    loss: str = "l2",
    target: float = 0.0,
    tons: float = 0.2,
    tsim: int = 200,
    n_epochs: int = 100,
    lr: float = 0.01,
    roi: str = "EllipsoidBody",
    activation: str = "tanh_pos",
    dt: float = 0.1,
):
    """Pipeline to train a RNN constrained on the flybrain connectome and where the acti
    vations functions are share across cell types"""
    # Set up paths
    np.random.seed(30)
    ROOT = utils.get_root()
    output_paths = {
        "logs": os.path.join(ROOT, "data", "logs", "flybrain_RNN_cell"),
        "fig": os.path.join(ROOT, "data", "fig", "flybrain_RNN_cell"),
        "model": os.path.join(ROOT, "data", "models", "flybrain_RNN_cell"),
    }
    for path in output_paths.values():
        os.makedirs(path, exist_ok=True)

    # Experiment parameters
    loss_func = {"l2": functional.l2_norm(target), "MSE": functional.mse(target)}[loss]
    activation_func = {
        "tanh_pos": functional.tanh_positive(),
        "tanh_strech": functional.tanh_strech(),
    }[activation]
    ROI = {"EB": "ellipsoid_body"}[roi]
    experiment_name = (
        f"{activation_func.name()}_ROI_{roi}_Weights{False}_Shifts{True}_"
        f"Gains{True}_lr{lr}_NLE{nle}_Epochs{n_epochs}_{loss_func.name()}_"
        f"Tons{tons}_Tsim{tsim}_dt{dt}"
    )

    training_loss, training_maxlambda, spectrum = [], [], []

    for sample in range(n_samples):
        # Create model
        ROI_data = utils.load_flybrain(ROI=ROI, types=subpopulation)

        W, C = utils.get_Normalize_Connectivity(
            W=ROI_data["weights"], C=ROI_data["connectivity"]
        )
        c0 = np.random.normal(0.0, 1.0, W.shape[0])
        rnn = model.RNN(
            connectivity_matrix=C,
            weights_matrix=W,
            initial_condition=c0,
            activation_function=activation_func,
            cell_types_vector=ROI_data["types"],
        )

        run_name = f"{experiment_name}_Sample{sample}_SubpopTypes{subpopulation}"

        # Train model
        train_RD_RNN(
            rnn_model=rnn,
            loss=loss_func,
            nLE=nle,
            N_epoch=n_epochs,
            tSim=tsim,
            tONs=tons,
            dt=dt,
            train_weights=False,
            train_shifts=True,
            train_gains=True,
            lr=lr,
            run_name=run_name,
            run_type="flybrain_RNN_cell",
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
    run_training_flybrain_RNN()
