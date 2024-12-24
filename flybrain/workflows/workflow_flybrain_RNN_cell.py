import os

import click
import numpy as np

import flybrain.connectome as connectome
import flybrain.functional as functional
import flybrain.model as model
import flybrain.utils as utils
from flybrain.training import train_RD_RNN


@click.command()
@click.option("--n_samples", type=int, default=1, help="Number of runs to perform")
@click.option(
    "--nle",
    type=int,
    default=1,
    help="Number of Lyapunov exponents used for the optimization",
)
@click.option("--n_epochs", type=int, default=10, help="Number of epochs used")
@click.option(
    "--roi", type=str, required=True, default="EB", help="Region of interest (Flybrain)"
)
@click.option(
    "--subpopulation",
    type=click.Choice(["cell_fibers", "neurotransmitter"]),
    default="neurotransmitter",
    help="Subpopulation feature( cell types based or neurotransmitter based)",
)
@click.option(
    "--activation",
    type=click.Choice(["tanh_pos", "tanh_streched"]),
    default="tanh_pos",
    help="Activation function used",
)
@click.option(
    "--loss", type=click.Choice(["l2", "MSE"]), default="l2", help="Loss function"
)
@click.option(
    "--target", type=float, default=0.0, help="Target value for the Lyapunov vector"
)
@click.option(
    "--tons", type=float, default=0.2, help="Step size between QR factorizations"
)
@click.option("--tsim", type=int, default=200, help="Simulation length [tau]")
@click.option("--lr", type=float, default=0.01, help="Learning rate")
def run_training_flybrain_pop(
    n_samples,
    subpopulation,
    nle,
    loss,
    target,
    tons,
    tsim,
    n_epochs,
    lr,
    roi,
    activation,
):
    """
    Pipeline to train an RNN model constrained on the flybrain connectome.

    Parameters:
        n_samples (int): Number of samples to use for training.
        subpopulation (str): Defines subpopulation feature (choices: 'cell_fiber' or 'neurotransmitter').
        nle (int): Number of Lyapunov exponents to compute.
        loss (str): Loss function to use, either 'l2' or 'MSE'.
        target (float): Target Lyapunov vector.
        tons (float): Step size between consecutive QR factorizations.
        tsim (int): Simulation length (in tau).
        n_epochs (int): Number of training epochs.
        lr (float): Learning rate.
        roi (str): Region of interest for model data, default is "EB".
        activation (str): Activation function type.

    Returns:
        None
    """
    np.random.seed(30)
    ROOT = utils.get_root()
    output_paths = {
        "logs": os.path.join(ROOT, "data", "logs", "flybrain_RNN_cell"),
        "fig": os.path.join(ROOT, "data", "fig", "flybrain_RNN_cell"),
        "model": os.path.join(ROOT, "data", "models", "flybrain_RNN_cell"),
    }
    for path in output_paths.values():
        os.makedirs(path, exist_ok=True)

    loss_func = {"l2": functional.l2_norm(target), "MSE": functional.mse(target)}[loss]
    activation_func = {
        "tanh_pos": functional.tanh_positive(),
        "tanh_strech": functional.tanh_strech(),
    }[activation]

    experiment_name = (
        f"{activation_func.name()}_ROI_{roi}_Subpop_{subpopulation}_Weights{False}_Shifts{True}_"
        f"Gains{True}_lr{lr}_NLE{nle}_Epochs{n_epochs}_{loss_func.name()}_"
        f"Tons{tons}_Tsim{tsim}_dt{0.1}"
    )

    training_loss, training_maxlambda, spectrum = [], [], []
    for sample in range(n_samples):
        ROI_data = connectome.load_flybrain(ROI=roi, types=subpopulation)
        W, C = connectome.normalize_connectome(
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
        run_name = f"{experiment_name}_Sample{sample}"
        train_RD_RNN(
            rnn_model=rnn,
            loss=loss_func,
            nLE=nle,
            N_epoch=n_epochs,
            tSim=tsim,
            tONs=tons,
            dt=0.1,
            train_weights=False,
            train_shifts=True,
            train_gains=True,
            lr=lr,
            run_name=run_name,
            run_type="flybrain_RNN_cell",
        )

        data = utils.load_logs(
            os.path.join(output_paths["logs"], f"{run_name}_logs.json")
        )
        training_loss.append(data["training_loss"])
        training_maxlambda.append(data["training_lambda_max"])
        spectrum.append(data["spectrum"])

        utils.plot_losses(
            os.path.join(output_paths["fig"], experiment_name),
            training_loss,
            training_maxlambda,
        )
        utils.plot_spectrum(
            os.path.join(output_paths["fig"], experiment_name), spectrum
        )

    utils.plot_losses(
        os.path.join(output_paths["fig"], experiment_name),
        training_loss,
        training_maxlambda,
    )
    utils.plot_spectrum(os.path.join(output_paths["fig"], experiment_name), spectrum)


if __name__ == "__main__":
    run_training_flybrain_pop()
