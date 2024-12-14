import os

import click
import numpy as np

import flybrain.functional as functional
import flybrain.model as model
import flybrain.utils as utils
from flybrain.training import train_RD_RNN_with_fixed_param


@click.command()
@click.option("--n", type=int, required=True, help="Number of neurons in the model")
@click.option("--n_samples", type=int, required=True, help="Number of samples used")
@click.option("--nle", type=int, required=True, help="Number of Lyapunov exponents")
@click.option("--g", type=float, required=True, help="Synaptic distribution parameter")
@click.option("--n_epochs", type=int, default=10, help="Number of epochs")
@click.option(
    "--activation",
    type=click.Choice(["tanh", "tanh_pos", "tanh_streched"]),
    required=True,
    help="Activation function",
)
@click.option(
    "--loss",
    type=click.Choice(["l2", "MSE", "Sinai"]),
    required=True,
    help="Loss function",
)
@click.option("--target", type=float, default=0.0, help="Target Lyapunov vector")
@click.option(
    "--number_param",
    type=int,
    nargs=3,
    default=[100, 0, 0],
    help="Number of parameters to optimize",
)
@click.option(
    "--tons", type=float, default=0.2, help="Step size between QR factorization"
)
@click.option("--tsim", type=int, default=200, help="Length of simulation [tau]")
@click.option("--train_weights", type=bool, default=True, help="Optimize weights")
@click.option("--train_shifts", type=bool, default=False, help="Optimize shifts")
@click.option("--train_gains", type=bool, default=False, help="Optimize gains")
@click.option("--lr", type=float, default=1e-3, help="Learning rate")
def run_training_RD_RNN_fixed_param(
    n,
    n_samples,
    nle,
    loss,
    target,
    tons,
    tsim,
    g,
    n_epochs,
    lr,
    train_weights,
    train_shifts,
    train_gains,
    activation,
    number_param,
):
    """Main function to train RD-RNN with fixed parameters."""
    # Seed initialization for reproducibility
    np.random.seed(30)

    # Generate output paths
    #ROOT = utils.get_root()
    ROOT="/pscratch/sd/e/ekoehn/FlyStability"
    output_paths = {
        name: os.path.join(ROOT, "data", name, "rd_RNN_fixed_param")
        for name in ["logs", "fig", "models"]
    }
    for path in output_paths.values():
        os.makedirs(path, exist_ok=True)

    # Setup loss and activation functions
    loss_func = {
        "l2": functional.l2_norm(target),
        "MSE": functional.mse(target),
        "Sinai": functional.sinai_entropy(),
    }[loss]
    maximize_options = loss == "Sinai"

    activation_func = {
        "tanh": functional.tanh(),
        "tanh_pos": functional.tanh_positive(),
        "tanh_streched": functional.tanh_strech(),
    }[activation]

    # Validate parameters
    validate_parameters(n, number_param)

    # Create experiment name
    experiment_name = generate_experiment_name(
        activation_func,
        train_weights,
        train_shifts,
        train_gains,
        n,
        lr,
        nle,
        n_epochs,
        loss_func,
        g,
        tons,
        tsim,
        number_param,
    )

    # Initialize results
    training_loss, training_maxlambda, spectrum = [], [], []

    for sample in range(n_samples):
        # Construct model
        rnn = construct_rnn_model(n, g, activation_func)

        # Train model
        run_name = f"{experiment_name}_Sample{sample}"
        train_RD_RNN_with_fixed_param(
            rnn_model=rnn,
            loss=loss_func,
            nLE=nle,
            N_epoch=n_epochs,
            tSim=tsim,
            tONs=tons,
            dt=0.1,
            train_weights=train_weights,
            train_shifts=train_shifts,
            train_gains=train_gains,
            lr=lr,
            run_name=run_name,
            run_type="rd_RNN_fixed_param",
            maximize_options=maximize_options,
            number_para_used=number_param,
        )

        # Store logs
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


def validate_parameters(n, number_param):
    """Ensure parameter counts are within valid ranges."""
    assert 0 <= number_param[0] <= n**2, "Invalid number of weight parameters"
    assert 0 <= number_param[1] <= n, "Invalid number of shift parameters"
    assert 0 <= number_param[2] <= n, "Invalid number of gain parameters"


def generate_experiment_name(
    activation_func,
    train_weights,
    train_shifts,
    train_gains,
    n,
    lr,
    nle,
    n_epochs,
    loss_func,
    g,
    tons,
    tsim,
    number_param,
):
    """Generate a unique experiment name based on parameters."""
    param_str = "_".join(map(str, number_param))
    return (
        f"{activation_func.name()}_Weights{train_weights}_Shifts{train_shifts}_"
        f"Gains{train_gains}_N{n}_lr{lr}_NLE{nle}_Epochs{n_epochs}_"
        f"{loss_func.name()}_g{g}_Tons{tons}_Tsim{tsim}_Param{param_str}"
    )


def construct_rnn_model(n, g, activation_func):
    """Construct the RNN model."""
    W, C = utils.construct_Random_Matrix_simple(n_neurons=n, coupling=g, showplot=False)
    c0 = np.random.normal(0.0, 1.0, n)
    return model.RNN(
        connectivity_matrix=C,
        weights_matrix=W,
        initial_condition=c0,
        activation_function=activation_func,
    )


if __name__ == "__main__":
    run_training_RD_RNN_fixed_param()
