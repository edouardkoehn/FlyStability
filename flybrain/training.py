import json
import os
import time

import numpy as np
import torch

import flybrain.functional as functional
import flybrain.model as model
import flybrain.utils as utils
from flybrain.lyapunov import Lyapunov


def train_RD_RNN(
    model: model.RNN,
    loss: functional.loss,
    nLE: int,
    N_epoch: int,
    tSim: int,
    tONs: float,
    dt: float,
    train_weights: bool,
    train_shifts: bool,
    train_gains: bool,
    lr: float,
    run_name: str,
):
    """
    Trains a recurrent neural network (RNN) model with random connectivity, utilizing Lyapunov exponents as feedback.

    Parameters:
    - model (model.RNN): RNN model instance with random connectivity matrix.
    - loss (functional.loss): Loss function to minimize during training.
    - nLE (int): Number of Lyapunov exponents to compute.
    - N_epoch (int): Number of training epochs.
    - tSim (int): Simulation time for each epoch.
    - tONs (float): Time onset parameter for Lyapunov spectrum computation.
    - dt (float): Time step for Lyapunov spectrum calculation.
    - train_weights (bool): Whether to train weight matrix `W`.
    - train_shifts (bool): Whether to train shift parameters.
    - train_gains (bool): Whether to train gain parameters.
    - lr (float): Learning rate for the optimizer.
    - run_name (str): Name for the current training run, used for saving logs and model.
    """

    # Set up paths for saving logs and models
    ROOT_PATH = utils.get_root()
    output_logs_path = os.path.join(ROOT_PATH, "data", "logs", "rd_RNN")
    output_model_path = os.path.join(ROOT_PATH, "data", "model", "rd_RNN")

    # Configure the model for training with specified parameters
    model.train(weight=train_weights, shifts=train_shifts, gains=train_gains)
    model.reset_states()

    # Set up optimizer based on selected training parameters
    optimizer = set_optimizer(
        lr=lr,
        train_weights=train_weights,
        train_gains=train_gains,
        train_shifts=train_shifts,
    )
    optimizer.zero_grad()
    c0 = model.H_0.detach().numpy()

    # Initialize logging arrays for errors, spectrum, and gradient norms
    error = np.zeros(N_epoch)
    spectrum_hist = np.zeros((nLE, N_epoch))
    maxLambda_hist = np.zeros(N_epoch)
    grad_norm_gains = np.zeros(N_epoch)
    grad_norm_shifts = np.zeros(N_epoch)
    grad_norm_weights = np.zeros(N_epoch)

    # Training loop with logging and saving progress
    t0 = time.time()
    print(f"{run_name}: {time.time() - t0:.2f} - Starting training...")

    for epoch in range(N_epoch):
        # Reset model states and optimizer gradients
        model.set_states(c0)
        optimizer.zero_grad()

        # Compute Lyapunov spectrum and loss
        spectrum = Lyapunov().compute_spectrum(
            model, tSim=tSim, dt=dt, tONS=tONs, nLE=nLE
        )
        loss_val = loss.call(spectrum)

        # Backpropagation
        loss_val.backward()

        # Gradient clipping and storing gradient norms if applicable
        if train_shifts:
            torch.nn.utils.clip_grad_norm_([model.shifts], max_norm=1000, norm_type=2.0)
            grad_norm_shifts[epoch] = torch.norm(model.shifts.grad)
        if train_gains:
            torch.nn.utils.clip_grad_norm_([model.gains], max_norm=1000, norm_type=2.0)
            grad_norm_gains[epoch] = torch.norm(model.gains.grad)
        if train_weights:
            torch.nn.utils.clip_grad_norm_([model.W], max_norm=1000, norm_type=2.0)
            grad_norm_weights[epoch] = torch.norm(model.W.grad)

        optimizer.step()

        # Log error and spectrum history
        error[epoch] = loss_val.detach().item()
        spectrum_hist[:, epoch] = spectrum.detach().numpy()
        maxLambda_hist[epoch] = spectrum.max().item()

        # Periodically log progress and save data
        if epoch % 10 == 0 or epoch == N_epoch - 1:
            print(
                f"{epoch}-Loss: {error[epoch]:.3f} - Lambda_max: {maxLambda_hist[epoch]:.3f}",
                end=" ",
            )
            if train_shifts:
                print(f"Shifts_norm: {grad_norm_shifts[epoch]:.3f}", end=" ")
            if train_gains:
                print(f"Gains_norm: {grad_norm_gains[epoch]:.3f}", end=" ")
            if train_weights:
                print(f"Weights_norm: {grad_norm_weights[epoch]:.3f}", end=" ")
            print()

            # Save detailed spectrum logs every 10 epochs
            spectrum_full = Lyapunov().compute_spectrum(
                model, tSim=tSim, dt=dt, tONS=tONs, nLE=model.N - 1
            )
            with open(
                os.path.join(output_logs_path, run_name + "_logs.json"), "w"
            ) as f:
                json.dump(
                    {
                        "training_loss": error.tolist(),
                        "training_lambda_max": maxLambda_hist.tolist(),
                        "spectrum": spectrum_full.tolist(),
                        "grad_gains": grad_norm_gains.tolist(),
                        "grad_shifts": grad_norm_shifts.tolist(),
                        "grad_weights": grad_norm_weights.tolist(),
                    },
                    f,
                )
            model.save(os.path.join(output_model_path, run_name))

    print(f"{run_name}: {time.time() - t0:.2f} - Training completed.")
    return


def set_optimizer(
    lr: float, train_weights: bool, train_shifts: bool, train_gains: bool
):
    """
    Configures and returns an Adam optimizer based on selected training parameters.

    Parameters:
    - lr (float): Learning rate for the optimizer.
    - train_weights (bool): If True, include model weights in optimization.
    - train_shifts (bool): If True, include shift parameters in optimization.
    - train_gains (bool): If True, include gain parameters in optimization.

    Returns:
    - torch.optim.Adam: Configured optimizer for selected parameters.
    """
    # Collect parameters for optimization
    parameters = []
    if train_weights:
        parameters.append(model.W)
    if train_shifts:
        parameters.append(model.shifts)
    if train_gains:
        parameters.append(model.gains)

    # Initialize and return the optimizer with the selected parameters
    return torch.optim.Adam(parameters, lr=lr)
