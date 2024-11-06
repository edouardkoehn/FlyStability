import numpy as np
import torch

import flybrain.functional as func
import flybrain.model as model
from flybrain.lyapunov import Lyapunov


def test_lyapunov_spectrum():
    C = np.array(
        [
            [
                0,
                -1,
            ],
            [1, 0],
        ]
    )
    W = np.array(
        [
            [
                0,
                0.9,
            ],
            [0.5, 0],
        ]
    )
    c0 = np.array([[-0.5, -0.1]])
    RNN = model.RNN(
        connectivity_matrix=C,
        weights_matrix=W,
        initial_condition=c0[0, :],
        activation_function=func.tanh(),
    )
    RNN.reset_states()
    LE_spectrum = Lyapunov().compute_spectrum(RNN, tSim=1001, dt=0.1, tONS=10, nLE=2)
    print(LE_spectrum)
    assert torch.allclose(LE_spectrum, torch.tensor([-1.0056, -1.0059]), atol=1e-3)
