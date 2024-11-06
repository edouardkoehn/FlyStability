import numpy as np
import torch

import flybrain.model as model

C = np.array([[0, 1, 0], [0, 0, 1], [0, 1, 0]])
W = np.array([[0, 1, 0], [0, 0, 1], [0, 1, 0]])
co = np.array([1, 1, 1])


def test_RNN_init():
    RNN = model.RNN(connectivity_matrix=C, weights_matrix=W, initial_condition=co)
    assert torch.equal(RNN.W, torch.tensor(W))
    assert torch.equal(RNN.C, torch.tensor(C))
    assert torch.equal(RNN.H, torch.tensor(co))
    assert torch.equal(RNN.H_0, torch.tensor(co))
    assert RNN.N == 3
    assert torch.equal(RNN.types, torch.tensor([0, 1, 2]))
    assert RNN.name() == "tanh_streched"

    RNN = model.RNN(
        connectivity_matrix=C,
        weights_matrix=W,
        initial_condition=co,
        cell_types_vector=np.array([0, 0, 1]),
    )
    assert RNN.gains.shape[0] == 2
    assert RNN.shifts.shape[0] == 2
    assert torch.equal(RNN.gains, torch.ones(2))
    assert torch.equal(RNN.shifts, torch.zeros(2))

    RNN = model.RNN(
        connectivity_matrix=C,
        weights_matrix=W,
        initial_condition=co,
        cell_types_vector=np.array([0, 0, 1]),
        gains_vector=np.array([10, 0]),
    )
    assert torch.equal(RNN.gains, torch.tensor([10, 0]))
    assert torch.equal(RNN.shifts, torch.tensor([0, 0]))

    RNN = model.RNN(
        connectivity_matrix=C,
        weights_matrix=W,
        initial_condition=co,
        cell_types_vector=np.array([0, 0, 1]),
        shifts_vector=np.array([40, 10]),
    )
    assert torch.equal(RNN.shifts, torch.tensor([40, 10]))
    assert torch.equal(RNN.gains, torch.tensor([1, 1]))
    assert RNN.dt == 0.1
    return


def test_train():
    RNN = model.RNN(connectivity_matrix=C, weights_matrix=W, initial_condition=co)
    RNN.train(weight=True)
    assert RNN.W.requires_grad == True

    RNN.train(gains=True, shifts=True)
    assert RNN.gains.requires_grad == True
    assert RNN.shifts.requires_grad == True

    RNN.train(weight=True, gains=True, shifts=True)
    assert RNN.W.requires_grad == True
    assert RNN.gains.requires_grad == True
    assert RNN.shifts.requires_grad == True
    assert RNN.types.requires_grad == False
    assert RNN.C.requires_grad == False
    return


def test_eval():
    RNN = model.RNN(connectivity_matrix=C, weights_matrix=W, initial_condition=co)
    RNN.eval()
    assert RNN.W.requires_grad == False
    assert RNN.gains.requires_grad == False
    assert RNN.shifts.requires_grad == False
    assert RNN.types.requires_grad == False
    assert RNN.C.requires_grad == False
    return


def test_reset():
    RNN = model.RNN(connectivity_matrix=C, weights_matrix=W, initial_condition=co)
    assert torch.equal(RNN.H_0, torch.tensor(co))
    assert torch.equal(RNN.A, RNN.get_activation(states=torch.tensor(co)))

    RNN.reset_states()
    assert torch.equal(RNN.H_0, torch.zeros(3))
    assert torch.equal(RNN.A, RNN.get_activation(states=torch.zeros(3)))

    RNN = model.RNN(connectivity_matrix=C, weights_matrix=W, initial_condition=co)
    assert torch.equal(RNN.W, torch.tensor(W))
    assert torch.equal(RNN.A, RNN.get_activation(states=torch.tensor(co)))
    RNN.reset_weights()
    assert torch.equal(RNN.W, torch.zeros(3, 3))
    return


def test_getoutput():
    RNN = model.RNN(connectivity_matrix=C, weights_matrix=W, initial_condition=co)
    output = RNN.get_output()
    expected_output = torch.tensor(0.761, dtype=torch.float32)

    print(output)
    assert torch.isclose(output, expected_output, atol=1e-3)


def test_getactivation():
    RNN = model.RNN(connectivity_matrix=C, weights_matrix=W, initial_condition=co)
    print(RNN.get_activation(states=torch.zeros(3)))
    assert torch.all(
        torch.isclose(
            RNN.get_activation(states=torch.zeros(3)), torch.zeros(3), atol=1e-3
        )
    )
