import torch

import flybrain.qr as qr


def test_QR():
    A_our_implementation = torch.tensor(
        [[3, 0, 0], [0, 6, 0], [0, 0, 5]], dtype=float, requires_grad=True
    )  # Arbitrary-shaped matrix example

    Q_our, R = qr.QR.apply(A_our_implementation)
    # Backpropagate some gradient
    R.sum().backward()

    A_torch_implementation = torch.tensor(
        [[3, 0, 0], [0, 6, 0], [0, 0, 5]], dtype=float, requires_grad=True
    )
    Q_torch, R = torch.linalg.qr(A_torch_implementation)
    R.sum().backward()
    assert torch.equal(Q_torch, Q_our)
    assert torch.allclose(
        A_torch_implementation.grad, A_our_implementation.grad, atol=1e-3
    )
