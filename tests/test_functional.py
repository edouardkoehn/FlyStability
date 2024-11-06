import torch

import flybrain.functional as functional


def test_tanh():
    assert functional.tanh()(torch.tensor(0)) == 0.0
    assert torch.allclose(
        functional.tanh()(torch.tensor(10)), torch.tensor(1.0), rtol=1e-4
    )
    assert torch.allclose(
        functional.tanh()(torch.tensor(-10)), torch.tensor(-1.0), rtol=1e-4
    )


def test_tanhpos():
    assert functional.tanh_positive()(torch.tensor(0)) == 0.5
    assert functional.tanh_positive()(torch.tensor(0), 1, 0) == 0.5
    assert functional.tanh_positive()(torch.tensor(10), 1, 0) == 1.0
    assert functional.tanh_positive()(torch.tensor(-10), 1, 0) == 0.0
    assert functional.tanh_positive()(torch.tensor(1), 2, 1) == 0.5


def test_tanhstreched():
    assert functional.tanh_strech()(torch.tensor(0)) == 0.0
    assert functional.tanh_strech()(torch.tensor(0), 1, 0) == 0.0
    assert functional.tanh_strech()(torch.tensor(10), 1, 0) == 1.0
    assert functional.tanh_strech()(torch.tensor(-10), 1, 0) == -1.0
    assert functional.tanh_strech()(torch.tensor(1), 2, 1) == 0.0


def test_l2Loss():
    vector = torch.tensor([1, 3])
    assert functional.l2_norm(target_value=0.0).call(vector) == torch.sqrt(
        torch.tensor(10)
    )

    vector = torch.tensor([1, 3])
    assert functional.l2_norm(target_value=1.0).call(vector) == torch.sqrt(
        torch.tensor(4)
    )

    vector = torch.tensor([1, 3, 2])
    assert functional.l2_norm(target_value=0.0).call(vector) == torch.sqrt(
        torch.tensor(14)
    )


def test_MSELoss():
    vector = torch.tensor([1, 3])
    assert functional.mse(target_value=0.0).call(vector) == torch.tensor(10) / 2
    vector = torch.tensor([1, 3, 2])
    assert functional.mse(target_value=0.0).call(vector) == torch.tensor(14) / 3
