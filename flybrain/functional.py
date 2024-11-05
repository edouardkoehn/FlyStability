import torch


def tanh(x):
    return torch.tanh(x)


def tanh_strech(x, gain, shift):
    return torch.tanh(gain * (x - shift))


def tanh_positive(x, gain, shift):
    return 0.5 * (1 + torch.tanh(gain * (x - shift)))


class loss:
    """Generic class to generate custom loss"""

    def __init__(self, target_value: float):
        self.target_value = target_value

    def name(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def call(self, spectrum, target_value):
        raise NotImplementedError("Subclasses should implement this method.")


class l2_norm(loss):
    """Define our custom l2 loss"""

    def __init__(self, target_value):
        super().__init__(target_value)

    def name(self):
        return f"l2Norm_{self.target_value:.2f}"

    def call(self, spectrum: torch.tensor):
        target = torch.ones(spectrum.shape[0], requires_grad=False) * self.target_value
        return torch.norm(spectrum - target)


class mse(loss):
    """Define our custom MSE loss"""

    def __init__(self, target_value):
        super().__init__(target_value)

    def name(self):
        return f"MSE_{self.target_value:.2f}"

    def call(self, spectrum: torch.tensor):
        target = torch.ones(spectrum.shape[0], requires_grad=False) * self.target_value
        return torch.sum((spectrum - target) ** 2) / spectrum.shape[0]
