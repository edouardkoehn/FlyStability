import torch


class functional:
    def name(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def call(self, x):
        raise NotImplementedError("Subclasses should implement this method.")


class tanh(functional):
    """f(x)=tanh(x)"""

    def name(self):
        return "tanh"

    def __call__(self, x, gain=1, shift=0):
        return torch.tanh(x)


class tanh_strech(functional):
    """f(x)=tanh(gain*(x-shift)))"""

    def name(self):
        return "tanh_streched"

    def __call__(self, x, gain=1, shift=0):
        return torch.tanh(gain * (x - shift))


class tanh_positive(functional):
    """f(x)=0.5(1+tanh(gain*(x-shift)))"""

    def name(self):
        return "tanh_positive"

    def __call__(self, x, gain=1, shift=0):
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
    """Define our custom l2 loss with a constant target vector"""

    def __init__(self, target_value):
        super().__init__(target_value)

    def name(self):
        return f"l2Norm_{self.target_value:.2f}"

    def call(self, spectrum: torch.tensor):
        target = torch.ones(spectrum.shape[0], requires_grad=False) * self.target_value
        return torch.norm(spectrum - target)


class mse(loss):
    """Define our custom MSE loss with a constant target vector"""

    def __init__(self, target_value):
        super().__init__(target_value)

    def name(self):
        return f"MSE_{self.target_value:.2f}"

    def call(self, spectrum: torch.tensor):
        target = torch.ones(spectrum.shape[0], requires_grad=False) * self.target_value
        return torch.sum((spectrum - target) ** 2) / spectrum.shape[0]


class mse_custom(loss):
    """Define our custom MSE loss. It allow the user to define the target vector"""

    def __init__(self):
        super().__init__(0.0)

    def name(self):
        return f"MSE_1.25_-1.25"

    def call(self, spectrum: torch.tensor):
        target = torch.linspace(
            1.24,
            -1.25,
            spectrum.shape[0],
            requires_grad=False,
        )
        return torch.sum((spectrum - target) ** 2) / spectrum.shape[0]


class sinai_entropy(loss):
    """Loss based on the Sinai-Kolmogorov entropy defined from the Lyapunov exponent. See Englekamp et al. 2021, for more details"""

    def __init__(self):
        super().__init__(0.0)

    def name(self):
        return f"Entropy"

    def call(self, spectrum: torch.tensor):
        return torch.sum(torch.nn.functional.relu(spectrum)) - torch.sum(
            torch.nn.functional.relu(-spectrum)
        )
