import numpy as np
import torch

from flybrain.model import RNN
from flybrain.qr import QR


class Lyapunov:
    """
    A class for computing the Lyapunov spectrum of a recurrent neural network (RNN).

    Attributes:
        spectrum_history (list): Stores the evolution of the Lyapunov spectrum over time.
        time_history (list): Stores the time values corresponding to each step in the simulation.
        stepON_history (list): Keeps track of orthonormalization intervals.
        spectrum (float): Computed Lyapunov spectrum after simulation.
        seeds (int): Random seed for reproducibility of the orthonormal basis initialization.
    """

    def __init__(self):
        self.spectrum_history = []
        self.time_history = []
        self.stepON_history = []
        self.spectrum = 0
        self.seeds = 41

    def compute_spectrum(
        self,
        model: RNN,
        dt: float = 0.1,
        tSim: int = 200,
        nLE: int = 1,
        tONS: float = 0.2,
        logs: bool = False,
    ):
        """
        Computes the Lyapunov spectrum for an RNN model over a given simulation time.

        Args:
            model (RNN): The RNN model to evaluate.
            dt (float): Discrete time step for the simulation.
            tSim (int): Total simulation time.
            nLE (int): Number of Lyapunov exponents to compute.
            tONS (float): Time interval between orthonormalizations.
            logs (bool): If True, logs additional information on the evolution of the spectrum.

        Returns:
            torch.Tensor: Computed Lyapunov spectrum for the specified parameters.
        """
        # Validate inputs
        assert 0 < nLE <= model.N, "nLE must be between 1 and model dimension N"
        assert dt > 0 and dt <= 1, "Time discretization dt must be in (0, 1]"

        # Adaptive orthonormalization step
        ad_step = tONS is None
        tONS = tONS if tONS is not None else dt * 2
        assert dt <= tONS, "Time step dt must not exceed ONS interval tONS"

        # Calculate simulation and orthonormalization steps
        nStep = np.ceil(tSim / dt).astype(int)
        nStepTransient = np.ceil(nStep * 0.01).astype(int)
        nstepONS = np.ceil(tONS / dt).astype(int)
        nStepTransientONS = np.ceil(nStep * 0.01).astype(int)
        nONS = np.ceil((nStepTransientONS + nStep) / nstepONS) - 1

        # Initialize state variables
        N = model.N
        N_steps = nStep + nStepTransient + nStepTransientONS
        h = torch.zeros((N, N_steps), dtype=torch.float32)
        h[:, 0] = model.H

        # Initialize random orthonormal basis for Lyapunov exponents
        torch.manual_seed(self.seeds)
        rd = torch.randn(N, nLE, dtype=torch.float32)
        q, r = QR.apply(rd)

        LS = torch.zeros(nLE)  # Lyapunov spectrum accumulator
        t = 1.0  # Time tracker

        # Optional logging of initial orthonormalization step
        if logs:
            self.stepON_history.append(nstepONS)
            self.time_history.append(t)

        # Main simulation loop
        for n in range(N_steps - 1):
            # Update network dynamics
            model()
            h[:, n + 1] = model.H

            # Skip transient steps for Lyapunov computation
            if n + 1 > nStepTransient:
                # Perturbation dynamics using the Jacobian
                D = model.jacobian()
                q = torch.mm(D, q)  # Evolve orthonormal basis

                # Perform QR decomposition at each orthonormalization step
                if (n + 1) % nstepONS == 0 or n == N_steps - 2:
                    q, r = QR.apply(q)  # Orthonormalize

                    # Adjust orthonormalization interval if adaptive step is enabled
                    if ad_step and nLE > 1:
                        k2 = r[0, 0] / r[nLE - 1, nLE - 1]
                        lambda_min, lambda_max = r[nLE - 1, nLE - 1], r[0, 0]
                        updated_nstepONS = np.ceil(
                            torch.log(k2).item() / (lambda_max - lambda_min) / dt
                        )
                        if not np.isnan(updated_nstepONS):
                            nstepONS += int(
                                np.sign(updated_nstepONS - nstepONS)
                                * np.ceil(nstepONS * 0.1)
                            )
                            if logs:
                                self.stepON_history.append(nstepONS)

                    # Accumulate log of diagonal elements of R for Lyapunov spectrum post-transient
                    if n + 1 > nStepTransientONS + nStepTransient:
                        LS += torch.log(r.diag().abs())
                        if logs:
                            self.spectrum_history.append((LS / t).numpy())
                            self.time_history.append(t)
                    t = (n + 1) * dt  # Update time

        LS = LS / t
        return LS
