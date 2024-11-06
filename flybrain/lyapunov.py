import numpy as np
import torch

from flybrain.model import RNN
from flybrain.qr import QR


class Lyapunov:
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
        tONS: int = 0.2,
        logs: bool = False,
    ):
        # Set parameters
        assert nLE <= model.N, "Number of Lyapunov exponents must be smaller than N"
        assert nLE > 0, "Number of Lyapunov exponents must be greater than 0"
        assert dt <= 1, "Time discretization must be less than or equal to 1"
        if tONS is None:
            tONS = dt * 2  # Set the tONS of 1% of the time simulation
            ad_step = True  # Activate the adaptive time ons
        else:
            ad_step = False
        assert dt <= tONS, "dt must be less than or equal to ONS interval"

        # Calculate steps and intervals
        nStep = np.ceil(tSim / dt)  # Total number of steps
        nStepTransient = np.ceil(nStep * 0.01)
        # nStepTransient = 10
        nstepONS = np.ceil(tONS / dt)  # Steps between orthonormalizations
        nStepTransientONS = np.ceil(nStep * 0.01)
        # nStepTransientONS = 20  # Steps during transient of ONS
        nONS = (
            np.ceil((nStepTransientONS + nStep) / nstepONS) - 1
        )  # Number of ONS steps

        # Initialize the network
        N = model.N
        N_steps = int(np.ceil(nStep + nStepTransient + nStepTransientONS))
        h = torch.zeros((N, N_steps))  # Preallocate local fields
        h[:, 0] = model.H  # Initialize network state

        # Initialize orthonormal basis
        torch.manual_seed(self.seeds)  # Set seed for orthonormal system
        rd = torch.randn(N, nLE, requires_grad=False)
        q, r = QR.apply(rd)

        LS = torch.zeros(nLE)  # Initialize Lyapunov spectrum
        t = 1.0  # Set time to 0

        # Initialize array to store convergence of Lyapunov spectrum
        if logs:
            self.stepON_history.append(nstepONS)
            self.time_history.append(t)

        for n in range(N_steps - 1):
            # Compute the dynamics
            model()
            h[:, n + 1] = model.H

            if n + 1 > nStepTransient:
                # Compute the dynamics of the perturbation
                D = model.jacobian()  # Jacobian
                q = torch.mm(D, q)  # Evolve orthonormal system using Jacobian

                if (np.mod(n + 1, nstepONS) == 0) | (
                    n == nStep + nStepTransient + nStepTransientONS - 1
                ):

                    q, r = QR.apply(q)  # Perform QR-decomposition

                    if ad_step and nLE > 1:
                        k2 = torch.diag(r)[0] / torch.diag(r)[nLE - 1]
                        lambda_min, lambda_max = (
                            torch.diag(r)[nLE - 1],
                            torch.diag(r)[0],
                        )
                        updated_nstepONS = (
                            ((torch.log(k2) / (lambda_max - lambda_min)) / dt)
                            .detach()
                            .numpy()
                        )
                        if not np.isnan(updated_nstepONS):
                            updated_nstepONS = np.ceil(updated_nstepONS)

                            if (updated_nstepONS > nstepONS) | (
                                updated_nstepONS < nstepONS
                            ):

                                nstepONS = np.ceil(
                                    np.max(
                                        [
                                            1,
                                            nstepONS
                                            + np.sign(updated_nstepONS - nstepONS)
                                            * np.ceil(nstepONS * 0.1),
                                        ]
                                    )
                                )
                                nstepONS = nstepONS
                            if logs:
                                self.stepON_history.append(nstepONS)

                    if n + 1 > nStepTransientONS + nStepTransient:
                        lambda_i = torch.log(abs(torch.diag(r)))
                        LS += lambda_i  # Collect log of diagonal elements of R for Lyapunov spectrum
                        if logs:
                            self.spectrum_history.append(LS.numpy() / t)
                            self.time_history.append(t)
                    t = (n + 1) * dt  # Increment time

        LS = LS / t
        return LS

    # TO DO BruteForceSpectrum
    # To Do adding the adaptive tons
