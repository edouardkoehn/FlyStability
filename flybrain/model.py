import os
import sys

import numpy as np
import torch

from flybrain.functional import tanh, tanh_positive, tanh_strech
from flybrain.utils import get_root


class RNN:
    """
    A generic class for Recurrent Neural Networks (RNNs).

    This class enables the creation of RNN models with customizable parameters and feed-forward dynamics, while maintaining a common structural framework.

    Attributes:
        C (torch.Tensor): Connectivity matrix defining the network structure (N x N).
        W (torch.Tensor): Weights matrix representing the strength of connections between nodes (N x N).
        cell_types (torch.Tensor): Tensor specifying the type of each node, which can influence its dynamics.
        gains (torch.Tensor): Gain values applied to each cell type, modifying the input strength for each node type.
        shifts (torch.Tensor): Shift values applied to each cell type, acting as a bias term.
        H_0 (torch.Tensor): Initial activity state for each node (N x 1).
        H (torch.Tensor): Current activity state of each node (N x 1).
        dt (float): Time step used in the simulation.
        N (int): Number of nodes in the network.
        activation (function): Activation function applied to each unit (default is tanh_strech).
        A (torch.Tensor): Current activations, obtained by applying the activation function to `H`.

    Methods:
        __init__: Initializes the RNN with specified connectivity, weights, cell types, and other optional parameters.
        test_init: Validates the integrity of the model initialization to ensure compatibility of shapes and connectivity.
    """

    def __init__(self, *inp, **kwargs):
        if len(inp) == 2:
            self.construct_model_based_on_path(
                *inp
            )  # Expect two arguments: path and activation
        else:
            self.default_constructor(
                **kwargs
            )  # Expect named arguments for model parameters

    def default_constructor(
        self,
        connectivity_matrix: np.array,
        weights_matrix: np.array,
        initial_condition: np.array,
        gains_vector: np.array = None,
        shifts_vector: np.array = None,
        cell_types_vector: np.array = None,
        activation_function=tanh_strech(),
        manual_seed=41,
        dt=0.1,
    ):
        """
        Initialize the RNN model with connectivity, weights, and optional cell-type specific parameters.

        Args:
            connectivity_matrix (np.array): Binary connectivity matrix (N x N), where non-zero entries indicate connections.
            weights_matrix (np.array): Matrix of weights (N x N) corresponding to connection strengths.
            gains_vector (np.array): Array specifying the gain for each cell type; scales the input to each type.
            shifts_vector (np.array): Array specifying the shift (bias) for each cell type.
            cell_types_vector (np.array): Array defining the type of each cell (length N).
            initial_condition (np.array): Initial state for each node's activity (N x 1).
            activation_function (function, optional): Activation function for node activity. Defaults to tanh_strech.
            manual_seed (int, optional): Seed for random initialization to ensure reproducibility. Defaults to 41.
            dt (float, optional): Time step for the simulation dynamics. Defaults to 0.1.
        """
        torch.manual_seed(manual_seed)

        # Initialize matrices and parameters
        self.C = torch.tensor(
            connectivity_matrix, dtype=torch.float32, requires_grad=False
        )
        self.W = torch.tensor(weights_matrix, dtype=torch.float32, requires_grad=False)
        self.cell_types = (
            cell_types_vector
            if cell_types_vector is not None
            else np.arange(0, initial_condition.shape[0])
        )
        self.types = torch.tensor(self.cell_types, dtype=torch.int, requires_grad=False)
        self.gains = (
            gains_vector
            if gains_vector is not None
            else np.ones(len(np.unique(self.types)))
        )
        self.gains = torch.tensor(self.gains, dtype=torch.float32, requires_grad=False)
        self.shifts = (
            shifts_vector
            if shifts_vector is not None
            else np.zeros(len(np.unique(self.types)))
        )
        self.shifts = torch.tensor(
            self.shifts, dtype=torch.float32, requires_grad=False
        )

        # Initialize node states
        self.H_0 = torch.tensor(
            initial_condition, dtype=torch.float32, requires_grad=False
        )
        self.H = torch.tensor(
            initial_condition, dtype=torch.float32, requires_grad=False
        )

        # General model parameters
        self.dt = dt
        self.N = self.C.shape[0]
        self.activation = activation_function

        # Compute initial activations
        self.A = self.get_activation()

        # Test initialization
        self.test_init()

    def construct_model_based_on_path(self, path_to_matrix: str, activation: str):
        """
        Constructs a model by loading required matrices and parameters from specified file paths.

        Parameters:
        - path_to_matrix (str): Directory path containing the saved model matrices and parameters.
        Expects the following files in the directory:
            - "W.pt": Weight matrix of the model.
            - "C.pt": Connectivity matrix.
            - "H.pt": State vector of neurons.
            - "H_0.pt": Initial condition of the neurons' state vector.
            - "gains.pt": Gain values vector for neurons.
            - "shifts.pt": Shift values vector for neurons.
            - "types.pt": Cell type vector for neuron classification.
        - activation (str): Specifies the activation function to use in the model. Acceptable values:
            - "std": Standard tanh activation function.
            - "pos": Positive tanh activation function.
            - "strech": Stretched tanh activation function.

        Returns:
        - None: Initializes the model's components in-place using the specified parameters.

        Raises:
        - KeyError: If an unsupported activation function string is passed.
        - FileNotFoundError: If any required file is missing in `path_to_matrix`.

        Notes:
        - The function relies on `self.default_constructor` to initialize the model with loaded parameters.
        """
        # Load matrices and parameters from files
        W = torch.load(os.path.join(path_to_matrix, "W.pt"))
        C = torch.load(os.path.join(path_to_matrix, "C.pt"))
        H = torch.load(os.path.join(path_to_matrix, "H.pt"))
        H_0 = torch.load(os.path.join(path_to_matrix, "H_0.pt"))
        gains = torch.load(os.path.join(path_to_matrix, "gains.pt"))
        shifts = torch.load(os.path.join(path_to_matrix, "shifts.pt"))
        types = torch.load(os.path.join(path_to_matrix, "types.pt"))

        # Select activation function based on input string
        activation_function = {
            "std": tanh(),
            "pos": tanh_positive(),
            "strech": tanh_strech(),
        }[activation]

        # Initialize model using loaded parameters
        self.default_constructor(
            connectivity_matrix=C,
            weights_matrix=W,
            initial_condition=H_0,
            gains_vector=gains,
            shifts_vector=shifts,
            shifts=shifts,
            cell_types_vector=types,
            activation_function=activation_function,
        )
        return

    def test_init(self):
        """Simple method to check that the initialization is successful and shapes are compatible."""
        try:
            assert (
                self.C.shape[0] == self.C.shape[1]
            ), "Connectivity matrix must be square."
            assert self.W.shape[0] == self.W.shape[1], "Weights matrix must be square."
            assert torch.all(
                (self.W != 0) == (self.C != 0)
            ), "Weight and connectivity matrices must align in sparsity."
            assert (
                self.H.shape[0] == self.N
            ), "Initial state dimension mismatch with number of nodes."
        except AssertionError as e:
            print(f"Initialization error: {e}")
            sys.exit()

    def train(self, weight=True, shifts=False, gains=False):
        """
        Configures the model for training by setting the required parameters to require gradients.

        Args:
            weight (bool, optional): If True, enables gradient tracking for weights, allowing them to be optimized. Defaults to True.
            shift (bool, optional): If True, enables gradient tracking for shifts, allowing them to be optimized. Defaults to False.
            gains (bool, optional): If True, enables gradient tracking for gains, allowing them to be optimized. Defaults to False.
        """
        # Prevent gradients for the activity state H
        self.H.requires_grad = False

        # Set requires_grad based on input arguments
        self.W.requires_grad = weight
        self.gains.requires_grad = gains
        self.shifts.requires_grad = shifts

    def eval(self):
        """
        Set the model to evaluation mode.
        """
        self.H.requires_grad = False
        self.W.requires_grad = False
        self.C.requires_grad = False
        self.gains.requires_grad = False
        self.shifts.requires_grad = False
        self.types.requires_grad = False
        return

    # Setter methods
    def set_states(self, states_matrix: np.array):
        """
        Set the current activity states of the network.

        Args:
            states_matrix (np.array): Matrix containing the new states (N x 1).
        """
        self.H = torch.tensor(states_matrix, dtype=torch.float32).reshape(self.N)
        self.A = self.get_activation(self.H)

    def set_gains(self, gains_matrix: np.array):
        """
        Set the gains of the network.

        Args:
            gains_matrix (np.array): Array containing the new gains (N x 1).
        """
        self.gains = torch.tensor(gains_matrix)
        self.A = self.get_activation(self.H)

    def set_shifts(self, shifts_matrix: np.array):
        """
        Set the shifts of the network.

        Args:
            shifts_matrix (np.array): Array containing the new shifts (N x 1).
        """
        self.shifts = torch.tensor(shifts_matrix)
        self.A = self.get_activation(self.H)

    # Resetter methods
    def reset_states(self):
        """
        Reset the states of the network to zero.
        """
        self.H = torch.zeros((self.N), dtype=torch.float32)
        self.H_0 = torch.zeros((self.N), dtype=torch.float32)
        self.A = self.get_activation(self.H)

    def reset_weights(self):
        """
        Reset the weights and connectivity matrices to zero.
        """
        self.W = torch.zeros((self.N, self.N), dtype=torch.float32)
        self.C = torch.zeros((self.N, self.N), dtype=torch.float32)
        self.A = self.get_activation(self.H)

    # Dynamic methods
    def get_output(self):
        """
        Get the output of the network, defined as the average of the activations.

        Returns:
            torch.Tensor: The average activation of the network.
        """
        return (torch.sum(self.A) / self.N).to(dtype=torch.float32)

    def get_activation(self, states=None):
        """
        Computes the activations for the current or provided states using the activation function,
        gains, and shifts specific to each node type.

        Args:
            states (torch.Tensor, optional): An optional tensor of states for each node (N,).
                                            If None, uses the current state self.H.

        Returns:
            torch.Tensor: The activation values for each node (N,).
        """
        states = self.H if states is None else states

        # Apply the activation function using vectorized operations
        A = self.activation(states, self.gains[self.types], self.shifts[self.types])

        return A

    def __call__(self, ext_inputs=None, update: bool = True):
        """
        Computes the next step of the system and updates the state and activations if specified.

        Args:
            ext_inputs (torch.Tensor, optional): External input applied at the current step (shape: N). Defaults to a zero tensor.
            update (bool, optional): If True, updates the model's state and activity with the computed values. Defaults to True.

        Returns:
            tuple: (state_plus_one, activity_plus_one) - The updated state and activity of the system.
        """
        # Set external inputs to zero if not provided
        ext_inputs = (
            ext_inputs
            if ext_inputs is not None
            else torch.zeros(self.N, dtype=torch.float32)
        )

        # Compute the next state based on current state, weights, connectivity, and external inputs
        state_plus_one = (
            (1 - self.dt) * self.H
            + self.dt * torch.matmul(self.C * self.W, self.A)
            + ext_inputs
        )

        # Compute the next activation
        activity_plus_one = self.get_activation(states=state_plus_one)

        # Update internal states if requested
        if update:
            self.H = state_plus_one
            self.A = activity_plus_one
        return state_plus_one, activity_plus_one

    def jacobian(self):
        """
        Computes the Jacobian matrix for the instantaneous states.
        Returns:
            torch.Tensor: The Jacobian matrix (N x N).
        """
        # Initialize the identity matrix for the Kronecker term
        kronecker_matrix = torch.eye(self.N) * (1 - self.dt)

        # Compute the gains vector for each node type, and apply the derivative of the activation
        G = self.gains[self.types]
        derivative_activation = torch.diag(G / torch.cosh(self.H) ** 2) * self.dt

        # Compute the Jacobian matrix by combining the Kronecker term and weighted connectivity
        jacobian_matrix = kronecker_matrix + torch.matmul(
            self.C * self.W, derivative_activation
        )

        return jacobian_matrix

    def name(self):
        return self.activation.name()

    def save(self, model_name: str):
        """
        Save the model parameters to disk.

        Args:
            model_name (str): Name of the model to save.
        """

        os.makedirs(model_name, exist_ok=True)
        torch.save(self.W, os.path.join(model_name, "W.pt"))
        torch.save(self.C, os.path.join(model_name, "C.pt"))
        torch.save(self.H, os.path.join(model_name, "H.pt"))
        torch.save(self.H_0, os.path.join(model_name, "H0pt"))
        torch.save(self.gains, os.path.join(model_name, "gains.pt"))
        torch.save(self.shifts, os.path.join(model_name, "shifts.pt"))
        torch.save(self.types, os.path.join(model_name, "types.pt"))
        return
