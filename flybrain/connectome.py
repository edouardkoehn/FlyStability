import copy
import os
import pickle

import numpy as np

from flybrain.utils import get_root

NEUROTRANSMITTER_2_SIGNS = {0: "Inhibitorry", 1: "Excitatory"}
NEUROTRANSMITTER_2_TYPES = {
    0: "gaba",
    1: "acetylcholine",
    2: "glutamate",
    3: "serotonin",
    4: "octopamine",
    5: "dopamine",
    6: "neither",
}
CELLFIBER_2_TYPES = {
    0: "PDM25",
    1: "ADM09",
    2: "PDM30",
    3: "PDM24",
    4: "PDL06",
    5: "PDM10",
    6: "PDM18",
    7: "PDL19",
    8: "ADL07",
    9: "PDM07",
    10: "ADL16",
    11: "PDM15",
    12: "PDL17",
    13: "ADL09",
    14: "PDM09",
    15: "PDL07",
    16: "PDM20",
    17: "PDM08",
    18: "PDL13",
    19: "PVL17",
    20: "ADL04",
    21: "ADL01",
    22: "ADL24",
    23: "PDL10",
    24: "ADL14",
    25: "AVM08",
    26: "PDL25",
    27: "ADL05",
    28: "PDL14",
    29: "ADL23",
    30: "PDL11",
    31: "AVM04",
    32: "AVM07",
    33: "PDM31",
    34: "PDM05",
    35: "ADL20",
    36: "ADL10",
    37: "ADL02",
    38: "PDL23",
    39: "PDM04",
    40: "PDM12",
    41: "ADM04",
    42: "ADL13",
    43: "PDL27",
    44: "ADL30",
    45: "PVL04",
    46: "ADM07",
    47: "PDM22",
    48: "ADM01",
    49: "AVL08",
    50: "PDM29",
    51: "AVM13",
    52: "PDL26",
    53: "PVL06",
    54: "ADM05",
    55: "ADL25",
    56: "PDM27",
    57: "ADL18",
    58: "ADM02",
    59: "ADM10",
    60: "PDL01",
    61: "PDL04",
    62: "PDL24",
    63: "PDL15",
    64: "PDL16",
    65: "AVL06",
    66: "AVL23",
    67: "ADL03",
    68: "PDM11",
    69: "PDL18",
    70: "ADL15",
    71: "ADL17",
    72: "ADL11",
    73: "ADL19",
    74: "AVL18",
    75: "PDL08",
    76: "AVM06",
    77: "PDL22",
    78: "PDM13",
    79: "ADL12",
    80: "PDL02",
    81: "PDL03",
    82: "ADL27",
    83: "ADL22",
    84: "AVM15",
    85: "AVL10",
    86: "PDM21",
    87: "PDL28",
    88: "PDL12",
    89: "AVL12",
    90: "AVL07",
    91: "ADL29",
    92: "ADL21",
    93: "ADM11",
    94: "PDL21",
    95: "PDM06",
    96: "PVL20",
    97: "PVL02",
    98: "AVM03",
    99: "AVL27",
    100: "ADM12",
    101: "ADM08",
    102: "ADL06",
    103: "PVL01",
    104: "PVM12",
    105: "ADL08",
    106: "AVL13",
    107: "AVL04",
    108: "PVL10",
    109: "PVL11",
    110: "PVM03",
    111: "PVM14",
    112: "PVM09",
    113: "PDM17",
    114: "PVL21",
    115: "AVM14",
    116: "AVL03",
    117: "ADL28",
    118: "AVL02",
    119: "AVM02",
    120: "PVL03",
    121: "PVM11",
    122: "AVM01",
    123: "AVM18",
    124: "PVL09",
    125: "PDL09",
    126: "PVM10",
    127: "PVL12",
    128: "PVL08",
    129: "AVL01",
    130: "PDM28",
    131: "PVL13",
    132: "ADM03",
    133: "PVL19",
    134: "AVM17",
    135: "AVL05",
    136: "AVM16",
    137: "PDM32",
    138: "PDM26",
    139: "PVL14",
    140: "PVL05",
    141: "AVL25",
    142: "PVL18",
    143: "PDM35",
    144: "PDM16",
    145: "PDM33",
    146: "AVL11",
    147: "AVM09",
    148: "PDM34",
    149: "PVM05",
    150: "AVL14",
    151: "AVM19",
    152: "PVL15",
    153: "PVM08",
    154: "PVM13",
    155: "ADM06",
    156: "PVM01",
    157: "PDM14",
    158: "PVM16",
    159: "ADL26",
    160: "PVL16",
    161: "AVL20",
    162: "PDM23",
    163: "PVM15",
    164: "AVL09",
    165: "PVL07",
    166: "PDM19",
    167: "AVL16",
    168: "PVM02",
    169: "AVL26",
    170: "AVM11",
    171: "PVM06",
    172: "AVL19",
    173: "AVL22",
    174: "PVM04",
    175: "AVL24",
    176: "AVM20",
    177: "PVM17",
    178: "PVM07",
    179: "AVM22",
    180: "AVL17",
    181: "AVL21",
    182: "AVM21",
    183: "AVM12",
    184: "AVM05",
    185: "AVL15",
}

"""Methods for handling the connectome data"""


def load_flybrain(ROI: str = "EB", types: str = "neurotransmitter"):
    """
    Loads and processes connectome data for the specified region of interest (ROI) in the fly brain.
    By convention Wij correspond to the link from neuron i to neuron j. Thus one row of W correspond to
    all the connection from node i to the other neurons (outputs) and one column of the W correspond to
    all the connection that go toward i (inputs).
    Parameters
    ----------
    ROI : str, optional
        Brain region of interest (ROI) to load, by default "ellipsoid_body". Currently, only
        "ellipsoid_body" is supported.
    types : str, optional
        Type of cell classification to return. Options are:
        - 'neurotransmitter': Returns data on neurotransmitter assignments.
        - 'cell_fibers': Returns data on cell fiber assignments.

    Returns
    -------
    tuple
        A tuple containing:
        - W : ndarray
            Adjacency matrix of the connectome data.
        - C : ndarray
            Adjusted connectivity matrix with excitatory/inhibitory classifications applied.
        - neuroTransmitter_type or cellFiber_types : dict
            Dictionary containing either neurotransmitter or cell fiber type assignments,
            based on the value of `types`.
    """

    general_path = os.path.join(get_root(), "data", "connectomics", ROI)
    W = np.load(os.path.join(general_path, "adjacency_scc.npy")).T
    neuroTransmitter_sign = np.load(os.path.join(general_path, "ei_neuron_types.npy"))
    neuron_id = np.load(os.path.join(general_path, "nid_scc.npy"))

    # Compute C
    neuroTransmitter_sign = np.diag(2 * neuroTransmitter_sign - 1)
    C = np.where(W > 0, 1, np.where(W < 0, 0, 0))
    C = np.matmul(neuroTransmitter_sign, C)

    if types == "neurotransmitter":
        with open(
            os.path.join(general_path, "neurotransmitter_assignments.pkl"), "rb"
        ) as f:
            neuroTransmitter_type = pickle.load(f)
        types = [neuroTransmitter_type[id] for id in neuron_id]

    elif types == "cell_fibers":
        with open(
            os.path.join(general_path, "cell_body_fiber_assignments.pkl"), "rb"
        ) as f:
            cellFiber_types = pickle.load(f)
        types = [cellFiber_types[id] for id in neuron_id]
    return {"weights": W, "connectivity": C, "types": types}


def sinkhorn_knopp(A, max_iter=10, tol=1e-6, epsilon=1e-10):
    """
    Sinkhorn-Knopp algorithm for matrix normalization to get a doubly stochastic matrix
    Such that the excit/inhib are balanced and the norm of the columns row equal to one
    """
    # Ensure the matrix is non-negative
    A = np.abs(A)

    for _ in range(max_iter):
        # Normalize rows
        row_sums = 2 * A.sum(axis=1, keepdims=True) + epsilon
        A = A / row_sums

        # Normalize columns
        col_sums = 2 * A.sum(axis=0, keepdims=True) + epsilon
        A = A / col_sums

        # Check for convergence
        if np.allclose(A.sum(axis=1), 1, atol=tol) and np.allclose(
            A.sum(axis=0), 1, atol=tol
        ):
            break
    return A


def normalize_connectome(W, C):
    """Simple method to normalize using Sinkhorn the W matrix"""
    C_inhib = copy.deepcopy(C)
    C_inhib[np.where(C > 0)] = 0
    C_inhib = C_inhib * -1
    W_inhib = W * C_inhib

    C_exci = copy.deepcopy(C)
    C_exci[np.where(C < 0)] = 0
    W_exci = W * C_exci

    W_exci_N = sinkhorn_knopp(W_exci)
    W_inhib_N = sinkhorn_knopp(W_inhib)

    return (W_inhib_N + W_exci_N), C
