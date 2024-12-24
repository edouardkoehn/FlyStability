# 🪰  RandomNet 🪰
This repository contains all the code related to the RandomNet project. This project was part of my master thesis conducted in the [Neural Systems and Machine Learning Lab](https://bouchardlab.lbl.gov/) under the supervision of K. Bouchard.

[Edouard Koehn](edouard.koehn@berkeley.edu) \
15.01.2024


## Index
- [Abstract](##Abstract)

- [Structure of the repo](##Structure-of-the-code)

- [Data usage](##Data-usage)

- [Workflows](##Workflows)

- [Installation](##Installation)
## Abstract

## Structure of the code
This repository is designed to serve as a framework for the RandomNet project and any future related projects. All classes are implemented as abstract classes to ensure reusability in other projects.

The main idea is that when creating a new experiment, you can leverage the existing framework and only need to implement your specific workflow. Each workflow is callable through the command line, enhancing the reproducibility of the code.

The current module allows the user to:
- Evaluate the phase transition of RNNs in terms of the Lyapunov spectrum of the system.
- Train a random RNN using a loss function based on the Lyapunov spectrum of the system.
- Train a biologically constrained RNN.

The structure of the repository is as follows:
```bash
FlyStability
|
|-- .github/            # Continuous Integration setup using GitHub Actions
|
|-- data/               # Storage for connectome data and workflow exports
|   |-- connectomics/   # Raw connectomics data
|   |-- fig/            # Generated figures
|   |-- logs/           # Logs for workflows
|   |-- model/          # Trained models and related data
|
|-- bash/               # Bash scripts for automating experiments
|
|-- flybrain/           # Core module containing workflows and utilities
|   |-- workflows/      # Workflow implementations
|   |-- __init__.py     # Module initialization
|   |-- connectome.py   # Connectome processing utilities
|   |-- functional.py   # Functional processing utilities
|   |-- lyapunov.py     # Lyapunov spectrum calculations
|   |-- model.py        # Model definitions
|   |-- qr.py           # Custom QR factorization routines
|   |-- training.py     # Training scripts
|   |-- utils.py        # General utility functions
|
|-- tests/              # Unit tests for the framework
|
|-- notebooks/          # Notebooks generating plots for the report
|-- .gitignore          # Git ignore file
|-- pyproject.toml      # Project configuration and dependencies
|-- README.md           # Repository documentation
```
## Model and Data
A) The model: \
We used the standard dynamic defined by : (1) $ \frac{\partial{h_i}}{\partial{t}}=-h_i + \sum_{j=0}^NC_{ij}W_{ij}*(\phi(h_j))$\
Where C represent the sign matrix and W repraent the connectivity matrix. In every model, we always assume the model has no-self connection.\
The user can define the non-linearity, to use : $tanh(), tanh(g(x-s)),\frac{1}{2}(1+tanh(g(x-s)))$\
We declaring a model, we alway need to give explicitely the different matrix, please use the utils.py method to retrive ROI specific matrix or generate random one.
```bash
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
"""

```
B) The connectomic data: \
All the connectomics data come from this [work](https://elifesciences.org/articles/66039) by the [Janelia Research Campus](https://www.janelia.org/). The preprocessing of the data was performed using the `bouchardlab/maxent_diffusion/batch_ergm_manager.py`. This code was used to extract information about each desired region of interest(ROI). In the current version, we only published 5 different ROI. You can find the complete name of each roi [here](https://neuprint.janelia.org/results). \
For each ROI, we require the following files
- `adjacency_scc.npy`  # Connectivity matrix
- `cell_body_fiber_assignement.pkl`  # Correspondence between an ID and a specific cell type
- `nid_scc.npy`  # Indicates which cell type is assigned to each neuron
- `ei_neuron_types.pkl`  # Correspondence between an ID and a specific neurotransmitter
- `ei_neuron.npy`  # Indicates which neurotransmitter is assigned to each neuron

You can also find the cellular type and the neurotransmitter correspondant under `flybrain/connectomes.py`

## Worklows
### 1) Convergence_lyapunov
```bash
% lyapu_convergence --help
Usage: lyapu_convergence [OPTIONS]

  Simulates the convergence of Lyapunov exponents in a recurrent neural
  network (RNN) model.

  Parameters:   tons (float): Initial transient time before recording starts.
                n_samples (int): Number of samples to test.
                g (float):Coupling strength for connectivity matrix.
                n (int): Number of neurons in the network.
                nle (int): Number of Lyapunov exponents to compute.
                dt(float): Time step size for integration.
                tsim (int): Total simulation time.
                activation (str): Type of activation function ('std', 'pos','strech').

Options:
  --tons FLOAT                    tons used in this experiment  [required]
  --activation [tanh|tanh_pos|tanh_streched]
                                  Which loss we want to use for theoptimisation  [required]
  --g FLOAT                       Syn dist
  --n_samples INTEGER             Amount of sample
  --n INTEGER                     Size of the model, number of neurons used
  --nle INTEGER                   number of lyapunoc computed
  --help                          Show this message and exit.
```
### D.2 transition_2_Chaos
```bash
% transition_2_chaos --help
Usage: transition_2_chaos [OPTIONS]
  Generates multiple RNN models with varying coupling values and calculates Lyapunov spectrum.

Options:
  --gmin FLOAT                   Minimum value for coupling strength g [required]
  --gmax FLOAT                   Maximum value for coupling strength g [required]
  --m_g INTEGER                  Number of g values between gmin and gmax (default: 5)
  --n_samples INTEGER            Number of samples for each g value [required]
  --activation [std|pos|strech]  Activation function used in the model (default: std)
  --save BOOLEAN                 Options to save the output
  --help                         Show this message and exit.
```
### D.3 train_rd_model
```bash
% train_rd_model --help
Usage: train_rd_model [OPTIONS]

Options:
  --n INTEGER                     Size of the model, number of neurons used [required]
  --n_samples, --n_samples INTEGER
                                  Number of sample used, (default:1) [required]
  --nLE INTEGER                   Number of Lyapunov exponent used  [required]
  --loss [l2|MSE]                 Which loss we want to use for the optimisation  [required]
  --activation [tanh|tanh_pos|tanh_streched]
                                  Which loss we want to use for the optimisation  [required]
  --target FLOAT                  Target lyapunov vector
  --tOns FLOAT                    Step size between two consecutive QR facto
  --tSim INTEGER                  Length of the simulation [tau]
  --g FLOAT                       Synaptic distribution parameter  [required]
  --n_epochs INTEGER              Number of epochs used
  --lr FLOAT                      Learning rate used
  --train_weights BOOLEAN         Optimizition on the weights
  --train_shifts BOOLEAN          Optimizition on the shitfs
  --train_gains BOOLEAN           Optimizition on the gains
  --help                          Show this message and exit.
```
### D.4 train_flybrain_full
```bash
% train_flybrain_full --help
Usage: train_flybrain_full [OPTIONS]

Options:
  --n_samples INTEGER             Number of sample used, (default:1)
                                  [required]
  --nLE INTEGER                   Number of Lyapunov exponent used  [required]
  --loss [l2|MSE]                 Which loss we want to use for the
                                  optimisation  [required]
  --ROI TEXT                      Which ROI, we would like to use  [required]
  --activation [tanh_pos|tanh_streched]
                                  Which loss we want to use for the
                                  optimisation  [required]
  --target FLOAT                  Target lyapunov vector
  --tOns FLOAT                    Step size between two consecutive QR facto
  --tSim INTEGER                  Length of the simulation [tau]
  --n_epochs INTEGER              Number of epochs used
  --lr FLOAT                      Learning rate used
  --help                          Show this message and exit.
```
### D.4 train_flybrain_pop
```bash
% train_flybrain_pop --help
Usage: train_flybrain_pop [OPTIONS]
Options:
  --n_samples INTEGER             Number of sample used
  --nle INTEGER                   Number of Lyapunov exponent used
  --subpopulation [cell_fibers|neurotransmitter]
                                  Which features would you like to use to
                                  define the subpopulation
  --ROI TEXT                      Which ROI, we would like to use  [required]
  --activation [tanh_pos|tanh_streched]
                                  Which loss we want to use for the
                                  optimisation
  --loss [l2|MSE]                 Which loss we want to use for the
                                  optimisation
  --target FLOAT                  Target lyapunov vector
  --tOns FLOAT                    Step size between two consecutive QR facto
  --tSim INTEGER                  Length of the simulation [tau]
  --n_epochs INTEGER              Number of epochs used
  --lr FLOAT                      Learning rate used
  --help                          Show this message and exit.
```
## Installation

- Clone the repo

```bash
git clone https://github.com/edouardkoehn/WM_Atlas.git
```
- Create your virtual env
```bash
conda create -n fly python=3.12
conda activate fly
```
- Install poetry
```bash
pip install poetry
```
- install the module and set up the precommit
```bash
poetry install
poetry run pre-commit install
poetry env info
```
