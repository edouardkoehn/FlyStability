# 🪰  Neuronal Response As A Learning Substrate For Artificial Neuronal
Network 🪰
This repository contains all the code related to the NRM project. This project was part of my master thesis conducted in the [Neural Systems and Machine Learning Lab](https://bouchardlab.lbl.gov/) under the supervision of K. Bouchard.
[Edouard Koehn](edouard.koehn@berkeley.edu) \
20.01.2024

## Index
- [Abstract](#Abstract)

- [Structure of the repo](#Structure-of-the-code)

- [Data usage](#Model-and-Data)

- [Workflows](#Workflows)

- [Installation](#Installation)

## Abstract
Neural systems are characterized by complex dynamics emerging from billions of interconnected neurons, exhibiting temporal variability and chaotic behavior. While Recurrent Neural Networks (RNNs) have been extensively employed to model such dynamics, they remain limited in their biological plausibility, learning stability, and integration with empirical data. This work introduces a novel framework that extends canonical RNN learning paradigms by incorporating features of biological neural circuits. Using a random RNN architecture, we investigate neuronal response modulation (NRM) as a learning substrate. Our findings demonstrate that NRM, through its influence on the system’s Lyapunov spectrum, enables effective control of network dynamics despite having relatively low parameterization compared to direct synaptic plasticity. Furthermore, we successfully adapt this framework to biologically constrained systems informed by real connectomics data. This approach represents a step toward the development of more brain-like systems, offering biologically grounded insights into neural dynamics and advancing the creation of computational models inspired by the brain.

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
### A) The model:
We used the standard dynamic defined by : (1) $\frac{\partial{h_i}}{\partial{t}}=-h_i + \sum_{j=0}^NC_{ij}W_{ij}*(\phi(h_j))$\
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
### B) The connectomic data:
All the connectomics data come from this [work](https://elifesciences.org/articles/66039) by the [Janelia Research Campus](https://www.janelia.org/). The preprocessing of the data was performed using the `bouchardlab/maxent_diffusion/batch_ergm_manager.py`. This code was used to extract information about each desired region of interest(ROI). In the current version, we only published 5 different ROI, that you can find [HERE](https://drive.google.com/drive/folders/11uwJG8WXrDFQjgepxHp347E_X8uaJl-F?usp=sharing). You can find the complete name of each roi [here](https://neuprint.janelia.org/results).\
For each ROI, we require the following files
- `adjacency_scc.npy`  # Connectivity matrix
- `cell_body_fiber_assignement.pkl`  # Correspondence between an ID and a specific cell type
- `nid_scc.npy`  # Indicates which cell type is assigned to each neuron
- `ei_neuron_types.pkl`  # Correspondence between an ID and a specific neurotransmitter
- `ei_neuron.npy`  # Indicates which neurotransmitter is assigned to each neuron

You can also find the cellular type and the neurotransmitter correspondant under `flybrain/connectomes.py`

## Workflows
### 1) Convergence_lyapunov
This experiment allow the user to test and visualize the convergence of the Lyapnuov exponent. You can try different parameters to see how the Tons. Notice the fact that tons has a direct on the time of compute every lyapunov spectrum. (see report Fig.X)
```bash
% lyapu_convergence --help
  Compute the convergence of Lyapunov exponents in a recurrent neural network (RNN) model.

Options:
  --tons FLOAT                    Tons used in this experiment  [required]
  --activation [tanh|tanh_pos|tanh_streched]
                                  Which loss we want to use for the
                                  optimisation  [required]
  --g FLOAT                       Syn dist
  --n_samples INTEGER             Amount of sample used
  --n INTEGER                     Size of the model, number of neurons used
  --nle INTEGER                   number of lyapunoc computed
  --help                          Show this message and exit.
```
### 2) Transition 2 Chaos
This experiment allows to test the phase transition of the any random netowrk. You can try different activation function and can also choose wich parameter need to be drawn from a normal disttribution. (see report Fig.X)
```bash
% transition_2_chaos --help
 Find the transition to chaos in a recurrent neural network (RNN) model.

Options:
  --gmin FLOAT                    Minimum value for coupling strength g
                                  [required]
  --gmax FLOAT                    Maximum value for coupling strength g
                                  [required]
  --m_g INTEGER                   Number of g values between gmin and gmax
                                  (default: 5)
  --n_samples INTEGER             Number of samples for each g value
                                  [required]
  --parameter [weights|shifts|gains]
                                  Which parameter to vary
  --activation [std|pos|strech]   Activation function used in the model
                                  (default: std)
  --save BOOLEAN                  Options to save the output
  --help                          Show this message and exit.
```
### 3) Spectrum
Similar workflow than the "Trans 2 Chaos" but instee of only looking at the maximum lypunov exponent, we look at the full spectrum, the strange attractor dimensionality and the Sinai-Kolmogorov attractor.
```bash
% spectrum --help
  Characterize the spectrum of a recurrent neural network (RNN) model.

Options:
  --n INTEGER                     Size of the network   [required]
  --n_samples INTEGER             Number of samples for each g value
                                  [required]
  --variance_max INTEGER          Number of samples for each g value
  --parameter [weights|shifts|gains]
                                  Which parameter to vary   [required]
  --activation [std|pos|strech]   Activation function used in the model
                                  (default: std)
  --save BOOLEAN                  Options to save the output
  --help                          Show this message and exit.
```
### 4) Train Random RNN
This pipeline allow you to train a random rnn. Once you choose one training options (train_weight, ...) the optimisation would be on those parameters.
```bash
% train_rd_model --help
  Train an RNN model with specified parameters on the flybrain connectome.

Options:
  --n INTEGER                     Size of the model, number of neurons used
                                  [required]
  --n_samples INTEGER             Number of sample used, (default:1)
                                  [required]
  --nle INTEGER                   Number of Lyapunov exponent used  [required]
  --g FLOAT                       Synaptic distribution parameter  [required]
  --n_epochs INTEGER              Number of epochs used
  --activation [tanh|tanh_pos|tanh_streched]
                                  Which loss we want to use for the
                                  optimisation  [required]
  --loss [l2|MSE|Sinai|MSE_Custom]
                                  Which loss we want to use for the
                                  optimisation  [required]
  --target FLOAT                  Target lyapunov vector
  --tons FLOAT                    Step size between two consecutive QR facto
  --tsim INTEGER                  Length of the simulation [tau]
  --train_weights BOOLEAN         Optimizition on the weights
  --train_shifts BOOLEAN          Optimizition on the shitfs
  --train_gains BOOLEAN           Optimizition on the gains
  --lr FLOAT                      Learning rate used
  --help                          Show this message and exit.

```
### 5) Train Random RNN
Same pipeline than the previous one but instead of train the full range of parameter, you can specify how many parameters you want to optimize.
```bash
% train_rd_model_fixed_param --help
  Train an RNN model with fixed parameters on the flybrain connectome.

Options:
  --n INTEGER                     Number of neurons in the model  [required]
  --n_samples INTEGER             Number of samples used  [required]
  --nle INTEGER                   Number of Lyapunov exponents  [required]
  --g FLOAT                       Synaptic distribution parameter  [required]
  --n_epochs INTEGER              Number of epochs
  --activation [tanh|tanh_pos|tanh_streched]
                                  Activation function  [required]
  --loss [l2|MSE|Sinai]           Loss function  [required]
  --target FLOAT                  Target Lyapunov vector
  --number_param INTEGER...       Number of parameters to optimize
  --tons FLOAT                    Step size between QR factorization
  --tsim INTEGER                  Length of simulation [tau]
  --train_weights BOOLEAN         Optimize weights
  --train_shifts BOOLEAN          Optimize shifts
  --train_gains BOOLEAN           Optimize gains
  --lr FLOAT                      Learning rate
  --help                          Show this message and exit.

```
### 6) Train Flybrain
This experiment allow you to train a biologicaly constrained RNN. Indeed you can specify wich ROI should be used as connectivity for you model. In this pipeline, only the shifts and gains are optimised. Watch out, each neurons has it's own shift and gains parameter.
```bash
% train_flybrain_full --help

Options:
  --n_samples INTEGER             Number of sample used, (default:1)
                                  [required]
  --nle INTEGER                   Number of Lyapunov exponent used  [required]
  --loss [l2|MSE|MSE_Custom]      Which loss we want to use for the
                                  optimisation  [required]
  --n_epochs INTEGER              Number of epochs used
  --roi TEXT                      Which ROI, we would like to use  [required]
  --activation [tanh_pos|tanh_streched]
                                  Which loss we want to use for the
                                  optimisation  [required]
  --loss [l2|MSE|MSE_Custom]      Which loss we want to use for the
                                  optimisation  [required]
  --target FLOAT                  Target lyapunov vector
  --tons FLOAT                    Step size between two consecutive QR facto
  --tsim INTEGER                  Length of the simulation [tau]
  --lr FLOAT                      Learning rate used
  --help                          Show this message and exit.
```
### 7) Train Flybrain population specific
Similar pipeline to the previous one but this time the shift and the gains parameter are shared across neuron populations. We can define the population based on the neurotransmitter type or the cell type.
```bash
% train_flybrain_pop --help
  Pipeline to train an RNN model constrained on the flybrain connectome.

Options:
  --n_samples INTEGER             Number of runs to perform
  --nle INTEGER                   Number of Lyapunov exponents used for the
                                  optimization
  --n_epochs INTEGER              Number of epochs used
  --roi TEXT                      Region of interest (Flybrain)  [required]
  --subpopulation [cell_fibers|neurotransmitter]
                                  Subpopulation feature (cell types based or
                                  neurotransmitter based)
  --activation [tanh_pos|tanh_streched]
                                  Activation function used
  --loss [l2|MSE]                 Loss function
  --target FLOAT                  Target value for the Lyapunov vector
  --tons FLOAT                    Step size between QR factorizations
  --tsim INTEGER                  Simulation length [tau]
  --lr FLOAT                      Learning rate
  --help                          Show this message and exit.
```
## Installation
All the module management has been done using poetry. Here are the steps to set-up the code
- Clone the repo

```bash
git clone git@github.com:edouardkoehn/FlyStability.git
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
