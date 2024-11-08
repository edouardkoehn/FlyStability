# 🪰  Fly Stability  🪰
This repository contains all the code related to the Fly Stability project. This project was part of my master thesis conducted in the [Neural Systems and Machine Learning Lab](https://bouchardlab.lbl.gov/) under the supervision of K. Bouchard.

The goal of this project was to answer the following questions:
- How does the stability of an RNN influence the emegence of functional dynamic?

-Is stability a sufficient or necessary condition for functionnla dynamic to develop?
# Index
A. [Background](#installation)

B. [Structure of the repo](###Structureoftherepo)

C. [Data usage/storage](###Structureoftherepo)

D. [Workflows](###Workflows)

E. [Installation](#installation)


## A. Background and results

## B. Structure of the repo

## C. Data usage

## D. Worklows
### D.1 Convergence_lyapunov
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
## E. Installation

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
