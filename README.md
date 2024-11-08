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
### D.1 Convergence_lyapunoc
### D.2 transition_2_Chaos
### D.3 train_RD_RNN
### D.4 train_flybrain_rnn

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
