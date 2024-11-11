#!/bin/bash

# Define parameters
n_samples=1
loss="l2"
target=0.0
tons=0.2
tsim=300
n_epochs=300
lr=0.01
activation="tanh_pos"
pop='neurotransmitter' #'cell_fibers'
roi='EB'  # Define the value for the synaptic distribution parameter

# Loop over different nle values
for nle in 1 10 50 100
do
    command="train_flybrain_pop --n_samples $n_samples  --nle $nle --subpopulation neurotransmitter --roi $roi   --loss $loss --activation $activation --target $target --tons $tons --tsim $tsim --n_epochs $n_epochs --lr $lr"
    echo $command
    $command
done
