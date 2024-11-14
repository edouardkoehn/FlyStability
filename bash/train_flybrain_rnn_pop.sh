#!/bin/bash

# Define parameters
n_samples=1
loss="MSE"
target=0.0
tons=0.2
tsim=300
n_epochs=1000
lr=0.01
activation="tanh_pos"
pop= 'neurotransmitter' #'neurotransmitter' #'cell_fibers'
roi='EB'  # Define the value for the synaptic distribution parameter

# Loop over different nle values
for nle in 10 50
do
    command="train_flybrain_pop --n_samples $n_samples  --nle $nle --subpopulation neurotransmitter --roi $roi   --loss $loss --activation $activation --target $target --tons $tons --tsim $tsim --n_epochs $n_epochs --lr $lr"
    echo $command
    $command
done
