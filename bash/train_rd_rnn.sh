#!/bin/bash

# Define parameters
n=100
n_samples=5
loss="l2"
target=-0.
tons=0.2
tsim=200
n_epochs=100
lr=0.01
train_weights="False"
train_gains="True"
train_shifts="True"
activation="tanh_streched"
g=1.5  # Define the value for the synaptic distribution parameter

# Loop over different nle values
for nle in 1 10 50
do
    command="train_rd_model --n $n --n_samples $n_samples --nle $nle --loss $loss --activation $activation --target $target --tons $tons --tsim $tsim --g $g --n_epochs $n_epochs --lr $lr --train_weights $train_weights --train_shifts $train_shifts --train_gains $train_gains"
    echo $command
    $command
done
