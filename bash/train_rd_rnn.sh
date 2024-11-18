#!/bin/bash
#Usage: train_rd_model [OPTIONS]

# Options:
#   --n INTEGER                     Size of the model, number of neurons used
#                                   [required]
#   --n_samples INTEGER             Number of sample used, (default:1)
#                                   [required]
#   --nle INTEGER                   Number of Lyapunov exponent used  [required]
#   --g FLOAT                       Synaptic distribution parameter  [required]
#   --n_epochs INTEGER              Number of epochs used
#   --activation [tanh|tanh_pos|tanh_streched]
#                                   Which loss we want to use for the
#                                   optimisation  [required]
#   --loss [l2|MSE]                 Which loss we want to use for the
#                                   optimisation  [required]
#   --target FLOAT                  Target lyapunov vector
#   --tons FLOAT                    Step size between two consecutive QR facto
#   --tsim INTEGER                  Length of the simulation [tau]
#   --train_weights BOOLEAN         Optimizition on the weights
#   --train_shifts BOOLEAN          Optimizition on the shitfs
#   --train_gains BOOLEAN           Optimizition on the gains
#   --lr FLOAT                      Learning rate used
#   --early_stopping FLOAT          Value of the loss at wich the optimization
#                                   would stop
#   --help                          Show this message and exit.
# Define parameters
n=100
n_samples=5

g=0.8  # Define the value for the synaptic distribution parameter
n_epochs=1000

activation="tanh_pos"
loss="MSE"
target=0.

tons=0.2
tsim=200

train_weights="False"
train_gains="True"
train_shifts="True"

lr=0.01
early_crit=1e-3

# Loop over different nle values
for nle in 75 100
do
    command="train_rd_model --n $n --n_samples $n_samples --nle $nle --g $g --n_epochs $n_epochs  --activation $activation --loss $loss --target $target --tons $tons --tsim $tsim   --train_weights $train_weights --train_shifts $train_shifts --train_gains $train_gains  --lr $lr --early_stopping $early_crit"
    echo $command
    $command
done
