#!/bin/bash
# train_flybrain_full [OPTIONS]

# Options:
#   --n_samples INTEGER             Number of sample used, (default:1)
#                                   [required]
#   --nle INTEGER                   Number of Lyapunov exponent used  [required]
#   --loss [l2|MSE]                 Which loss we want to use for the
#                                   optimisation  [required]
#   --n_epochs INTEGER              Number of epochs used
#   --roi TEXT                      Which ROI, we would like to use  [required]
#   --activation [tanh_pos|tanh_streched]
#                                   Which loss we want to use for the
#                                   optimisation  [required]
#   --loss [l2|MSE]                 Which loss we want to use for the
#                                   optimisation  [required]
#   --target FLOAT                  Target lyapunov vector
#   --tons FLOAT                    Step size between two consecutive QR facto
#   --tsim INTEGER                  Length of the simulation [tau]
#   --lr FLOAT                      Learning rate used
#   --early_stopping FLOAT          Value of the loss at wich the optimization
#                                   would stop
#   --help                          Show this message and exit.

# Define parameters
n_samples=1
n_epochs=500
roi='AB'

activation="tanh_pos"
loss="MSE"
target=0.0

tons=0.2
tsim=200

lr=0.01
early_crit=1e-3

# Loop over different nle values
for nle in 1 10 25 50
do
    command="train_flybrain_full --roi $roi --n_samples $n_samples --nle $nle --loss $loss --activation $activation --target $target --tons $tons --tsim $tsim --n_epochs $n_epochs --lr $lr --early_stopping $early_crit"
    echo $command
    #$command
done
