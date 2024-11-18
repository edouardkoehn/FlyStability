#!/bin/bash
# Usage: train_flybrain_pop [OPTIONS]
# Options:
#   --n_samples INTEGER             Number of sample used
#   --nle INTEGER                   Number of Lyapunov exponent used
#   --n_epochs INTEGER              Number of epochs used
#   --roi TEXT                      Which ROI, we would like to use  [required]
#   --subpopulation [cell_fibers|neurotransmitter]
#                                   Which features would you like to use to
#                                   define the subpopulation
#   --activation [tanh_pos|tanh_streched]
#                                   Which loss we want to use for the
#                                   optimisation
#   --loss [l2|MSE]                 Which loss we want to use for the
#                                   optimisation
#   --target FLOAT                  Target lyapunov vector
#   --tons FLOAT                    Step size between two consecutive QR facto
#   --tsim INTEGER                  Length of the simulation [tau]
#   --lr FLOAT                      Learning rate used
#   --early_stopping FLOAT          Value of the loss at wich the optimization
#                                   would stop
#   --help                          Show this message and exit.
# Define parameters
n_samples=1
n_epochs=1000
roi='EB'
pop='neurotransmitter' #'cell_fibers'

activation="tanh_pos" #"tanh_streched"
loss="MSE"
target=0.0

tons=0.2
tsim=200

lr=0.01
early_crit=1e-3


# Loop over different nle values
for nle in 1 10 50
do
    command="train_flybrain_pop --n_samples $n_samples  --nle $nle --subpopulation $pop --roi $roi   --loss $loss --activation $activation --target $target --tons $tons --tsim $tsim --n_epochs $n_epochs --lr $lr --early_stopping $early_crit"
    echo $command
    $command
done
