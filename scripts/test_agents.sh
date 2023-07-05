#!/bin/bash

# Check if there's any error that crashes the experiments

# exit the script if any statement returns a non-true return value
set -e

declare -a scenarios=("grid" "nav" "soccer")

cd src/

for scenario in "${scenarios[@]}"
do
    printf "\n+++++ Testing in $scenario +++++\n"

    printf "\n----- Testing BPR+ agent -----\n"
    python run.py $scenario bpr+ -n 5 -e 2

    printf "\n----- Testing Deep BPR+ agent -----\n"
    python run.py $scenario deep-bpr+ -n 5 -e 2

    printf "\n----- Testing ToM agent -----\n"
    python run.py $scenario tom -n 5 -e 2

    printf "\n----- Testing BPR-OKR agent -----\n"
    python run.py $scenario bpr-okr -n 5 -e 2

    printf "\n----- Testing BSI agent -----\n"
    python run.py $scenario bsi -n 5 -e 2

    printf "\n----- Testing BSI-PT agent -----\n"
    python run.py $scenario bsi-pt -n 5 -e 2
done