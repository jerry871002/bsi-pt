#!/bin/bash

# exit the script if any statement returns a non-true return value
set -e

declare -a scenarios=("grid" "nav" "soccer")

cd src/

for scenario in "${scenarios[@]}"
do
    printf "\n----- Testing random switch opponent -----\n"
    python run.py $scenario bpr+ -n 5 -e 2

    printf "\n----- Testing BPR opponent -----\n"
    python run.py $scenario bpr+ -n 5 -b

    printf "\n----- Testing Phi opponent -----\n"
    for i in {1..11}
    do
        printf "\n+++++ Phi $i +++++\n"
        python run.py $scenario bpr+ -n 5 -po -ph $i
    done

    printf "\n----- Testing New Phi opponent -----\n"
    for q in {0..3}
    do
        printf "\n+++++ q = $q +++++\n"
        python run.py $scenario bpr+ -n 5 -np -q $q
    done

    printf "\n----- Testing New Phi Noise opponent -----\n"
    for p in $(seq 2 2 8)  # 2 4 6 8
    do
        printf "\n+++++ p = 0.$p +++++\n"
        python run.py $scenario bpr+ -n 5 -nnp -pat 0.$p
    done
done