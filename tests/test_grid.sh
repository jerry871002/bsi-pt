#!/bin/bash

# exit the script if any statement returns a non-true return value
set -ex

declare -a agents=("bpr+" "deep-bpr+" "tom" "bpr-okr" "bsi" "bsi-pt")

top_dir=$(git rev-parse --show-toplevel)
cd "$top_dir/src/"

for agent in "${agents[@]}"
do
    printf "\n----- Test $agent agent -----\n"

    printf "\n----- ($agent agent) Test random switch opponent -----\n"
    python run.py grid $agent -n 5 -e 2

    printf "\n----- ($agent agent) Test BPR opponent -----\n"
    python run.py grid $agent -n 5 -b

    printf "\n----- ($agent agent) Test Phi opponent -----\n"
    for i in {1..11}
    do
        printf "\n----- ($agent agent, Phi opponent) Phi $i -----\n"
        python run.py grid $agent -n 5 --phi-opponent --phi $i
    done

    printf "\n----- ($agent agent) Test New Phi opponent -----\n"
    for q in {0..3}
    do
        printf "\n----- ($agent agent, New Phi opponent) q = $q -----\n"
        python run.py grid $agent -n 5 --new-phi-opponent -q $q
    done

    printf "\n----- ($agent agent) Test New Phi Noise opponent -----\n"
    for p in $(seq 2 2 8)  # seq [first] [incr] [last] -> seq 2 2 8 = 2 4 6 8
    do
        printf "\n----- ($agent agent, New Phi Noise opponent) p = 0.$p -----\n"
        python run.py grid $agent -n 5 --new-phi-noise-opponent -pat 0.$p
    done
done
