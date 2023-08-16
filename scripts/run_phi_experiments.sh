#!/bin/bash

# Disclaimer:
# This script is an example of how to use run_experiment.py
# and does not generate any results in our paper

# Run experiments over various types of phi opponent

# exit the script if any statement returns a non-true return value
set -e

program_name=$0

function usage {
    echo "Usage: $program_name [NUM_RUNS] [NUM_EPISODE] [MODE]"
    echo "  - The default value of NUM_RUNS is 1000"
    echo "  - The default value of NUM_EPISODE is 50"
    echo "  - The default value of MODE is \"multi\""
    exit 1
}

if [[ $1 =~ help|--help|-h ]]
then
    usage
fi

scenario=$1
num_runs=${2:-1000}
num_episodes=${3:-50}
mode=${4:-"multi"}  # "multi" for multi-process and "normal" for single process

if [ $mode = "multi" ]
then
    mode_flag="-m"
else
    mode_flag=""
fi

top_dir=$(git rev-parse --show-toplevel)
cd "$top_dir/src/"

mkdir -p logs/$scenario
mkdir -p data/

for i in {1..11}
do
   python run_experiment.py $scenario -r $num_runs -n $num_episodes -po -ph $i $mode_flag > logs/$scenario/phi_$i.log &
done
