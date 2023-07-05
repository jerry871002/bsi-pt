#!/bin/bash

# Run experiments over eight types of phi opponent

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

cd src/

mkdir -p logs/$scenario
mkdir -p data/

for i in {1..8}
do
   python run_experiment.py $scenario -r $num_runs -n $num_episodes -po -ph $i $mode_flag > logs/$scenario/phi_$i.log &
done