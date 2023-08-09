#!/bin/bash

# Run experiments over three types of opponent
#   - Random Switch Slow (switches every 10 episodes)
#   - Random Switch Fast (switches every 2 episodes)
#   - BPR Oppnent

program_name=$0

function usage {
    echo "Usage: $program_name [SCENARIO] [NUM_RUNS] [NUM_EPISODE] [MODE]"
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

if [ -z $scenario ]
then
    echo "Please choose a scenario"
fi

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

# 1. Random switch opponent which changes every 10 episode
python run_experiment.py $scenario -r $num_runs -n $num_episodes -e 10 $mode_flag > logs/$scenario/random_switch_10.log &

# 2. Random switch opponent which changes every 2 episode
python run_experiment.py $scenario -r $num_runs -n $num_episodes -e 2 $mode_flag > logs/$scenario/random_switch_2.log &

# 3. BPR opponent
python run_experiment.py $scenario -r $num_runs -n $num_episodes -b $mode_flag > logs/$scenario/bpr.log &
