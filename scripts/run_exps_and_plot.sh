#!/bin/bash

# Run all the experiments in our paper
# then save the results and plot them

# exit the script if any statement returns a non-true return value
set -e

program_name=$0

function usage {
    echo "Usage: $program_name [SCENARIO] [NUM_RUNS] [NUM_EPISODE]"
    echo "  - The default scenario is baseball"
    echo "  - The default value of NUM_RUNS is 1000"
    echo "  - The default value of NUM_EPISODE is 30"
    exit 1
}

if [[ $1 =~ help|--help|-h ]]
then
    usage
fi

scenario=${1:-baseball}
num_runs=${2:-1000}
num_episodes=${3:-30}

top_dir=$(git rev-parse --show-toplevel)
cd "$top_dir/src/"

mkdir -p logs/

declare -a exp_nums=("1" "2" "3")

for exp_num in "${exp_nums[@]}"
do
    printf "Running experiment $exp_num on $scenario\n"
    python run_exp_and_plot.py -e $exp_num -s $scenario -r $num_runs -n $num_episodes > logs/exp${exp_num}_${scenario}.log

    secs=$SECONDS
    mins=$(( secs/60 ))
    secs=$(( secs-mins*60 ))
    printf "Experiment $exp_num on $scenario done, time elapsed: %02dm%02ds\n\n" $mins $secs
done
