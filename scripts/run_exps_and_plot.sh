#!/bin/bash

# Run all the experiments in our paper
# save the results and plot them

# exit the script if any statement returns a non-true return value
set -e

program_name=$0

function usage {
    echo "Usage: $program_name [NUM_RUNS] [NUM_EPISODE]"
    echo "  - The default value of NUM_RUNS is 1000"
    echo "  - The default value of NUM_EPISODE is 50"
    exit 1
}

if [[ $1 =~ help|--help|-h ]]
then
    usage
fi

num_runs=${1:-1000}
num_episodes=${2:-50}

cd src/

mkdir -p logs/

declare -a exp_nums=("1" "2" "3")
declare -a scenarios=("grid" "nav" "soccer")

i=1

for exp_num in "${exp_nums[@]}"
do
    for scenario in "${scenarios[@]}"
    do
        printf "Running experiment $exp_num on $scenario\n"
        python run_exp_and_plot.py -e $exp_num -s $scenario -r $num_runs -n $num_episodes > logs/exp${exp_num}_${scenario}.log

        secs=$SECONDS
        mins=$(( secs/60 ))
        secs=$(( secs-mins*60 ))
        printf "Experiment $exp_num on $scenario done, time elapsed: %02dm%02ds, %d%% of jobs done\n\n" $mins $secs $(( i*100/${#exp_nums[@]}/${#scenarios[@]} ))

        i=$(( i+1 ))
    done
done