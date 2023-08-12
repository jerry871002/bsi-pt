# Run Experiment Individually

There are three entry points in this project.

- `run.py`: Run a game in a given environment with an agent and an opponent type
- `run_experiment.py`: Run **multiple runs** of a game in a given scenario with an agent and an opponent type
- `run_exp_and_plot.py`: Run a experiment in the paper and plot the results

From a high level, the `run_exps_and_plot.sh` script calls `run_exp_and_plot.py` multiple times to produce results of multiple experiments. And each run of `run_exp_and_plot.py` includes a call to `run_experiment.py` which runs a multiple-runs experiment under a given scenario and agent type.

On the other hand, `run.py` provides a finer granularity of only running one run at a time, it's best to start from `run.py` to get better understanding of how the code works.

Make sure the working directory is `src/` before running any of the scripts.

## `run.py`

!!! warning
    This refers to `src/run.py`, do not confuse with the `run.py` under each environment.

The usage is as follow, no files will be generated after the run, all outputs will print out on screen.

```bash
# run bsi in navigation game against phi 5 opponent
python run.py nav bsi -n 10 -po -ph 5

# run bsi-pt in grid world against phi 6 opponent
python run.py grid bsi-pt -n 10 -po -ph 6
```

To see the full usage, run `python run.py -h`.

## `run_experiment.py`

There are various ways to run this script, for example

```bash
# run all agents in navigation game against phi 2 opponent
# with 8 runs and 10 episodes each
python run_experiment.py nav -r 8 -n 10 -po -ph 2

# run BPR+ and Deep BPR+ in grid world against phi 1 opponent
# with 5 runs and 20 episodes each
python run_experiment.py grid -r 5 -n 20 -po -ph 1 -a bpr+ deep-bpr+
```

After the run, the results will be generated and stored as pickle files in `data/`

To see the full usage, run `python run_experiment.py -h`. Also, check [`scripts/run_experiments.sh`](https://github.com/jerry871002/bayesian-strategy-inference/blob/master/scripts/run_experiments.sh) and [`scripts/run_phi_experiments.sh`](https://github.com/jerry871002/bayesian-strategy-inference/blob/master/scripts/run_phi_experiments.sh) for more examples.

## `run_exp_and_plot.py`

!!! note
    Check [this page](experiment_definition.md) to see the definition of each experiment.

This script is for running the experiments in our paper, you have to specify which experiment and which scenario you want to run, for example

```bash
# run experiment 1 in grid world
python run_exp_and_plot.py 1 grid -r 5 -n 10

# run experiment 2 in navigation game
python run_exp_and_plot.py 2 nav -r 5 -n 10

# run experiment 3 in soccer game
python run_exp_and_plot.py 3 soccer -r 5 -n 10
```

After the run, pickle files, csv files, and figures will be generated and will store in `data/`, `csv/`, `fig/`, respectively

To see the full usage, run `python run_exp_and_plot.py -h`.
