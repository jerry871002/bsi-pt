# Run the Experiments Individually

## Getting Started

There are three entry points of this project

- `run.py`: Run the experiment in a given scenario with an agent type
- `run_experiment.py`: Run **multiple runs** of a experiment in a given scenario with an agent type
- `run_exp_and_plot.py`: Run a experiment in the paper and plot the results

At a high level, the `run_exps_and_plot.sh` script calls `run_exp_and_plot.py` multiple times to produce results of multiple experiments. And each run of `run_exp_and_plot.py` includes a call to `run_experiment.py` which runs a multiple-runs experiment under a given scenario and agent type.

On the other hand, `run.py` provides a finer granularity of only running one run at a time, it's best to start from `run.py` to get better understanding of how the code works.

Make sure the working directory is `src/` before running any of the scripts.

### `run.py`

!!! warning
    This refers to `src/run.py`, do not confuse with the `run.py` under each environment.

The usage is as follow, no files will be generated after the run, all outputs will print out on screen.

```bash
# run bsi in navigation game against phi 5 opponent
python run.py nav bsi -n 10 -po -ph 5

# run bsi-pt in grid world against phi 6 opponent
python run.py grid bsi-pt -n 10 -po -ph 6
```

To see the full usage, run `python run.py -h`

```
usage: run.py [-h] [-o {1,2,3,4,5}] [-a {1,2,3,4,5}] [-ph {1,2,3,4,5,6,7,8,9,10,11}] [-e EPISODE_RESET] [-p] [-pa] [-n NUM_EPISODES] [-q Q_DISTANCE] [-pat P_PATTERN] [-b | -po | -np | -nnp]
              {grid,nav,soccer} {bpr+,deep-bpr+,tom,bpr-okr,bsi,bsi-pt}

Run the experiment in a given scenario with an agent type

positional arguments:
  {grid,nav,soccer}     the scenario you would like to run on
  {bpr+,deep-bpr+,tom,bpr-okr,bsi,bsi-pt}
                        type of agent you would like to use

optional arguments:
  -h, --help            show this help message and exit
  -o {1,2,3,4,5}, --op-policy {1,2,3,4,5}
                        enter 1~5 to choose the initial policy (tau) used by the opponent, randomly choose one if not set
  -a {1,2,3,4,5}, --agent-policy {1,2,3,4,5}
                        enter 1~5 to choose the initial policy (pi) used by the agent, randomly choose one if not set
  -ph {1,2,3,4,5,6,7,8,9,10,11}, --phi {1,2,3,4,5,6,7,8,9,10,11}
                        enter 1~11 to choose the phi used by the opponent, this value will only be used if `--phi-opponent` is also set, default value is 1
  -e EPISODE_RESET, --episode-reset EPISODE_RESET
                        number of episodes before the opponent switches its policy, default value is 10
  -p, --print-map       print the map of the game while episodes are played
  -pa, --print-action   print the action of agent and opponent in each step
  -n NUM_EPISODES, --num-episodes NUM_EPISODES
                        enter the number of episodes you want to run, default value is 1
  -q Q_DISTANCE, --q-distance Q_DISTANCE
                        distance between existing phi and new phi, this value will only be used if `--new-phi-opponent` is also set, default value is 0
  -pat P_PATTERN, --p-pattern P_PATTERN
                        probability of using policy in existing phi, this value will only be used if `--new-phi-noise-opponent` is also set, default value is 1
  -b, --bpr-opponent    whether the environment uses a BPR opponent
  -po, --phi-opponent   whether the environment uses a Phi opponent, you should also use `--phi` to specify the type of phi to use
  -np, --new-phi-opponent
                        whether the environment uses a new Phi opponent, you should use `--q-distance` to specify the probability of using policy in existing phi
  -nnp, --new-phi-noise-opponent
                        whether the environment uses a new Phi noise opponent, you should use `--p-pattern` to control the probability of randomly choosing policy
```

### `run_experiment.py`

There are various ways to run this script, for example

- Run all agents in navigation game against phi 2 opponent with 8 runs and 10 episodes each
    ```
    python run_experiment.py nav -r 8 -n 10 -po -ph 2
    ```
- Only run BPR+ and Deep BPR+ in grid world against phi 1 opponent with 5 runs and 20 episodes each
    ```
    python run_experiment.py grid -r 5 -n 20 -po -ph 1 -a bpr+ deep-bpr+
    ```

After the run, the results will be generated and stored as pickle files in `data/`

To see the full usage, run `python run_experiment.py -h`

```
usage: run_experiment.py [-h] [-r NUM_RUNS] [-n NUM_EPISODES] [-a [{bpr+,deep-bpr+,tom,bpr-okr,bsi,bsi-pt} ...]] [-ph {1,2,3,4,5,6,7,8,9,10,11}] [-e EPISODE_RESET] [-p] [-pa] [-m]
                         [-d DATA_DIR] [-q Q_DISTANCE] [-pat P_PATTERN] [-b | -po | -np | -nnp]
                         {grid,nav,soccer}

Run multiple runs of the experiment in a given scenario with an agent type

positional arguments:
  {grid,nav,soccer}     the scenario you would like to run on

optional arguments:
  -h, --help            show this help message and exit
  -r NUM_RUNS, --num-runs NUM_RUNS
                        number of total runs, default value is 10
  -n NUM_EPISODES, --num-episodes NUM_EPISODES
                        number of episodes in each run, default value is 1
  -a [{bpr+,deep-bpr+,tom,bpr-okr,bsi,bsi-pt} ...], --agents [{bpr+,deep-bpr+,tom,bpr-okr,bsi,bsi-pt} ...]
                        the agents you want to run in the experiment, if not set, all agents will be run
  -ph {1,2,3,4,5,6,7,8,9,10,11}, --phi {1,2,3,4,5,6,7,8,9,10,11}
                        enter 1~11 to choose the phi used by the opponent, this value will only be used if `--phi-opponent` is also set, default value is 1
  -e EPISODE_RESET, --episode-reset EPISODE_RESET
                        number of episodes before the opponent switches its policy, default value is 10
  -p, --print-map       print the map of the game while episodes are played
  -pa, --print-action   print the action of agent and opponent in each step
  -m, --multi-processing
                        whether to compute using multi-processing
  -d DATA_DIR, --data-dir DATA_DIR
                        where to store the result pickle files
  -q Q_DISTANCE, --q-distance Q_DISTANCE
                        distance between existing phi and new phi, this value will only be used if `--new-phi-opponent` is also set, default value is 0
  -pat P_PATTERN, --p-pattern P_PATTERN
                        probability of using policy in existing phi, this value will only be used if `--new-phi-noise-opponent` is also set, default value is 1
  -b, --bpr-opponent    whether the environment uses a BPR opponent
  -po, --phi-opponent   whether the environment uses a Phi opponent, you should also use `--phi` to specify the type of phi to use
  -np, --new-phi-opponent
                        whether the environment uses a new Phi opponent, you should use `--q-distance` to specify the probability of using policy in existing phi
  -nnp, --new-phi-noise-opponent
                        whether the environment uses a new Phi noise opponent, you should use `--p-pattern` to control the probability of randomly choosing policy
```

### `run_exp_and_plot.py`

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

To see the full usage, run `python run_exp_and_plot.py -h`

```
usage: run_exp_and_plot.py [-h] [-e [{1,2,3,4,5,6} ...]] [-s [{grid,nav,soccer} ...]] [-r NUM_RUNS] [-n NUM_EPISODES] [-p] [-pa]

Run the experiments in our paper, save the results (pkl and csv) and plot them

optional arguments:
  -h, --help            show this help message and exit
  -e [{1,2,3,4,5,6} ...], --exp-nums [{1,2,3,4,5,6} ...]
                        the experiment(s) you would like to run
  -s [{grid,nav,soccer} ...], --scenarios [{grid,nav,soccer} ...]
                        the scenario(s) you would like to run on
  -r NUM_RUNS, --num-runs NUM_RUNS
                        number of total runs, default value is 10
  -n NUM_EPISODES, --num-episodes NUM_EPISODES
                        number of episodes in each run, default value is 1
  -p, --print-map       print the map of the game while episodes are played
  -pa, --print-action   print the action of agent and opponent in each step
```
