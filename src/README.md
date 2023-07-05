# Source Code of the Experiments

## Getting Started

There are three entry points of this project
- `run.py`: Run the experiment in a given scenario with an agent type
- `run_experiment.py`: Run **multiple runs** of the experiment in a given scenario with an agent type
- `run_exp_and_plot.py`: Run the experiments in the paper

### `run.py`

The usage is as follow, no files will be generated after the run, all outputs will print out on screen

```bash
# run bsi in navigation game against phi 5 opponent
python run.py nav bsi -n 10 -po -ph 5 

# run bsi-pt in grid world against phi 6 opponent
python run.py grid bsi-pt -n 10 -po -ph 6
```

To see the full usage, run `python run.py -h`

```
usage: run.py [-h] [-o {1,2,3,4,5}] [-a {1,2,3,4,5}] [-ph {1,2,3,4,5,6,7,8}] [-e EPISODE_RESET] [-p] [-pa] [-n NUM_EPISODES] [-b | -po] {grid,nav,soccer} {bpr+,deep-bpr+,tom,bpr-okr,bsi,bsi-pt}

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
  -ph {1,2,3,4,5,6,7,8}, --phi {1,2,3,4,5,6,7,8}
                        enter 1~8 to choose the phi used by the opponent, this value will only be used if `--phi-opponent` is also set, default value is 1, notice that 9 and 10 is not a valid value
                        for grid world and soccer game
  -e EPISODE_RESET, --episode-reset EPISODE_RESET
                        number of episodes before the opponent switches its policy, default value is 10
  -p, --print-map       print the map of the game while episodes are played
  -pa, --print-action   print the action of agent and opponent in each step
  -n NUM_EPISODES, --num-episodes NUM_EPISODES
                        enter the number of episodes you want to run, default value is 1
  -b, --bpr-opponent    whether the environment uses a BPR opponent, if this option is set, `--episode-reset` will not be used
  -po, --phi-opponent   whether the environment uses a Phi opponent, if this option is set, `--episode-reset` will not be used
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
usage: run_experiment.py [-h] [-r NUM_RUNS] [-n NUM_EPISODES] [-a [{bpr+,deep-bpr+,tom,bpr-okr,bsi,bsi-pt} ...]] [-ph {1,2,3,4,5,6,7,8}] [-e EPISODE_RESET] [-p] [-pa] [-m] [-d DATA_DIR]
                         [-b | -po]
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
  -ph {1,2,3,4,5,6,7,8}, --phi {1,2,3,4,5,6,7,8}
                        enter 1~8 to choose the phi used by the opponent, this value will only be used if `--phi-opponent` is also set, default value is 1, notice that 9 and 10 is not a valid value
                        for grid world and soccer game
  -e EPISODE_RESET, --episode-reset EPISODE_RESET
                        number of episodes before the opponent switches its policy, default value is 10
  -p, --print-map       print the map of the game while episodes are played
  -pa, --print-action   print the action of agent and opponent in each step
  -m, --multi-processing
                        whether to compute using multi-processing
  -d DATA_DIR, --data-dir DATA_DIR
                        where to store the result pickle files
  -b, --bpr-opponent    whether the environment uses a BPR opponent, if this option is set, `--episode-reset` will not be used
  -po, --phi-opponent   whether the environment uses a Phi opponent, if this option is set, `--episode-reset` will not be used
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
usage: run_exp_and_plot.py [-h] [-r NUM_RUNS] [-n NUM_EPISODES] [-p] [-pa] {1,2,3,4} {grid,nav,soccer}

Run the experiments in our paper, save the results (pkl and csv) and plot them

positional arguments:
  {1,2,3,4}             the experiment you would like to run
  {grid,nav,soccer}     the scenario you would like to run on

optional arguments:
  -h, --help            show this help message and exit
  -r NUM_RUNS, --num-runs NUM_RUNS
                        number of total runs, default value is 10
  -n NUM_EPISODES, --num-episodes NUM_EPISODES
                        number of episodes in each run, default value is 1
  -p, --print-map       print the map of the game while episodes are played
  -pa, --print-action   print the action of agent and opponent in each step
```

## Environment

### Grid World

![Grid World](https://i.imgur.com/mMNnb58.png)

source: [Bayes-OKR Paper](../BPR&#32;Variants&#32;Papers/Bayes-OKR.pdf)

#### Create the environment

```python
from grid_world.env import *
env = GridWorld()
```

#### Reset the environment to initial state

```python
env.reset()
env.show()
```

The `env.show()` will print out the following result

```
1.2
...
...
△.○
```

#### Move the agent

There are five moves to choose from
- `Move.UP`
- `Move.RIGHT`
- `Move.DOWN`
- `Move.LEFT`
- `Move.STANDBY`

```python
env.step(Move.RIGHT)
```

You can check the result by calling `env.show()` again.

The `env.step()` method will return the following values

```
(<done>, <reward>, <state>, <actions>)
```

`<done>` is a boolean value indicating if the game reaches an end, and `<reward>` is an integer showing the reward the agent gets.

The `<state>` return value contains

```
(agent_x, agent_y, opponent_x, opponent_y)
```

The `<actions>` return value contains

```
(agent_action, opponent_action)
```

For instance, `env.step()` could return the following values

```
(True, 30, (0, 3, 2, 3), (<Move.STANDBY: 4>, <Move.LEFT: 3>))
```

#### Change the opponent policy

There are five policies to choose from
- `Opponent.Policy.ONE`
- `Opponent.Policy.TWO`
- `Opponent.Policy.THREE`
- `Opponent.Policy.FOUR`
- `Opponent.Policy.FIVE`

Set the policy using the following code (`env` being the `GridWorld` environment)

```python
env.opponent.policy = Opponent.Policy.ONE
```

You will see the following message printing on the console

```
Opponent policy is Policy.ONE
```

#### Change the agent policy

To set a policy, your agent has to be a `BprAgent`, see `BprAgent` in [`grid_world/agent.py`](grid_world/agent.py)

There are five policies to choose from
- `BprAgent.Policy.ONE`
- `BprAgent.Policy.TWO`
- `BprAgent.Policy.THREE`
- `BprAgent.Policy.FOUR`
- `BprAgent.Policy.FIVE`

Set the policy using the following code (`env` being the `GridWorld` environment)

```python
env.agent.policy = BprAgent.Policy.ONE
```

You will see the following message printing on the console

```
BprAgent policy is Policy.ONE
```

### Navigation Game

![Navigation Game](https://i.imgur.com/7dZ3hDw.png)

source: [Bayes-OKR Paper](../BPR&#32;Variants&#32;Papers/Bayes-OKR.pdf)

#### Create the environment

```python
from navigation_game.env import *
env = NavigationGame()
```

#### Reset the environment to initial state

```python
env.reset()
env.show()
```

The `env.show()` will print out the following result

```
.......△.......
...............
...............
...............
1+2++++3++++4+5
...............
...............
...............
.......○.......
```

#### Move the agent

There are five moves to choose from
- `Move.UP`
- `Move.RIGHT`
- `Move.DOWN`
- `Move.LEFT`
- `Move.STANDBY`

```python
env.step(Move.RIGHT)
```

You can check the result by calling `env.show()` again.

The `env.step()` method will return the following values

```
(<done>, <reward>, <state>, <actions>)
```

`<done>` is a boolean value indicating if the game reaches an end, and `<reward>` is an integer showing the reward the agent gets.

The `<state>` return value contains

```
(agent_x, agent_y, opponent_x, opponent_y)
```

The `<actions>` return value contains

```
(agent_action, opponent_action)
```

For instance, `env.step()` could return the following values

```
(False, -1, (7, 3, 7, 5), (<Move.DOWN: 2>, <Move.UP: 0>))
```

#### Change the opponent policy

There are five policies to choose from
- `Opponent.Policy.ONE`
- `Opponent.Policy.TWO`
- `Opponent.Policy.THREE`
- `Opponent.Policy.FOUR`
- `Opponent.Policy.FIVE`

Set the policy using the following code (`env` being the `NavigationGame` environment)

```python
env.opponent.policy = Opponent.Policy.ONE
```

You will see the following message printing on the console

```
Opponent policy is Policy.ONE
```

#### Change the agent policy

To set a policy, your agent has to be a `BprAgent`, see `BprAgent` in [`navigation_game/agent.py`](navigation_game/agent.py)

There are five policies to choose from
- `BprAgent.Policy.ONE`
- `BprAgent.Policy.TWO`
- `BprAgent.Policy.THREE`
- `BprAgent.Policy.FOUR`
- `BprAgent.Policy.FIVE`

Set the policy using the following code (`env` being the `NavigationGame` environment)

```python
env.agent.policy = BprAgent.Policy.ONE
```

You will see the following message printing on the console

```
BprAgent policy is Policy.ONE
```

### Soccer Game

![Soccer Game](https://i.imgur.com/I8k1jcw.png)

source: [Bayes-OKR Paper](../BPR&#32;Variants&#32;Papers/Bayes-OKR.pdf)

#### Create the environment

```python
from soccer_game.env import *
env = SoccerGame()
```

#### Reset the environment to initial state

```python
env.reset()
env.show()
```

The `env.show()` will print out the following result

```
 ...... 
+......+
+..△●..+
+......+
 ...... 
```

#### Move the agent

There are five moves to choose from
- `Move.UP`
- `Move.RIGHT`
- `Move.DOWN`
- `Move.LEFT`
- `Move.STANDBY`

```python
env.step(Move.RIGHT)
```

You can check the result by calling `env.show()` again.

The `env.step()` method will return the following values

```
(<done>, <reward>, <state>, <actions>)
```

`<done>` is a boolean value indicating if the game reaches an end, and `<reward>` is an integer showing the reward the agent gets.

The `<state>` return value contains

```
(agent_left_x, agent_left_y, agent_right_x, agent_right_y, ball_possession)
```

The `<actions>` return value contains

```
(agent_action, opponent_action)
```

For instance, `env.step()` could return the following values

```
(False, 0, (2, 4, 5, 3, <Possession.RIGHT: 1>), (<Move.RIGHT: 1>, <Move.DOWN: 2>))
```

#### Change the opponent policy

There are five policies to choose from
- `Opponent.Policy.ONE`
- `Opponent.Policy.TWO`
- `Opponent.Policy.THREE`
- `Opponent.Policy.FOUR`
- `Opponent.Policy.FIVE`

Set the policy using the following code (`env` being the `SoccerGame` environment)

```python
env.opponent.policy = Opponent.Policy.ONE
```

You will see the following message printing on the console

```
Opponent policy is Policy.ONE
```

#### Change the agent policy

To set a policy, your agent has to be a `BprAgent`, see `BprAgent` in [`navigation_game/agent.py`](navigation_game/agent.py)

There are five policies to choose from
- `BprAgent.Policy.ONE`
- `BprAgent.Policy.TWO`
- `BprAgent.Policy.THREE`
- `BprAgent.Policy.FOUR`
- `BprAgent.Policy.FIVE`

Set the policy using the following code (`env` being the `SoccerGame` environment)

```python
env.agent.policy = BprAgent.Policy.ONE
```

You will see the following message printing on the console

```
BprAgent policy is Policy.ONE
```