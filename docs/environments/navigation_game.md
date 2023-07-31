# Navigation Game

![Navigation Game](https://hackmd.io/_uploads/ryloeQ9cn.png)

In Navigation Game, the agent with policy $\pi_i$ aims to move to the same goal that the opponent with $\tau_i$ plans to reach. Note that any action that causes the agent to pass through the obstacles (the grey grids) or to move out of the field is ignored.

## Create the environment

```python
from navigation_game.env import *
env = NavigationGame()
```

## Reset the environment to initial state

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

## Move the agent

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

## Change the opponent policy

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

## Change the agent policy

To set a policy, your agent has to be a `BprAgent`, see `BprAgent` in [`navigation_game/agent.py`](https://github.com/jerry871002/bayesian-strategy-inference/blob/master/src/navigation_game/agent.py)

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