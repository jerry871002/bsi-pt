# Soccer Game

![Soccer Game](https://hackmd.io/_uploads/ryloeQ9cn.png)

In Soccer Game, the agent with $\pi_i$ aims to approach the opponent with $\tau_i$ to steal the ball and then brings the ball to the corresponding goal on the right side.

## Create the environment

```python
from soccer_game.env import *
env = SoccerGame()
```

## Reset the environment to initial state

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

## Change the opponent policy

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

## Change the agent policy

To set a policy, your agent has to be a `BprAgent`, see `BprAgent` in [`soccer_game/agent.py`](https://github.com/jerry871002/bayesian-strategy-inference/blob/master/src/soccer_game/agent.py)

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
