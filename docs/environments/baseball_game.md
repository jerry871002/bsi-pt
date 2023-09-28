# Extended batter vs. pitcher game (EBvPG)

The extended batter vs. pitcher game (EBvPG) is an extension of the batter vs. pitcher game (BvPG)[^bvpg]. In the BvPG, the batter's goal is to anticipate the pitcher's intended delivery location to hit the ball, while the pitcher tries to prevent the batter from making contact with the ball. The original BvPG featured only one pitch per episode, whereas the EBvPG involves multiple pitches per episode to simulate a more realistic batter vs. pitcher game scenario.

In the EBvPG game, the pitcher throws the ball until one of four possible outcomes occur: **hit**, **out**, **strikeout**, or **walk**. The batter's goal is to win each episode by successfully hitting the ball or receiving a walk. If the result is anything other than a hit or walk, the batter loses the episode.

To learn how to use this environment, please refer to [`baseball_game/run.py`](https://github.com/jerry871002/bayesian-strategy-inference/blob/master/src/baseball_game/run.py).

[^bvpg]: Lee, Wang. "Bayesian Opponent Exploitation by Inferring the Opponent’s Policy Selection Pattern." (2022)
