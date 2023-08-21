import copy
import random
from enum import Enum
from typing import List, Tuple

import numpy as np
from utils import normalize_distribution

Location = Tuple[int, int]


class Move(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    STANDBY = 4


class SoccerGame:
    class Possession(Enum):
        LEFT = 0
        RIGHT = 1

    def __init__(
        self,
        width=6,
        height=5,
        goal_size=3,
        max_steps=15,
        reward_G1=10,
        reward_G2=30,
        reward_G3=10,
        step_punishment=1,
        agent=None,
        bpr_opponent=False,
        phi_opponent=False,
        new_phi_opponent=False,
        new_phi_noise_opponent=False,
        p_pattern=0,
        q=0,
    ):
        # check if the dimension is valid
        if width < 2:
            raise ValueError('`width` must be greater than 2')
        if not 0 < goal_size <= height:
            raise ValueError('`goal_size` must be greater than 0 and smaller or equal to `height`')
        if (height - goal_size) % 2 != 0:
            raise ValueError('`height` and `goal_size` must both be odd or even')

        # set dimension of the field
        self.width = width
        self.height = height
        self.goal_size = goal_size

        # set goals
        self.G1 = (self.width, (self.height - self.goal_size) // 2)
        self.G2 = (self.width, self.height // 2)
        self.G3 = (self.width, (self.height + self.goal_size) // 2 - 1)

        self.G1_op = (-1, (self.height - self.goal_size) // 2)
        self.G2_op = (-1, self.height // 2)
        self.G3_op = (-1, (self.height + self.goal_size) // 2 - 1)

        # set rewards
        self.reward_G1 = reward_G1
        self.reward_G2 = reward_G2
        self.reward_G3 = reward_G3
        self.step_punishment = step_punishment

        # initialize agent and opponent
        if agent is not None and not isinstance(agent, Agent):
            raise ValueError('Agent should be subclass of the `Agent` class')
        self.agent = agent if agent else Agent()

        if bpr_opponent and phi_opponent:
            raise ValueError('Can only choose one between BPR opponent and Phi opponent')
        if bpr_opponent:
            self.opponent = BprOpponent()
        elif phi_opponent:
            self.opponent = PhiOpponent()
        elif new_phi_opponent:
            self.opponent = NewPhiOpponent(q=q)
        elif new_phi_noise_opponent:
            self.opponent = NewPhiNoiseOpponent(p_pattern=p_pattern)
        else:
            self.opponent = Opponent()

        self.ball_possession = None

        self.steps = 0
        self.max_steps = max_steps

    def generate_performance_model(self) -> np.ndarray:
        """
        Generate the performance model according to the environment's settings.

        Returns:
            np.ndarray: The performance model.
        """
        agent = copy.deepcopy(self.agent)
        opponent = copy.deepcopy(self.opponent)
        env = copy.deepcopy(self)
        env.agent = agent
        env.opponent = opponent

        # each row represent an opponent policy (tau)
        # each column represent an agent policy (pi)
        performance_model = [[0 for _ in list(agent.Policy)] for _ in list(opponent.Policy)]

        for agent_policy in list(agent.Policy):
            for opponent_policy in list(opponent.Policy):
                agent.policy = agent_policy
                opponent.policy = opponent_policy
                rewards = 0
                state = env.reset()
                while True:
                    done, reward, state, _ = env.step(agent.get_action(state[2:4]))
                    rewards += reward
                    if done:
                        break
                performance_model[opponent_policy.value - 1][agent_policy.value - 1] = rewards

        return np.array(performance_model)

    def reset(self):
        self.agent.set_xy(2, 2)
        self.opponent.set_xy(3, 2)

        # the ball is asigned to the opponent at the beginning of each episode
        self.ball_possession = SoccerGame.Possession.RIGHT
        self.opponent.ball_possession = True
        self.agent.ball_possession = False

        # state = (agent_x, agent_y, opponent_x, opponent_y, ball_possession)
        state = self.agent.get_xy() + self.opponent.get_xy() + (self.ball_possession,)

        return state

    def step(self, agent_action):
        if not isinstance(agent_action, Move):
            raise ValueError('Action should be represented by the `Move` class')

        self.steps += 1

        opponent_action = self.opponent.get_action()

        # check if game is over and the rewards
        done, reward = self.game_over(agent_action, opponent_action)

        # underscore (_) after variable means next state
        agent_loc_ = self.agent.move(agent_action)
        opponent_loc_ = self.opponent.move(opponent_action)

        if not done:
            # check if next state locations are valid
            # if not, next state location = original location
            if not self.location_valid(agent_loc_):
                agent_loc_ = self.agent.get_xy()
            if not self.location_valid(opponent_loc_):
                opponent_loc_ = self.opponent.get_xy()

            if self.change_possesion(agent_loc_, opponent_loc_):
                # switch ball possession
                if self.ball_possession is SoccerGame.Possession.LEFT:
                    self.ball_possession = SoccerGame.Possession.RIGHT
                    self.opponent.ball_possession = True
                    self.agent.ball_possession = False
                else:
                    self.ball_possession = SoccerGame.Possession.LEFT
                    self.opponent.ball_possession = False
                    self.agent.ball_possession = True

                # if ball possession switched, next state locations = original locations
                agent_loc_ = self.agent.get_xy()
                opponent_loc_ = self.opponent.get_xy()

            reward -= self.step_punishment
        else:
            self.steps = 0

        self.agent.set_xy(*agent_loc_)
        self.opponent.set_xy(*opponent_loc_)

        # state = (agent_x, agent_y, opponent_x, opponent_y, ball_possession)
        state = self.agent.get_xy() + self.opponent.get_xy() + (self.ball_possession,)
        actions = (agent_action, opponent_action)

        return done, reward, state, actions

    def location_valid(self, location):
        x, y = location
        if 0 <= x < self.width and 0 <= y < self.height:
            return True
        else:
            return False

    def game_over(self, agent_action, opponent_action):
        if self.steps > self.max_steps:
            return True, 0

        # underscore (_) after variable means next state
        agent_x_, agent_y_ = self.agent.move(agent_action)
        # agent arrives at the goal
        if (
            self.ball_possession is SoccerGame.Possession.LEFT
            and agent_x_ == self.width
            and (self.height - self.goal_size) // 2
            <= agent_y_
            <= (self.height + self.goal_size) // 2 - 1
        ):
            # check if the goal is consistent with the opponent's policy
            # return if_game_over, reward
            if (
                self.opponent.policy in (Opponent.Policy.ONE, Opponent.Policy.TWO)
                and (agent_x_, agent_y_) == self.G1
            ):
                return True, self.reward_G1
            elif self.opponent.policy is Opponent.Policy.THREE and (agent_x_, agent_y_) == self.G2:
                return True, self.reward_G2
            elif (
                self.opponent.policy in (Opponent.Policy.FOUR, Opponent.Policy.FIVE)
                and (agent_x_, agent_y_) == self.G3
            ):
                return True, self.reward_G3

            return True, 0

        opponent_x_, opponent_y_ = self.opponent.move(opponent_action)
        # opponent arrives at the goal
        if (
            self.ball_possession is SoccerGame.Possession.RIGHT
            and opponent_x_ == -1
            and (self.height - self.goal_size) // 2
            <= opponent_y_
            <= (self.height + self.goal_size) // 2 - 1
        ):
            # check which goal is opponent arriving at
            # return if_game_over, reward
            if (opponent_x_, opponent_y_) == self.G1_op:
                return True, -self.reward_G1
            elif (opponent_x_, opponent_y_) == self.G2_op:
                return True, -self.reward_G2
            elif (opponent_x_, opponent_y_) == self.G3_op:
                return True, -self.reward_G3

            return True, 0

        # game not end yet
        return False, 0

    def change_possesion(self, al_next_loc, ar_next_loc):
        al_current_loc = self.agent.get_xy()
        ar_current_loc = self.opponent.get_xy()

        if (
            al_current_loc == ar_next_loc and ar_current_loc == al_next_loc
        ) or al_next_loc == ar_next_loc:
            return True
        else:
            return False

    def show(self):
        if self.ball_possession is SoccerGame.Possession.LEFT:
            left = '▲'
            right = '○'
        else:
            left = '△'
            right = '●'

        for y in range(self.height):
            for x in range(-1, self.width + 1):
                # draw the goals
                if (x == -1 or x == self.width) and (
                    (self.height - self.goal_size) // 2
                    <= y
                    <= (self.height + self.goal_size) // 2 - 1
                ):
                    print('+', end='')
                    continue
                elif x == -1 or x == self.width:
                    print(' ', end='')
                    continue

                if (x, y) == self.agent.get_xy():
                    print(left, end='')
                elif (x, y) == self.opponent.get_xy():
                    print(right, end='')
                else:
                    print('.', end='')
            print()


class Agent:
    def __init__(self, x=None, y=None):
        self.set_xy(x, y)
        self._ball_possession = False

    def move(self, action):
        moves = {
            Move.UP: (self.x, self.y - 1),
            Move.RIGHT: (self.x + 1, self.y),
            Move.DOWN: (self.x, self.y + 1),
            Move.LEFT: (self.x - 1, self.y),
            Move.STANDBY: (self.x, self.y),
        }

        return moves.get(action, (self.x, self.y))

    def set_xy(self, x, y):
        self.x = x
        self.y = y

    def get_xy(self):
        return (self.x, self.y)

    @property
    def ball_possession(self):
        return self._ball_possession

    @ball_possession.setter
    def ball_possession(self, possession):
        if type(possession) != bool:
            raise ValueError(
                f'Ball possession should be a boolean value, invalid value: {possession}'
            )
        self._ball_possession = possession


class Opponent(Agent):
    class Policy(Enum):
        ONE = 1
        TWO = 2
        THREE = 3
        FOUR = 4
        FIVE = 5

    def __init__(self, x=None, y=None):
        super().__init__(x, y)
        self._policy = None

    @property
    def policy(self):
        return self._policy

    @policy.setter
    def policy(self, new_policy):
        if not isinstance(new_policy, Opponent.Policy):
            raise ValueError('Policy should be represented by the `Opponent.Policy` class')
        self._policy = new_policy

    def get_action(self):
        if self._policy is Opponent.Policy.ONE:
            #    * * * *
            # G1 *     *
            #          s
            if self.get_xy() == (3, 2):
                return Move.UP
            if self.get_xy() == (3, 1):
                return Move.UP
            if self.get_xy() == (3, 0):
                return Move.LEFT
            if self.get_xy() == (2, 0):
                return Move.LEFT
            if self.get_xy() == (1, 0):
                return Move.LEFT
            if self.get_xy() == (0, 0):
                return Move.DOWN
            if self.get_xy() == (0, 1):
                return Move.LEFT
        elif self._policy is Opponent.Policy.TWO:
            # G1 * * * *
            #          s
            if self.get_xy() == (3, 2):
                return Move.UP
            if self.get_xy() == (3, 1):
                return Move.LEFT
            if self.get_xy() == (2, 1):
                return Move.LEFT
            if self.get_xy() == (1, 1):
                return Move.LEFT
            if self.get_xy() == (0, 1):
                return Move.LEFT
        elif self._policy is Opponent.Policy.THREE:
            # G2 * * * s
            if self.get_xy() == (3, 2):
                return Move.LEFT
            if self.get_xy() == (2, 2):
                return Move.LEFT
            if self.get_xy() == (1, 2):
                return Move.LEFT
            if self.get_xy() == (0, 2):
                return Move.LEFT
        elif self._policy is Opponent.Policy.FOUR:
            #          s
            # G3 * * * *
            if self.get_xy() == (3, 2):
                return Move.DOWN
            if self.get_xy() == (3, 3):
                return Move.LEFT
            if self.get_xy() == (2, 3):
                return Move.LEFT
            if self.get_xy() == (1, 3):
                return Move.LEFT
            if self.get_xy() == (0, 3):
                return Move.LEFT
        elif self._policy is Opponent.Policy.FIVE:
            #          s
            # G3 *     *
            #    * * * *
            if self.get_xy() == (3, 2):
                return Move.DOWN
            if self.get_xy() == (3, 3):
                return Move.DOWN
            if self.get_xy() == (3, 4):
                return Move.LEFT
            if self.get_xy() == (2, 4):
                return Move.LEFT
            if self.get_xy() == (1, 4):
                return Move.LEFT
            if self.get_xy() == (0, 4):
                return Move.UP
            if self.get_xy() == (0, 3):
                return Move.LEFT

        raise ValueError(f'Opponent with policy {self._policy} shouldn\'t be in {self.get_xy()}')


class BprOpponent(Opponent):
    def __init__(self, x=None, y=None):
        super().__init__(x, y)

        self.n_policies = len(Opponent.Policy)
        self._belief = np.ones(self.n_policies) / self.n_policies  # initial as uniform distribution

        # should be set before the game starts
        self.performance_model = None

    @property
    def belief(self):
        return self._belief

    @belief.setter
    def belief(self, new_belief):
        if not isinstance(new_belief, np.ndarray) or new_belief.shape != (self.n_policies,):
            raise ValueError(f'New policy should be a numpy array with {self.n_policies} items')
        self._belief = new_belief

    def update_belief(self, utility: int) -> None:
        """
        Update the belief according to the utility the agent gets.
        Notice that the utility is from the agent's perspective.

        Args:
            utility (int): The reward the agent gets in a episode.
        """
        # posterior (belief) = prior * likelihood (performance model)
        likelihood = (self.performance_model[self.policy.value - 1] == utility).astype(
            float
        )  # find the currently observed utility in the performance model
        new_belief_unnormalized = (
            self.belief * likelihood / (np.sum(likelihood * self.belief) + 1e-6)
        )
        self.belief = normalize_distribution(new_belief_unnormalized, 0.01)

    def update_policy(self):
        belief_mul_performance = self.belief @ np.transpose(self.performance_model)
        candidates = (
            np.argwhere(belief_mul_performance == np.amin(belief_mul_performance))
            .flatten()
            .tolist()
        )
        self.policy = list(Opponent.Policy)[random.choice(candidates)]


class PhiOpponent(Opponent):
    class Phi(Enum):
        ONE = 1
        TWO = 2
        THREE = 3
        FOUR = 4
        FIVE = 5
        SIX = 6
        SEVEN = 7
        EIGHT = 8
        NINE = 9
        TEN = 10
        ELEVEN = 11  # unknown policy

    def __init__(self, x=None, y=None):
        super().__init__(x, y)
        self._policy: self.Policy = None
        self._phi: self.Phi = None

    @property
    def phi(self):
        return self._phi

    @phi.setter
    def phi(self, new_phi):
        if not isinstance(new_phi, self.Phi):
            raise ValueError(
                'Phi should be represented by `PhiOpponent.Phi` class, '
                f'invalid value: {new_phi} ({type(new_phi)})'
            )
        self._phi = new_phi

    def update_policy(self, ternimal_state: Tuple[Location, Location]) -> None:
        """
        Update the policy of the opponent according to phi and sigma (terminal state).

        Args:
            ternimal_state (Tuple[Location, Location]): Opponent (x, y) + Agent (x, y)
        """
        terminal_state_combination = get_terminal_state_combination(
            self.ball_possession, ternimal_state
        )

        # phi 1 to phi 5 always keep the same policy
        if self.phi is self.Phi.ONE:
            self.policy = self.Policy.ONE
        elif self.phi is self.Phi.TWO:
            self.policy = self.Policy.TWO
        elif self.phi is self.Phi.THREE:
            self.policy = self.Policy.THREE
        elif self.phi is self.Phi.FOUR:
            self.policy = self.Policy.FOUR
        elif self.phi is self.Phi.FIVE:
            self.policy = self.Policy.FIVE
        # phi 6 randomly chooses its policy
        elif self.phi is self.Phi.SIX:
            self.policy = random.choice(list(self.Policy))
        # phi 7
        elif self.phi is self.Phi.SEVEN:
            if terminal_state_combination[3]:
                self.policy = self.Policy.ONE
            elif terminal_state_combination[1]:
                self.policy = self.Policy.TWO
            elif terminal_state_combination[2] or terminal_state_combination[4]:
                self.policy = self.Policy.THREE
            elif terminal_state_combination[0] or terminal_state_combination[5]:
                self.policy = self.Policy.FOUR
        # phi 8
        elif self.phi is self.Phi.EIGHT:
            if terminal_state_combination[2] or terminal_state_combination[5]:
                self.policy = self.Policy.ONE
            elif terminal_state_combination[1]:
                self.policy = self.Policy.TWO
            elif terminal_state_combination[0] or terminal_state_combination[4]:
                self.policy = self.Policy.FOUR
            elif terminal_state_combination[3]:
                self.policy = self.Policy.FIVE
        # phi 9
        elif self.phi is self.Phi.NINE:
            if terminal_state_combination[0]:
                self.policy = self.Policy.THREE
            elif terminal_state_combination[1]:
                self.policy = self.Policy.FOUR
            elif terminal_state_combination[2]:
                self.policy = self.Policy.TWO
            elif terminal_state_combination[3]:
                self.policy = self.Policy.TWO
            elif terminal_state_combination[4]:
                self.policy = self.Policy.THREE
            elif terminal_state_combination[5]:
                self.policy = self.Policy.FOUR
        # phi 10
        elif self.phi is self.Phi.TEN:
            if terminal_state_combination[0]:
                self.policy = self.Policy.THREE
            elif terminal_state_combination[1]:
                self.policy = self.Policy.FIVE
            elif terminal_state_combination[2]:
                self.policy = self.Policy.TWO
            elif terminal_state_combination[3]:
                self.policy = self.Policy.FOUR
            elif terminal_state_combination[4]:
                self.policy = self.Policy.THREE
            elif terminal_state_combination[5]:
                self.policy = self.Policy.ONE
        # phi 11: placeholder for unknown policy
        elif self.phi is self.Phi.ELEVEN:
            if terminal_state_combination[0]:
                self.policy = self.Policy.THREE
            elif terminal_state_combination[1]:
                self.policy = self.Policy.FIVE
            elif terminal_state_combination[2]:
                self.policy = self.Policy.TWO
            elif terminal_state_combination[3]:
                self.policy = self.Policy.FOUR
            elif terminal_state_combination[4]:
                self.policy = self.Policy.THREE
            elif terminal_state_combination[5]:
                self.policy = self.Policy.ONE


class NewPhiOpponent(Opponent):
    def __init__(self, q: int, x=None, y=None) -> None:
        super().__init__(x, y)
        self.strategy_library = {
            7: [4, 2, 3, 1, 3, 4],  # phi 7
            8: [4, 2, 1, 5, 4, 1],  # phi 8
            9: [3, 4, 2, 2, 3, 4],  # phi 9
            10: [3, 5, 2, 4, 3, 1],  # phi 10
        }
        self.corresponding_phi, self.strategy = self.generate_strategy(q)

    def generate_strategy(self, q: int) -> Tuple[int, List[int]]:
        def modify_strategy(base_strategy: List[int], q: int) -> List[int]:
            if len(base_strategy) < q:
                raise ValueError('q cannot be larger than total numbers of terminal states')

            new_strategy = list(base_strategy)
            for terminal_state_idx in random.sample(range(len(new_strategy)), k=q):
                original_policy = new_strategy[terminal_state_idx]
                candidates = [policy.value for policy in self.Policy]
                candidates.remove(original_policy)
                new_strategy[terminal_state_idx] = random.choice(candidates)

            return new_strategy

        def distance(strategy1: List[int], strategy2: List[int]) -> int:
            return np.sum(np.array(strategy1) != np.array(strategy2))

        base_strategy_idx = random.choice(list(self.strategy_library.keys()))
        base_strategy = self.strategy_library[base_strategy_idx]

        while True:
            new_strategy = modify_strategy(base_strategy, q)

            valid_new_strategy = True
            for strategy in self.strategy_library.values():
                if distance(new_strategy, strategy) < q:
                    valid_new_strategy = False
                    break

            if valid_new_strategy:
                break

        return base_strategy_idx, new_strategy

    def update_policy(self, ternimal_state: Tuple[Location, Location]) -> None:
        """
        Update the policy of the opponent according to phi and sigma (terminal state).

        Args:
            ternimal_state (Tuple[Location, Location]): Opponent (x, y) + Agent (x, y)
        """
        terminal_state_combination = get_terminal_state_combination(
            self.ball_possession, ternimal_state
        )
        self.policy = self.Policy(self.strategy[terminal_state_combination.index(True)])


class NewPhiNoiseOpponent(Opponent):
    def __init__(self, p_pattern: float, x=None, y=None) -> None:
        super().__init__(x, y)
        self.p_pattern = p_pattern
        self.strategy_library = {
            7: [4, 2, 3, 1, 3, 4],  # phi 7
            8: [4, 2, 1, 5, 4, 1],  # phi 8
            9: [3, 4, 2, 2, 3, 4],  # phi 9
            10: [3, 5, 2, 4, 3, 1],  # phi 10
        }
        self.corresponding_phi = random.choice(list(self.strategy_library.keys()))
        self.strategy = self.strategy_library[self.corresponding_phi]

    def update_policy(self, ternimal_state: Tuple[Location, Location]) -> None:
        terminal_state_combination = get_terminal_state_combination(
            self.ball_possession, ternimal_state
        )

        if random.random() > self.p_pattern:
            self.policy = self.Policy(random.randint(1, 5))
        else:
            self.policy = self.Policy(self.strategy[terminal_state_combination.index(True)])


def get_terminal_state_combination(
    ball_possession: bool, ternimal_state: Tuple[Location, Location]
) -> Tuple[bool, bool, bool, bool, bool, bool]:
    agent_location, opponent_location = ternimal_state

    # goals
    G1_left = (-1, 1)
    G2_left = (-1, 2)
    G3_left = (-1, 3)
    G1_right = (6, 1)
    G2_right = (6, 2)
    G3_right = (6, 3)

    return (
        not ball_possession and agent_location == G1_right,
        not ball_possession and agent_location == G2_right,
        not ball_possession and agent_location == G3_right,
        ball_possession and opponent_location == G1_left,
        ball_possession and opponent_location == G2_left,
        ball_possession and opponent_location == G3_left,
    )
