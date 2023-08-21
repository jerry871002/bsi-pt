import copy
import random
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
from utils import normalize_distribution

State = Tuple[int, int, int, int]
Location = Tuple[int, int]


class Move(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    STANDBY = 4


class GridWorld:
    def __init__(
        self,
        width=3,
        height=4,
        max_steps=10,  # originally 50, modify to 10 to avoid large punishment
        reward_G1=10,  # 10
        reward_G2=30,  # 30
        step_punishment=1,
        collide_punishment=1,
        agent=None,
        bpr_opponent=False,
        phi_opponent=False,
        new_phi_opponent=False,
        new_phi_noise_opponent=False,
        p_pattern=0,
        q=0,
    ):
        # set dimension of the field
        self.width = width
        self.height = height

        # set goals
        self.G1 = (0, 0)
        self.G2 = (self.width - 1, 0)

        # set rewards and punishments
        self.reward_G1 = reward_G1
        self.reward_G2 = reward_G2
        self.step_punishment = step_punishment
        self.collide_punishment = collide_punishment

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

        self.steps = 0
        self.max_steps = max_steps

        # performance models
        self.performance_models: Dict[Optional[State], np.ndarray] = {}

    def generate_performance_model(self, start_state: Optional[State] = None) -> np.ndarray:
        """
        Generate the performance model according to the environment's settings.
        Calling this function without specifying `start_state` will return the
        performance model from the initial state (i.e., the state after calling
        `env.reset()`).

        Args:
            start_state (Optional[State]): The starting state of calculating
                                           the performance model.

        Returns:
            np.ndarray: The performance model.
        """
        # if already at the goal
        # return the performance model starting from the very beginning
        if start_state is not None:
            agent_loc = start_state[:2]
            opponent_loc = start_state[2:]
            if agent_loc in (self.G1, self.G2) and opponent_loc in (self.G1, self.G2):
                start_state = None

        if start_state in self.performance_models:
            return self.performance_models[start_state]

        # do deep copies so that it won't effect the original settings
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

                if start_state is not None:
                    agent.set_xy(*start_state[:2])
                    opponent.set_xy(*start_state[2:])

                while True:
                    try:
                        done, reward, state, _ = env.step(agent.get_action(state[2:]))
                    except ValueError:
                        rewards = 0
                        break

                    rewards += reward
                    if done:
                        break

                performance_model[opponent_policy.value - 1][agent_policy.value - 1] = rewards

        self.performance_models[start_state] = np.array(performance_model)
        return self.performance_models[start_state]

    def reset(self) -> State:
        self.agent.set_xy(0, self.height - 1)
        self.opponent.set_xy(self.width - 1, self.height - 1)

        # state = (agent_x, agent_y, opponent_x, opponent_y)
        state = self.agent.get_xy() + self.opponent.get_xy()

        return state

    def step(self, agent_action: Move) -> Tuple[bool, int, State, Tuple[Move, Move]]:
        if not isinstance(agent_action, Move):
            raise ValueError('Action should be represented by the `Move` class')

        self.steps += 1

        opponent_action = self.opponent.get_action()

        # check if game is over and the rewards
        done, reward = self.game_over(agent_action, opponent_action)

        # underscore (_) after variable means next state
        agent_loc_ = self.agent.move(agent_action)
        opponent_loc_ = self.opponent.move(opponent_action)

        # check if next state locations are valid
        # if not, next state location = original location
        if not self.location_valid(agent_loc_):
            agent_loc_ = self.agent.get_xy()
        if not self.location_valid(opponent_loc_):
            opponent_loc_ = self.opponent.get_xy()

        # once a player reaches a goal
        # its position will not change
        if self.reach_goal(self.agent.get_xy()):
            agent_loc_ = self.agent.get_xy()
        if self.reach_goal(self.opponent.get_xy()):
            opponent_loc_ = self.opponent.get_xy()

        punishment = 0
        # if the players try to move into the same grid
        # the agent gets an r_collide = -5 punishment and the move fails
        if self.collide(agent_loc_, opponent_loc_):
            punishment += self.collide_punishment
            agent_loc_ = self.agent.get_xy()
            opponent_loc_ = self.opponent.get_xy()

        # the agent gets an r_step = -1 punishment
        # when executing action other than `Move.STANDBY`
        if agent_action is not Move.STANDBY:
            punishment += self.step_punishment

        self.agent.set_xy(*agent_loc_)
        self.opponent.set_xy(*opponent_loc_)

        if not done:
            reward -= punishment
        else:
            self.steps = 0

        # state = (agent_x, agent_y, opponent_x, opponent_y)
        state = self.agent.get_xy() + self.opponent.get_xy()
        actions = (agent_action, opponent_action)

        return done, reward, state, actions

    def location_valid(self, location: Location) -> bool:
        x, y = location
        return 0 <= x < self.width and 0 <= y < self.height

    def reach_goal(self, location: Location) -> bool:
        return True if location in (self.G1, self.G2) else False

    def game_over(self, agent_action: Move, opponent_action: Move) -> Tuple[bool, int]:
        if self.steps > self.max_steps:
            return True, 0

        # underscore (_) after variable means next state
        agent_loc_ = self.agent.move(agent_action)
        opponent_loc_ = self.opponent.move(opponent_action)

        # current location of agent and opponent
        agent_loc = self.agent.get_xy()
        opponent_loc = self.opponent.get_xy()

        if (agent_loc_ == self.G1 or agent_loc == self.G1) and (
            opponent_loc_ == self.G2 or opponent_loc == self.G2
        ):
            return True, self.reward_G1
        elif (agent_loc_ == self.G2 or agent_loc == self.G2) and (
            opponent_loc_ == self.G1 or opponent_loc == self.G1
        ):
            return True, self.reward_G2

        # game not end yet
        return False, 0

    def collide(self, agent_next_loc: Location, opponent_next_loc: Location) -> bool:
        agent_loc = self.agent.get_xy()
        opponent_loc = self.opponent.get_xy()
        return agent_next_loc == opponent_next_loc or (
            agent_next_loc == opponent_loc and opponent_next_loc == agent_loc
        )

    def show(self):
        agent, agent_goal = '△', '▲'
        opponent, opponent_goal = '○', '●'

        G1 = (0, 0)
        G2 = (self.width - 1, 0)

        for y in range(self.height):
            for x in range(self.width):
                # draw the goals
                if (x, y) == G1:
                    if (x, y) == self.agent.get_xy():
                        print(agent_goal, end='')
                    elif (x, y) == self.opponent.get_xy():
                        print(opponent_goal, end='')
                    else:
                        print('1', end='')
                    continue
                if (x, y) == G2:
                    if (x, y) == self.agent.get_xy():
                        print(agent_goal, end='')
                    elif (x, y) == self.opponent.get_xy():
                        print(opponent_goal, end='')
                    else:
                        print('2', end='')
                    continue

                if (x, y) == self.agent.get_xy():
                    print(agent, end='')
                elif (x, y) == self.opponent.get_xy():
                    print(opponent, end='')
                else:
                    print('.', end='')
            print()


class Agent:
    def __init__(self, x=None, y=None):
        self.set_xy(x, y)

    def move(self, action):
        # fmt: off
        moves = {
            Move.UP      : (self.x,   self.y-1),
            Move.RIGHT   : (self.x+1, self.y),
            Move.DOWN    : (self.x,   self.y+1),
            Move.LEFT    : (self.x-1, self.y),
            Move.STANDBY : (self.x  , self.y)
        }
        # fmt: on

        return moves.get(action, (self.x, self.y))

    def set_xy(self, x, y):
        self.x = x
        self.y = y

    def get_xy(self):
        return (self.x, self.y)


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
    def policy(self, new_policy: Policy) -> None:
        if not isinstance(new_policy, self.Policy):
            raise ValueError('Policy should be represented by the `Opponent.Policy` class')
        self._policy = new_policy

    def get_action(self) -> Move:
        """
        Return the action given its policy and current location.

        Raises:
            ValueError: When the location doesn't meet the policy.

        Returns:
            Move: The action to take.
        """
        # fmt: off
        if self.policy is Opponent.Policy.ONE:
            # 1   2
            #     *
            #     *
            #     s
            if self.get_xy() == (2, 3): return Move.UP
            if self.get_xy() == (2, 2): return Move.UP
            if self.get_xy() == (2, 1): return Move.UP
        elif self.policy is Opponent.Policy.TWO:
            # 1 * 2
            #   * *
            #     *
            #     s
            if self.get_xy() == (2, 3): return Move.UP
            if self.get_xy() == (2, 2): return Move.UP
            if self.get_xy() == (2, 1): return Move.LEFT
            if self.get_xy() == (1, 1): return Move.UP
            if self.get_xy() == (1, 0): return Move.LEFT
        elif self.policy is Opponent.Policy.THREE:
            # 1   2
            # * *
            #   * *
            #     s
            if self.get_xy() == (2, 3): return Move.UP
            if self.get_xy() == (2, 2): return Move.LEFT
            if self.get_xy() == (1, 2): return Move.UP
            if self.get_xy() == (1, 1): return Move.LEFT
            if self.get_xy() == (0, 1): return Move.UP
        elif self.policy is Opponent.Policy.FOUR:
            # 1 * 2
            #   *
            #   *
            #   * s
            if self.get_xy() == (2, 3): return Move.LEFT
            if self.get_xy() == (1, 3): return Move.UP
            if self.get_xy() == (1, 2): return Move.UP
            if self.get_xy() == (1, 1): return Move.UP
            if self.get_xy() == (1, 0): return Move.LEFT
        elif self.policy is Opponent.Policy.FIVE:
            # 1 * 2
            #   *
            #   *
            #   * s
            if self.get_xy() == (2, 3): return Move.LEFT
            if self.get_xy() == (1, 3): return Move.UP
            if self.get_xy() == (1, 2): return Move.UP
            if self.get_xy() == (1, 1): return Move.UP
            if self.get_xy() == (1, 0): return Move.RIGHT
        # fmt: on

        # `Move.STANDBY` when already arrived at the goal
        if self.get_xy() in ((0, 0), (2, 0)):
            return Move.STANDBY

        raise ValueError(f'Opponent with policy {self._policy} shouldn\'t be in {self.get_xy()}')

    def switch_policy(self, switch_point: Location, policy1: Policy, policy2: Policy):
        if self.get_xy() == switch_point:
            if self.policy is policy1:
                self.policy = policy2
            elif self.policy is policy2:
                self.policy = policy1


class BprOpponent(Opponent):
    def __init__(self, x=None, y=None):
        super().__init__(x, y)

        self.n_policies = len(self.Policy)
        self._belief = np.ones(self.n_policies) / self.n_policies  # initial as uniform distribution

        # should be set before the game starts
        self.performance_model = None

    @property
    def belief(self):
        return self._belief

    @belief.setter
    def belief(self, new_belief):
        if not isinstance(new_belief, np.ndarray) or new_belief.shape != (self.n_policies,):
            raise ValueError(f'Belief should be a numpy array with {self.n_policies} items')
        self._belief = new_belief

    def update_belief(self, utility: int) -> None:
        """
        Update the belief according to the utility the agent gets using
        posterior (belief) = prior * likelihood (performance model).
        Notice that the utility is from the agent's perspective.

        Args:
            utility (int): The reward the agent gets in a episode.
        """
        # find the currently observed utility in the performance model
        likelihood = (self.performance_model[self.policy.value - 1] == utility).astype(float)
        new_belief_unnormalized = (
            self.belief * likelihood / (np.sum(likelihood * self.belief) + 1e-6)
        )
        self.belief = normalize_distribution(new_belief_unnormalized, 0.01)

    def update_policy(self) -> None:
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
        terminal_state_combination = get_terminal_state_combination(ternimal_state)

        # phi 1 to phi 6 always keep the same policy
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
            if terminal_state_combination[0] or terminal_state_combination[5]:
                self.policy = self.Policy.ONE
            elif terminal_state_combination[1] or terminal_state_combination[6]:
                self.policy = self.Policy.TWO
            elif terminal_state_combination[2]:
                self.policy = self.Policy.THREE
            elif terminal_state_combination[3]:
                self.policy = self.Policy.FOUR
            elif terminal_state_combination[4]:
                self.policy = self.Policy.FIVE
        # phi 8
        elif self.phi is self.Phi.EIGHT:
            if any(
                (
                    terminal_state_combination[0],
                    terminal_state_combination[3],
                    terminal_state_combination[5],
                )
            ):
                self.policy = self.Policy.ONE
            elif terminal_state_combination[2]:
                self.policy = self.Policy.TWO
            elif terminal_state_combination[6]:
                self.policy = self.Policy.THREE
            elif terminal_state_combination[1]:
                self.policy = self.Policy.FOUR
            elif terminal_state_combination[4]:
                self.policy = self.Policy.FIVE
        # phi 9
        elif self.phi is self.Phi.NINE:
            if terminal_state_combination[0]:
                self.policy = self.Policy.FIVE
            elif terminal_state_combination[1]:
                self.policy = self.Policy.FIVE
            elif terminal_state_combination[2]:
                self.policy = self.Policy.THREE
            elif terminal_state_combination[3]:
                self.policy = self.Policy.THREE
            elif terminal_state_combination[4]:
                self.policy = self.Policy.TWO
            elif terminal_state_combination[5]:
                self.policy = self.Policy.ONE
            elif terminal_state_combination[6]:
                self.policy = self.Policy.FOUR
        # phi 10
        elif self.phi is self.Phi.TEN:
            if terminal_state_combination[0]:
                self.policy = self.Policy.FIVE
            elif terminal_state_combination[1]:
                self.policy = self.Policy.ONE
            elif terminal_state_combination[2]:
                self.policy = self.Policy.TWO
            elif terminal_state_combination[3]:
                self.policy = self.Policy.TWO
            elif terminal_state_combination[4]:
                self.policy = self.Policy.THREE
            elif terminal_state_combination[5]:
                self.policy = self.Policy.FOUR
            elif terminal_state_combination[6]:
                self.policy = self.Policy.FIVE
        # phi 11: placeholder for unknown policy
        elif self.phi is self.Phi.ELEVEN:
            if terminal_state_combination[0]:
                self.policy = self.Policy.FIVE
            elif terminal_state_combination[1]:
                self.policy = self.Policy.ONE
            elif terminal_state_combination[2]:
                self.policy = self.Policy.TWO
            elif terminal_state_combination[3]:
                self.policy = self.Policy.TWO
            elif terminal_state_combination[4]:
                self.policy = self.Policy.THREE
            elif terminal_state_combination[5]:
                self.policy = self.Policy.FOUR
            elif terminal_state_combination[6]:
                self.policy = self.Policy.FIVE


class NewPhiOpponent(Opponent):
    def __init__(self, q: int, x=None, y=None) -> None:
        super().__init__(x, y)
        self.strategy_library = {
            7: [1, 2, 3, 4, 5, 1, 2],  # phi 7
            8: [1, 4, 2, 1, 5, 1, 3],  # phi 8
            9: [5, 5, 3, 3, 2, 1, 4],  # phi 9
            10: [5, 1, 2, 2, 3, 4, 5],  # phi 10
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
        terminal_state_combination = get_terminal_state_combination(ternimal_state)
        self.policy = self.Policy(self.strategy[terminal_state_combination.index(True)])


class NewPhiNoiseOpponent(Opponent):
    def __init__(self, p_pattern: float, x=None, y=None) -> None:
        super().__init__(x, y)
        self.p_pattern = p_pattern
        self.strategy_library = {
            7: [1, 2, 3, 4, 5, 1, 2],  # phi 7
            8: [1, 4, 2, 1, 5, 1, 3],  # phi 8
            9: [5, 5, 3, 3, 2, 1, 4],  # phi 9
            10: [5, 1, 2, 2, 3, 4, 5],  # phi 10
        }
        self.corresponding_phi = random.choice(list(self.strategy_library.keys()))
        self.strategy = self.strategy_library[self.corresponding_phi]

    def update_policy(self, ternimal_state: Tuple[Location, Location]) -> None:
        terminal_state_combination = get_terminal_state_combination(ternimal_state)

        if random.random() > self.p_pattern:
            self.policy = self.Policy(random.randint(1, 5))
        else:
            self.policy = self.Policy(self.strategy[terminal_state_combination.index(True)])


def get_terminal_state_combination(
    ternimal_state: Tuple[Location, Location]
) -> Tuple[bool, bool, bool, bool, bool, bool, bool]:
    agent_location, opponent_location = ternimal_state

    # goals
    G1 = (0, 0)
    G2 = (2, 0)

    return (
        opponent_location == G1 and agent_location == G2,
        opponent_location == G1 and agent_location not in (G1, G2),
        opponent_location == G2 and agent_location == G1,
        opponent_location == G2 and agent_location not in (G1, G2),
        opponent_location not in (G1, G2) and agent_location == G1,
        opponent_location not in (G1, G2) and agent_location == G2,
        opponent_location not in (G1, G2) and agent_location not in (G1, G2),
    )
