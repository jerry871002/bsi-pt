import random
from enum import Enum
from typing import List, Tuple

import numpy as np

from utils import normalize_distribution

State = List[int]


class Move(Enum):
    # the location to which the pitcher decides to throw or the batter decides to hit
    # see Fig. 2(a) in the paper
    STRIKE_ALL = 0
    STRIKE_1 = 1
    STRIKE_2 = 2
    STRIKE_3 = 3
    STRIKE_4 = 4
    BALL_5 = 5
    BALL_6 = 6
    BALL_7 = 7
    BALL_8 = 8


class EpisodeResult(Enum):
    HIT = 0
    OUT = 1
    STRIKE_OUT = 2
    WALK = 3


class BaseballGame:
    def __init__(
        self,
        reward_out=0,
        reward_score=1,
        agent=None,
        bpr_opponent=False,
        phi_opponent=False,
        new_phi_opponent=False,
        new_phi_noise_opponent=False,
        p_pattern=0,
        q=0,
    ):
        # set rewards and punishment
        self.reward_out = reward_out
        self.reward_score = reward_score
        self.hit_count = 0  # agent swings and hits and scores
        self.walk_count = 0  # agent gets 4 balls, gets base on balls
        self.strike_out_count = 0  # agent gets 3 strikes, agent is out
        self.hit_out_count = 0  # agent swings and hits and out

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
        elif new_phi_noise_opponent:
            self.opponent = NewPhiNoiseOpponent(p_pattern=p_pattern)
        else:
            self.opponent = Opponent()

        self.steps = 0
        self._state = [0, 0]

    def generate_performance_model(self) -> np.ndarray:
        """
        Generate the performance model according to the environment's settings.

        Returns:
            np.ndarray: The performance model.
        """
        # see Table 3 in the paper
        performance_model = [
            [0.448, 0.254, 0.254, 0.247],
            [0.254, 0.448, 0.247, 0.254],
            [0.254, 0.247, 0.448, 0.254],
            [0.247, 0.254, 0.254, 0.448],
        ]

        return np.array(performance_model)

    def reset(self) -> State:
        self.state = [0, 0]
        self.steps = 0
        return self.state

    def step(self, agent_action: Move, stochastic=True) -> Tuple[bool, int, State, Tuple[Move, Move], EpisodeResult]:
        # return done, reward, next_state, [agent_action, opponent_action]
        if not isinstance(agent_action, Move):
            raise ValueError('Action should be represented by the `Move` class')
        self.steps += 1
        opponent_action = self.opponent.get_action(self.state)
        if stochastic:
            # see where the ball actually goes base on a probability distribution
            opponent_real_action = self.generate_real_action(opponent_action)
            # print(f'Agent action a={agent_action.value}')
            # print(f'env.py 109 op action = {opponent_action}')
            # print(f'Opponent action o={opponent_action.value}')
            # print(f'Opponent actual action o\'={opponent_real_action[0].value}')
            swing_prob, hit_prob, score_prob = self.generate_probs(agent_action, opponent_real_action)
            # print(f'Agent swing(u1), hit(v1), score(w1) probabilities = {round(swing_prob, 2)}, {round(hit_prob, 2)}, {round(score_prob, 2)}')
        else:
            opponent_real_action = opponent_action
            if agent_action == opponent_real_action:
                # agent's swing location matches opponent's pitch location
                swing_prob = 1
                hit_prob = 1
                score_prob = 1
            else:
                swing_prob = 0
                hit_prob = 0
                score_prob = 0

        done, reward, state_, result = self.generate_result(
            swing_prob, hit_prob, score_prob, self.state.copy(), opponent_real_action
        )
        actions = (agent_action, opponent_real_action)
        return done, reward, state_, actions, result

    def generate_real_action(self, action) -> Move:
        # given the opponent's choice of action, return what the opponent actually does
        # TODO: should match the probability defined in Agent.Intra_belief_model.opponent_model
        ball_control_k1 = 1
        ball_control_k2 = (1 - ball_control_k1) / 14
        options = [
            Move.STRIKE_1,
            Move.STRIKE_2,
            Move.STRIKE_3,
            Move.STRIKE_4,
            Move.BALL_5,
            Move.BALL_6,
            Move.BALL_7,
            Move.BALL_8,
        ]
        if action == Move.STRIKE_1:
            real_action = np.random.choice(
                options,
                1,
                p=[
                    ball_control_k1,
                    3 * ball_control_k2,
                    3 * ball_control_k2,
                    2 * ball_control_k2,
                    3 * ball_control_k2,
                    ball_control_k2,
                    ball_control_k2,
                    ball_control_k2,
                ],
            )
        elif action == Move.STRIKE_2:
            real_action = np.random.choice(
                options,
                1,
                p=[
                    3 * ball_control_k2,
                    ball_control_k1,
                    2 * ball_control_k2,
                    3 * ball_control_k2,
                    ball_control_k2,
                    3 * ball_control_k2,
                    ball_control_k2,
                    ball_control_k2,
                ],
            )
        elif action == Move.STRIKE_3:
            real_action = np.random.choice(
                options,
                1,
                p=[
                    3 * ball_control_k2,
                    2 * ball_control_k2,
                    ball_control_k1,
                    3 * ball_control_k2,
                    ball_control_k2,
                    ball_control_k2,
                    3 * ball_control_k2,
                    ball_control_k2,
                ],
            )
        elif action == Move.STRIKE_4:
            real_action = np.random.choice(
                options,
                1,
                p=[
                    2 * ball_control_k2,
                    3 * ball_control_k2,
                    3 * ball_control_k2,
                    ball_control_k1,
                    ball_control_k2,
                    ball_control_k2,
                    ball_control_k2,
                    3 * ball_control_k2,
                ],
            )

        ball_control_k2 = (1 - ball_control_k1) / 16

        if action == Move.BALL_5:
            real_action = np.random.choice(
                options,
                1,
                p=[
                    3 * ball_control_k2,
                    2 * ball_control_k2,
                    2 * ball_control_k2,
                    2 * ball_control_k2,
                    ball_control_k1,
                    3 * ball_control_k2,
                    3 * ball_control_k2,
                    ball_control_k2,
                ],
            )
        elif action == Move.BALL_6:
            real_action = np.random.choice(
                options,
                1,
                p=[
                    2 * ball_control_k2,
                    3 * ball_control_k2,
                    2 * ball_control_k2,
                    2 * ball_control_k2,
                    3 * ball_control_k2,
                    ball_control_k1,
                    ball_control_k2,
                    3 * ball_control_k2,
                ],
            )
        elif action == Move.BALL_7:
            real_action = np.random.choice(
                options,
                1,
                p=[
                    2 * ball_control_k2,
                    2 * ball_control_k2,
                    3 * ball_control_k2,
                    2 * ball_control_k2,
                    3 * ball_control_k2,
                    ball_control_k2,
                    ball_control_k1,
                    3 * ball_control_k2,
                ],
            )
        elif action == Move.BALL_8:
            real_action = np.random.choice(
                options,
                1,
                p=[
                    2 * ball_control_k2,
                    2 * ball_control_k2,
                    2 * ball_control_k2,
                    3 * ball_control_k2,
                    ball_control_k2,
                    3 * ball_control_k2,
                    3 * ball_control_k2,
                    ball_control_k1,
                ],
            )

        return real_action

    def generate_probs(self, agent_action, op_action) -> Tuple[float, float, float]:
        # see Appendix A in the paper
        match_probs = (0.95, 0.85, 0.6)
        mismatch_probs = tuple(0.1 + (ele - 0.5) / 5 for ele in match_probs)
        general_probs = (0.9, 0.7, 0.3)  # when agent action = 0 (strike all)
        if agent_action == Move.STRIKE_ALL:
            if op_action in [Move.STRIKE_1, Move.STRIKE_2, Move.STRIKE_3, Move.STRIKE_4]:
                return general_probs
            else:
                return (0.3, 0.5, 0.2)
        elif agent_action == Move.STRIKE_1:
            if op_action == Move.STRIKE_1:
                return (0.95, 0.85, 0.6)  # this agent is specifically good at doing action 1 (hit location 1)
            elif op_action in [Move.STRIKE_2, Move.STRIKE_3]:
                return (2 * ele for ele in mismatch_probs)
            elif op_action in [Move.STRIKE_4, Move.BALL_5]:
                return mismatch_probs
            else:
                return (0.1, 0.1, 0.1)
        elif agent_action == Move.STRIKE_2:
            if op_action == Move.STRIKE_2:
                return match_probs
            elif op_action in [Move.STRIKE_1, Move.STRIKE_4]:
                return (2 * ele for ele in mismatch_probs)
            elif op_action in [Move.STRIKE_3, Move.BALL_6]:
                return mismatch_probs
            else:
                return (0.1, 0.1, 0.1)
        elif agent_action == Move.STRIKE_3:
            if op_action == Move.STRIKE_3:
                return match_probs
            elif op_action in [Move.STRIKE_1, Move.STRIKE_4]:
                return (2 * ele for ele in mismatch_probs)
            elif op_action in [Move.STRIKE_2, Move.BALL_7]:
                return mismatch_probs
            else:
                return (0.1, 0.1, 0.1)
        elif agent_action == Move.STRIKE_4:
            if op_action == Move.STRIKE_4:
                return match_probs
            elif op_action in [Move.STRIKE_2, Move.STRIKE_3]:
                return (2 * ele for ele in mismatch_probs)
            elif op_action in [Move.STRIKE_1, Move.BALL_8]:
                return mismatch_probs
            else:
                return (0.1, 0.1, 0.1)

        raise ValueError(f'The agent shouldn\'t be doing {agent_action} or the op shouldn\'t be doing {op_action}')

    def generate_result(
        self, swing_prob, hit_prob, score_prob, state_, op_action
    ) -> Tuple[bool, int, List[int], EpisodeResult]:
        # given the u1, v1, w1 probabilities and the state, op_action
        # return the result of this state (hit, out swing&miss stand)
        swing, hit, score, reward, result = False, False, False, 0, None
        swing = True if random.random() < swing_prob else False
        if swing:
            hit = True if random.random() < hit_prob else False
            if hit:
                score = True if random.random() < score_prob else False
                if score:
                    self.hit_count += 1
                    result = EpisodeResult.HIT
                    # episode ends, agent scores
                    done = True
                    reward = self.reward_score
                else:
                    self.hit_out_count += 1
                    result = EpisodeResult.OUT
                    # episode ends, agent out
                    done = True
                    reward = self.reward_out
            else:
                # agent swings & misses, strike + 1
                done = False
                reward = 0
                state_[0] += 1
        else:
            # stand, should further determine whether it is a strike or ball
            done = False
            if op_action in [Move.STRIKE_1, Move.STRIKE_2, Move.STRIKE_3, Move.STRIKE_4]:
                # it's a strike
                state_[0] += 1
            else:
                # it's a ball
                state_[1] += 1

        if state_[0] >= 3:
            # 3 strikes, agent is out
            self.strike_out_count += 1
            result = EpisodeResult.STRIKE_OUT
            done = True
            reward = self.reward_out
        elif state_[1] >= 4:
            # 4 balls, agent gets base on balls (BB)
            result = EpisodeResult.WALK
            self.walk_count += 1
            done = True
            reward = self.reward_score

        return done, reward, state_, result

    def show(self, actions):
        print('------- Current State ---------')
        agent = 'A'
        opponent = 'O'
        # TODO: variable name 'agent_position' should be changed
        if actions[0] == Move.STRIKE_1:
            agent_position = (1, 1)
        elif actions[0] == Move.STRIKE_2:
            agent_position = (2, 1)
        elif actions[0] == Move.STRIKE_3:
            agent_position = (1, 2)
        elif actions[0] == Move.STRIKE_4:
            agent_position = (2, 2)
        elif actions[0] == Move.STRIKE_ALL:
            agent_position = (-1, -1)
        if actions[1] == Move.STRIKE_1:
            op_position = (1, 1)
        elif actions[1] == Move.STRIKE_2:
            op_position = (2, 1)
        elif actions[1] == Move.STRIKE_3:
            op_position = (1, 2)
        elif actions[1] == Move.STRIKE_4:
            op_position = (2, 2)
        elif actions[1] == Move.BALL_5:
            op_position = (0, 0)
        elif actions[1] == Move.BALL_6:
            op_position = (3, 0)
        elif actions[1] == Move.BALL_7:
            op_position = (0, 3)
        elif actions[1] == Move.BALL_8:
            op_position = (3, 3)

        if op_position == agent_position:
            hit_position = agent_position
        else:
            hit_position = (-1, -1)

        for y in range(4):
            for x in range(4):
                if (x, y) == hit_position:
                    print('H', end='')
                elif (x, y) == agent_position:
                    print(agent, end='')
                elif (x, y) == op_position:
                    print(opponent, end='')
                elif agent_position == (-1, -1):
                    print('+', end='')
                else:
                    print('.', end='')
            print()
        print('------------------------')

class Agent:
    def __init__(self, x=None, y=None):
        pass


class Opponent(Agent):
    class Policy(Enum):
        ONE = 1
        TWO = 2
        THREE = 3
        FOUR = 4

    def __init__(self, x=None, y=None):
        super().__init__(x, y)
        self._policy: self.Policy = None

    @property
    def policy(self):
        return self._policy

    @policy.setter
    def policy(self, new_policy: Policy) -> None:
        if not isinstance(new_policy, self.Policy):
            raise ValueError('Policy should be represented by the `Opponent.Policy` class')
        self._policy = new_policy

    def get_action(self, state) -> Move:
        """
        Return the action given its policy and current location.

        Raises:
            ValueError: When the location doesn't meet the policy.

        Returns:
            Move: The action to take.
        """
        if self.policy is self.Policy.ONE:
            # (0, 0) 1
            # (0, 1) 1
            # (0, 2) 1
            # (0, 3) 1
            # (1, 0) 5
            # (1, 1) 5
            # (1, 2) 1
            # (1, 3) 1
            # (2, 0) 5
            # (2, 1) 5
            # (2, 2) 1
            # (2, 3) 1
            if state in [[0, 0], [0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 2], [2, 3]]:
                return self.generate_action_from_tau(tau=1, situation=1)
            elif state in [[1, 0], [1, 1]]:
                return self.generate_action_from_tau(tau=1, situation=2)
            else:
                return self.generate_action_from_tau(tau=1, situation=3)
        elif self.policy is self.Policy.TWO:
            # (0, 0) 2
            # (0, 1) 2
            # (0, 2) 2
            # (0, 3) 2
            # (1, 0) 6
            # (1, 1) 6
            # (1, 2) 2
            # (1, 3) 2
            # (2, 0) 6
            # (2, 1) 6
            # (2, 2) 2
            # (2, 3) 2
            if state in [[0, 0], [0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 2], [2, 3]]:
                return self.generate_action_from_tau(tau=2, situation=1)
            elif state in [[1, 0], [1, 1]]:
                return self.generate_action_from_tau(tau=2, situation=2)
            else:
                return self.generate_action_from_tau(tau=2, situation=3)
        elif self.policy is self.Policy.THREE:
            # (0, 0) 3
            # (0, 1) 3
            # (0, 2) 3
            # (0, 3) 3
            # (1, 0) 7
            # (1, 1) 7
            # (1, 2) 3
            # (1, 3) 3
            # (2, 0) 7
            # (2, 1) 7
            # (2, 2) 3
            # (2, 3) 3
            if state in [[0, 0], [0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 2], [2, 3]]:
                return self.generate_action_from_tau(tau=3, situation=1)
            elif state in [[1, 0], [1, 1]]:
                return self.generate_action_from_tau(tau=3, situation=2)
            else:
                return self.generate_action_from_tau(tau=3, situation=3)
        elif self.policy is self.Policy.FOUR:
            # (0, 0) 4
            # (0, 1) 4
            # (0, 2) 4
            # (0, 3) 4
            # (1, 0) 8
            # (1, 1) 8
            # (1, 2) 4
            # (1, 3) 4
            # (2, 0) 8
            # (2, 1) 8
            # (2, 2) 4
            # (2, 3) 4
            if state in [[0, 0], [0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 2], [2, 3]]:
                return self.generate_action_from_tau(tau=4, situation=1)
            elif state in [[1, 0], [1, 1]]:
                return self.generate_action_from_tau(tau=4, situation=2)
            else:
                return self.generate_action_from_tau(tau=4, situation=3)
        raise ValueError(f'Opponent with policy {self.policy}')

    def generate_action_from_tau(
        self, tau: int, situation: int
    ) -> Move:  # tau=1~4, situation=1~3: states are catogorized into 3 situations
        action_control_constant_mu1 = 0.6  # mu1: probability of the action that is most likely to happen
        mu2 = (1 - action_control_constant_mu1) / 16  # define the probability of other less likely actions
        options = [
            Move.STRIKE_1,
            Move.STRIKE_2,
            Move.STRIKE_3,
            Move.STRIKE_4,
            Move.BALL_5,
            Move.BALL_6,
            Move.BALL_7,
            Move.BALL_8,
        ]

        # [0, 0], [0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 2], [2, 3]
        if situation == 1:
            if tau == 1:
                action = np.random.choice(
                    options, 1, p=[action_control_constant_mu1, 4 * mu2, 4 * mu2, 4 * mu2, mu2, mu2, mu2, mu2]
                )
            elif tau == 2:
                action = np.random.choice(
                    options, 1, p=[4 * mu2, action_control_constant_mu1, 4 * mu2, 4 * mu2, mu2, mu2, mu2, mu2]
                )
            elif tau == 3:
                action = np.random.choice(
                    options, 1, p=[4 * mu2, 4 * mu2, action_control_constant_mu1, 4 * mu2, mu2, mu2, mu2, mu2]
                )
            elif tau == 4:
                action = np.random.choice(
                    options, 1, p=[4 * mu2, 4 * mu2, 4 * mu2, action_control_constant_mu1, mu2, mu2, mu2, mu2]
                )
        # [1, 0], [1, 1]
        elif situation == 2:
            if tau == 1:
                action = np.random.choice(
                    options,
                    1,
                    p=[4 * mu2, 1 * mu2, 1 * mu2, 1 * mu2, action_control_constant_mu1, 4 * mu2, 4 * mu2, 1 * mu2],
                )
            elif tau == 2:
                action = np.random.choice(
                    options,
                    1,
                    p=[1 * mu2, 4 * mu2, 1 * mu2, 1 * mu2, 4 * mu2, action_control_constant_mu1, 1 * mu2, 4 * mu2],
                )
            elif tau == 3:
                action = np.random.choice(
                    options,
                    1,
                    p=[1 * mu2, 1 * mu2, 4 * mu2, 1 * mu2, 4 * mu2, 1 * mu2, action_control_constant_mu1, 4 * mu2],
                )
            elif tau == 4:
                action = np.random.choice(
                    options,
                    1,
                    p=[1 * mu2, 1 * mu2, 1 * mu2, 4 * mu2, 1 * mu2, 4 * mu2, 4 * mu2, action_control_constant_mu1],
                )
        # [2, 0], [2, 1]
        elif situation == 3:
            if tau == 1:
                action = np.random.choice(
                    options, 1, p=[mu2, mu2, mu2, mu2, action_control_constant_mu1, 4 * mu2, 4 * mu2, 4 * mu2]
                )
            elif tau == 2:
                action = np.random.choice(
                    options, 1, p=[mu2, mu2, mu2, mu2, 4 * mu2, action_control_constant_mu1, 4 * mu2, 4 * mu2]
                )
            elif tau == 3:
                action = np.random.choice(
                    options, 1, p=[mu2, mu2, mu2, mu2, 4 * mu2, 4 * mu2, action_control_constant_mu1, 4 * mu2]
                )
            elif tau == 4:
                action = np.random.choice(
                    options, 1, p=[mu2, mu2, mu2, mu2, 4 * mu2, 4 * mu2, 4 * mu2, action_control_constant_mu1]
                )

        return action[0]


class BprOpponent(Opponent):
    def __init__(self, x=None, y=None):
        super().__init__(x, y)

        self.n_policies = len(self.Policy)
        self._belief = np.ones(self.n_policies) / self.n_policies  # initial as uniform distribution

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
        Update the belief according to the utility the agent gets.
        Notice that the utility is from the agent's perspective.

        Args:
            utility (int): The reward the agent gets in a episode.
        """
        # find the currently observed utility in the performance model
        likelihood = (self.PERFORMANCE_MODEL[self.policy.value - 1] == utility).astype(float)
        # posterior (belief) = prior * likelihood (performance model)
        new_belief_unnormalized = self.belief * likelihood / (np.sum(likelihood * self.belief) + 1e-6)
        self.belief = normalize_distribution(new_belief_unnormalized, 0.01)

    def update_policy(self) -> None:
        belief_mul_performance = self.belief @ np.transpose(self.PERFORMANCE_MODEL)
        candidates = np.argwhere(belief_mul_performance == np.amin(belief_mul_performance)).flatten().tolist()
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
        ELEVEN = 11
        TWELVE = 12

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
                f'Phi should be represented by `PhiOpponent.Phi` class, invalid value: {new_phi} ({type(new_phi)})'
            )
        self._phi = new_phi

    def update_policy(self, final_action, final_result) -> None:
        """
        Update the policy of the opponent according to phi and sigma (terminal state).

        Args:
            ternimal_state (Tuple[Location, Location]): Opponent (x, y) + Agent (x, y)
        """
        if not self.phi:
            raise RuntimeError('Phi is not specified.')

        # phi 1 to phi 4 always keep the same policy
        if self.phi is self.Phi.ONE:
            self.policy = self.Policy.ONE
        elif self.phi is self.Phi.TWO:
            self.policy = self.Policy.TWO
        elif self.phi is self.Phi.THREE:
            self.policy = self.Policy.THREE
        elif self.phi is self.Phi.FOUR:
            self.policy = self.Policy.FOUR
        # phi 5 randomly chooses its policy
        elif self.phi is self.Phi.FIVE:
            self.policy = random.choice(list(self.Policy))
        # phi 6
        elif self.phi is self.Phi.SIX:
            # if (3478 hit/walk) or (1256 out/strike_out) or (78 walk)
            if (
                (
                    final_action in [Move.STRIKE_3, Move.STRIKE_4, Move.BALL_7, Move.BALL_8]
                    and final_result in [EpisodeResult.HIT, EpisodeResult.WALK]
                )
                or (
                    final_action in [Move.STRIKE_1, Move.STRIKE_2, Move.BALL_5, Move.BALL_6]
                    and final_result in [EpisodeResult.OUT, EpisodeResult.STRIKE_OUT]
                )
                or (final_action in [Move.BALL_7, Move.BALL_8] and final_result is EpisodeResult.WALK)
            ):
                self.policy = self.Policy.ONE
            else:
                self.policy = self.Policy.THREE
        # phi 7
        elif self.phi is self.Phi.SEVEN:
            # if (3478 hit/walk) or (1256 out/strike_out) or (78 walk)
            if (
                (
                    final_action in [Move.STRIKE_3, Move.STRIKE_4, Move.BALL_7, Move.BALL_8]
                    and final_result in [EpisodeResult.HIT, EpisodeResult.WALK]
                )
                or (
                    final_action in [Move.STRIKE_1, Move.STRIKE_2, Move.BALL_5, Move.BALL_6]
                    and final_result in [EpisodeResult.OUT, EpisodeResult.STRIKE_OUT]
                )
                or (final_action in [Move.BALL_7, Move.BALL_8] and final_result is EpisodeResult.WALK)
            ):
                self.policy = self.Policy.FOUR
            else:
                self.policy = self.Policy.TWO
        # phi 8
        elif self.phi is self.Phi.EIGHT:
            # if (1256 hit) or (1256 stirke out)
            if (
                final_action in [Move.STRIKE_1, Move.STRIKE_2, Move.BALL_5, Move.BALL_6]
                and final_result is EpisodeResult.HIT
            ) or (
                final_action in [Move.STRIKE_3, Move.STRIKE_4, Move.BALL_7, Move.BALL_8]
                and final_result is EpisodeResult.OUT
            ):
                self.policy = self.Policy.ONE
            else:
                self.policy = self.Policy.THREE
        # phi 9
        elif self.phi is self.Phi.NINE:
            # if hit/walk
            if final_result in [EpisodeResult.HIT, EpisodeResult.WALK]:
                self.policy = self.Policy.FOUR
            else:
                self.policy = self.Policy.THREE
        # phi 10 resembles phi6
        elif self.phi is self.Phi.TEN:

            if final_action in [Move.STRIKE_1, Move.STRIKE_2, Move.BALL_5, Move.BALL_6] and final_result in [
                EpisodeResult.HIT
            ]:
                self.policy = self.Policy.THREE
            elif final_action in [Move.STRIKE_3, Move.STRIKE_4, Move.BALL_7, Move.BALL_8] and final_result in [
                EpisodeResult.HIT
            ]:
                self.policy = self.Policy.ONE
            elif final_action in [Move.STRIKE_1, Move.STRIKE_2, Move.BALL_5, Move.BALL_6] and final_result in [
                EpisodeResult.OUT
            ]:
                self.policy = self.Policy.FOUR
            elif final_action in [Move.STRIKE_3, Move.STRIKE_4, Move.BALL_7, Move.BALL_8] and final_result in [
                EpisodeResult.OUT
            ]:
                self.policy = self.Policy.THREE
            elif final_action in [Move.STRIKE_1, Move.STRIKE_2, Move.BALL_5, Move.BALL_6] and final_result in [
                EpisodeResult.STRIKE_OUT
            ]:
                self.policy = self.Policy.ONE
            elif final_action in [Move.STRIKE_3, Move.STRIKE_4, Move.BALL_7, Move.BALL_8] and final_result in [
                EpisodeResult.STRIKE_OUT
            ]:
                self.policy = self.Policy.THREE
            elif final_action in [Move.BALL_5, Move.BALL_6] and final_result in [EpisodeResult.WALK]:
                self.policy = self.Policy.FOUR
            elif final_action in [Move.BALL_7, Move.BALL_8] and final_result in [EpisodeResult.WALK]:
                self.policy = self.Policy.ONE

        # phi 11 resembles phi5
        elif self.phi is self.Phi.ELEVEN:
            if final_action in [Move.STRIKE_1, Move.STRIKE_2, Move.BALL_5, Move.BALL_6] and final_result in [
                EpisodeResult.HIT
            ]:
                self.policy = self.Policy.THREE
            elif final_action in [Move.STRIKE_3, Move.STRIKE_4, Move.BALL_7, Move.BALL_8] and final_result in [
                EpisodeResult.HIT
            ]:
                self.policy = self.Policy.ONE
            elif final_action in [Move.BALL_5, Move.BALL_6] and final_result in [EpisodeResult.WALK]:
                self.policy = self.Policy.FOUR
            elif final_action in [Move.BALL_7, Move.BALL_8] and final_result in [EpisodeResult.WALK]:
                self.policy = self.Policy.TWO
            else:
                self.policy = random.choice(list(self.Policy))


class NewPhiNoiseOpponent(Opponent):
    def __init__(self, p_pattern: float, x=None, y=None):
        super().__init__(x, y)
        self.p_pattern = p_pattern
        self._phi = random.choice(list(PhiOpponent.Phi)[5:9])
        # print(f'opponent phi is now {self.phi}')

    @property
    def phi(self):
        return self._phi

    @phi.setter
    def phi(self, new_phi):
        if not isinstance(new_phi, self.Phi):
            raise ValueError(
                f'Phi should be represented by `PhiOpponent.Phi` class, invalid value: {new_phi} ({type(new_phi)})'
            )
        self._phi = new_phi

    def update_policy(self, final_action, final_result) -> None:
        """
        Update the policy of the opponent according to phi and sigma (terminal state).

        Args:
            ternimal_state (Tuple[Location, Location]): Opponent (x, y) + Agent (x, y)
        """
        if not self.phi:
            raise RuntimeError('Phi is not specified.')

        stochasticOp = random.random() < self.p_pattern
        if stochasticOp:
            self.policy = random.choice(list(self.Policy))
        else:
            # phi 1 to phi 4 always keep the same policy
            if self.phi is PhiOpponent.Phi.ONE:
                self.policy = self.Policy.ONE
            elif self.phi is PhiOpponent.Phi.TWO:
                self.policy = self.Policy.TWO
            elif self.phi is PhiOpponent.Phi.THREE:
                self.policy = self.Policy.THREE
            elif self.phi is PhiOpponent.Phi.FOUR:
                self.policy = self.Policy.FOUR
            # phi 5 randomly chooses its policy
            elif self.phi is PhiOpponent.Phi.FIVE:
                self.policy = random.choice(list(self.Policy))
            # phi 6
            elif self.phi is PhiOpponent.Phi.SIX:
                # if (3478 hit/walk) or (1256 out/strike_out) or (78 walk)
                if (
                    (
                        final_action in [Move.STRIKE_3, Move.STRIKE_4, Move.BALL_7, Move.BALL_8]
                        and final_result in [EpisodeResult.HIT, EpisodeResult.WALK]
                    )
                    or (
                        final_action in [Move.STRIKE_1, Move.STRIKE_2, Move.BALL_5, Move.BALL_6]
                        and final_result in [EpisodeResult.OUT, EpisodeResult.STRIKE_OUT]
                    )
                    or (final_action in [Move.BALL_7, Move.BALL_8] and final_result is EpisodeResult.WALK)
                ):
                    self.policy = self.Policy.ONE
                else:
                    self.policy = self.Policy.THREE
            # phi 7
            elif self.phi is PhiOpponent.Phi.SEVEN:
                # if (3478 hit/walk) or (1256 out/strike_out) or (78 walk)
                if (
                    (
                        final_action in [Move.STRIKE_3, Move.STRIKE_4, Move.BALL_7, Move.BALL_8]
                        and final_result in [EpisodeResult.HIT, EpisodeResult.WALK]
                    )
                    or (
                        final_action in [Move.STRIKE_1, Move.STRIKE_2, Move.BALL_5, Move.BALL_6]
                        and final_result in [EpisodeResult.OUT, EpisodeResult.STRIKE_OUT]
                    )
                    or (final_action in [Move.BALL_7, Move.BALL_8] and final_result is EpisodeResult.WALK)
                ):
                    self.policy = self.Policy.FOUR
                else:
                    self.policy = self.Policy.TWO
            # phi 8
            elif self.phi is PhiOpponent.Phi.EIGHT:
                # if (3478 hit) or (1256 stirke out)
                if (
                    final_action in [Move.STRIKE_1, Move.STRIKE_2, Move.BALL_5, Move.BALL_6]
                    and final_result is EpisodeResult.HIT
                ) or (
                    final_action in [Move.STRIKE_3, Move.STRIKE_4, Move.BALL_7, Move.BALL_8]
                    and final_result is EpisodeResult.OUT
                ):
                    self.policy = self.Policy.ONE
                else:
                    self.policy = self.Policy.THREE
            # phi 9
            elif self.phi is PhiOpponent.Phi.NINE:
                # if hit/walk
                if final_result in [EpisodeResult.HIT, EpisodeResult.WALK]:
                    self.policy = self.Policy.FOUR
                else:
                    self.policy = self.Policy.THREE
            # phi 10 resembles phi6
            elif self.phi is PhiOpponent.Phi.TEN:
                # if (12 hit) or (56 hit/walk)
                if (final_action in [Move.STRIKE_1, Move.STRIKE_2] and final_result is EpisodeResult.HIT) or (
                    final_action in [Move.BALL_5, Move.BALL_6]
                    and final_result in [EpisodeResult.HIT, EpisodeResult.WALK]
                ):
                    self.policy = self.Policy.FOUR
                # if (3478 out/strike out)
                elif final_action in [Move.STRIKE_3, Move.STRIKE_4, Move.BALL_7, Move.BALL_8] and final_result in [
                    EpisodeResult.OUT,
                    EpisodeResult.STRIKE_OUT,
                ]:
                    self.policy = self.Policy.THREE
                else:
                    self.policy = self.Policy.ONE
            # phi 11 resembles phi7
            elif self.phi is PhiOpponent.Phi.ELEVEN:
                # if (12 hit) or (56 hit/walk)
                if (final_action in [Move.STRIKE_1, Move.STRIKE_2] and final_result is EpisodeResult.HIT) or (
                    final_action in [Move.BALL_5, Move.BALL_6]
                    and final_result in [EpisodeResult.HIT, EpisodeResult.WALK]
                ):
                    self.policy = self.Policy.ONE
                # if (3478 out/strike out)
                elif final_action in [Move.STRIKE_3, Move.STRIKE_4, Move.BALL_7, Move.BALL_8] and final_result in [
                    EpisodeResult.OUT,
                    EpisodeResult.STRIKE_OUT,
                ]:
                    self.policy = self.Policy.TWO
                else:
                    self.policy = self.Policy.FOUR
            # phi 12 resembles phi5
            elif self.phi is PhiOpponent.Phi.TWELVE:
                if final_action in [Move.STRIKE_1, Move.STRIKE_2, Move.BALL_5, Move.BALL_6] and final_result in [
                    EpisodeResult.HIT,
                    EpisodeResult.WALK,
                ]:
                    self.policy = self.Policy.THREE
                elif final_action in [Move.STRIKE_3, Move.STRIKE_4, Move.BALL_7, Move.BALL_8] and final_result in [
                    EpisodeResult.HIT,
                    EpisodeResult.WALK,
                ]:
                    self.policy = self.Policy.ONE
                else:
                    self.policy = random.choice(list(self.Policy))


class OutOfLibraryPhiOpponent(Opponent):
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
        ELEVEN = 11
        TWELVE = 12

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
                f'Phi should be represented by `PhiOpponent.Phi` class, invalid value: {new_phi} ({type(new_phi)})'
            )
        self._phi = new_phi

    def update_policy(self, final_action, final_result, N) -> None:
        """
        Update the policy of the opponent according to phi and sigma (terminal state).

        Args:
            ternimal_state (Tuple[Location, Location]): Opponent (x, y) + Agent (x, y)
        """
        if not self.phi:
            raise RuntimeError('Phi is not specified.')

        policy_list = []
        if self.phi is self.Phi.ONE:
            policy_list = [
                self.Policy.ONE,
                self.Policy.ONE,
                self.Policy.ONE,
                self.Policy.ONE,
                self.Policy.ONE,
                self.Policy.ONE,
                self.Policy.ONE,
                self.Policy.ONE,
            ]
        elif self.phi is self.Phi.TWO:
            policy_list = [
                self.Policy.TWO,
                self.Policy.TWO,
                self.Policy.TWO,
                self.Policy.TWO,
                self.Policy.TWO,
                self.Policy.TWO,
                self.Policy.TWO,
                self.Policy.TWO,
            ]
        elif self.phi is self.Phi.THREE:
            policy_list = [
                self.Policy.THREE,
                self.Policy.THREE,
                self.Policy.THREE,
                self.Policy.THREE,
                self.Policy.THREE,
                self.Policy.THREE,
                self.Policy.THREE,
                self.Policy.THREE,
            ]
        elif self.phi is self.Phi.FOUR:
            policy_list = [
                self.Policy.FOUR,
                self.Policy.FOUR,
                self.Policy.FOUR,
                self.Policy.FOUR,
                self.Policy.FOUR,
                self.Policy.FOUR,
                self.Policy.FOUR,
                self.Policy.FOUR,
            ]
        elif self.phi is self.Phi.FIVE:
            for i in range(8):
                policy_list[i] = random.choice(list(self.Policy))
        elif self.phi is self.Phi.SIX:
            policy_list = [
                self.Policy.THREE,
                self.Policy.ONE,
                self.Policy.ONE,
                self.Policy.THREE,
                self.Policy.ONE,
                self.Policy.THREE,
                self.Policy.THREE,
                self.Policy.ONE,
            ]
        elif self.phi is self.Phi.SEVEN:
            policy_list = [
                self.Policy.TWO,
                self.Policy.FOUR,
                self.Policy.FOUR,
                self.Policy.TWO,
                self.Policy.FOUR,
                self.Policy.TWO,
                self.Policy.TWO,
                self.Policy.FOUR,
            ]
        elif self.phi is self.Phi.EIGHT:
            policy_list = [
                self.Policy.THREE,
                self.Policy.ONE,
                self.Policy.THREE,
                self.Policy.THREE,
                self.Policy.ONE,
                self.Policy.THREE,
                self.Policy.THREE,
                self.Policy.THREE,
            ]
        elif self.phi is self.Phi.NINE:
            policy_list = [
                self.Policy.FOUR,
                self.Policy.FOUR,
                self.Policy.THREE,
                self.Policy.THREE,
                self.Policy.THREE,
                self.Policy.THREE,
                self.Policy.FOUR,
                self.Policy.FOUR,
            ]

        # TODO: random replace elements in policy_list according to N
        index_to_replace = random.sample(range(8), N)
        for i in range(N):
            policy_list[index_to_replace[i]] = random.choice(list(self.Policy))
        # TODO: check the new policy_list is has at least N different elements from policy_lists of all other phi opponent

        # 1256 hit
        if final_action in [Move.STRIKE_1, Move.STRIKE_2, Move.BALL_5, Move.BALL_6] and final_result in [
            EpisodeResult.HIT
        ]:
            self.policy = policy_list[0]
        # 3478 hit
        elif final_action in [Move.STRIKE_3, Move.STRIKE_4, Move.BALL_7, Move.BALL_8] and final_result in [
            EpisodeResult.HIT
        ]:
            self.policy = policy_list[1]
        # 1256 out
        elif final_action in [Move.STRIKE_1, Move.STRIKE_2, Move.BALL_5, Move.BALL_6] and final_result in [
            EpisodeResult.OUT
        ]:
            self.policy = policy_list[2]
        # 3478 out
        elif final_action in [Move.STRIKE_3, Move.STRIKE_4, Move.BALL_7, Move.BALL_8] and final_result in [
            EpisodeResult.OUT
        ]:
            self.policy = policy_list[3]
        # 1256 strike out
        elif final_action in [Move.STRIKE_1, Move.STRIKE_2, Move.BALL_5, Move.BALL_6] and final_result in [
            EpisodeResult.STRIKE_OUT
        ]:
            self.policy = policy_list[4]
        # 3478 strike out
        elif final_action in [Move.STRIKE_3, Move.STRIKE_4, Move.BALL_7, Move.BALL_8] and final_result in [
            EpisodeResult.STRIKE_OUT
        ]:
            self.policy = policy_list[5]
        # 56 walk
        elif final_action in [Move.BALL_5, Move.BALL_6] and final_result in [EpisodeResult.WALK]:
            self.policy = policy_list[6]
        # 78 walk
        elif final_action in [Move.BALL_7, Move.BALL_8] and final_result in [EpisodeResult.WALK]:
            self.policy = policy_list[7]
