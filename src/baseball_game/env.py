import random
from enum import Enum
from typing import List, Optional, Tuple

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

    def step(
        self, agent_action: Move
    ) -> Tuple[bool, int, State, Tuple[Move, Move], Optional[EpisodeResult]]:
        # return done, reward, next_state, [agent_action, opponent_action]
        if not isinstance(agent_action, Move):
            raise ValueError('Action should be represented by the `Move` class')

        self.steps += 1
        opponent_action = self.opponent.get_action(self.state)
        swing_prob, hit_prob, score_prob = self.generate_probs(agent_action, opponent_action)

        done, reward, state_, result = self.generate_result(
            swing_prob, hit_prob, score_prob, self.state.copy(), opponent_action
        )
        actions = (agent_action, opponent_action)
        return done, reward, state_, actions, result

    def generate_probs(self, agent_action, op_action) -> Tuple[float, float, float]:
        # see Appendix A in the paper
        match_probs = (0.95, 0.85, 0.6)
        mismatch_probs = tuple(0.1 + (elem - 0.5) / 5 for elem in match_probs)
        general_probs = (0.9, 0.7, 0.3)  # when agent action = 0 (strike all)
        if agent_action == Move.STRIKE_ALL:
            if op_action in [Move.STRIKE_1, Move.STRIKE_2, Move.STRIKE_3, Move.STRIKE_4]:
                return general_probs
            else:
                return (0.3, 0.5, 0.2)
        elif agent_action == Move.STRIKE_1:
            if op_action == Move.STRIKE_1:
                return (
                    0.95,
                    0.85,
                    0.6,
                )  # this agent is specifically good at doing action 1 (hit location 1)
            elif op_action in [Move.STRIKE_2, Move.STRIKE_3]:
                return tuple(2 * ele for ele in mismatch_probs)  # type: ignore
            elif op_action in [Move.STRIKE_4, Move.BALL_5]:
                return mismatch_probs  # type: ignore
            else:
                return (0.1, 0.1, 0.1)
        elif agent_action == Move.STRIKE_2:
            if op_action == Move.STRIKE_2:
                return match_probs
            elif op_action in [Move.STRIKE_1, Move.STRIKE_4]:
                return tuple(2 * ele for ele in mismatch_probs)  # type: ignore
            elif op_action in [Move.STRIKE_3, Move.BALL_6]:
                return mismatch_probs  # type: ignore
            else:
                return (0.1, 0.1, 0.1)
        elif agent_action == Move.STRIKE_3:
            if op_action == Move.STRIKE_3:
                return match_probs
            elif op_action in [Move.STRIKE_1, Move.STRIKE_4]:
                return tuple(2 * ele for ele in mismatch_probs)  # type: ignore
            elif op_action in [Move.STRIKE_2, Move.BALL_7]:
                return mismatch_probs  # type: ignore
            else:
                return (0.1, 0.1, 0.1)
        elif agent_action == Move.STRIKE_4:
            if op_action == Move.STRIKE_4:
                return match_probs
            elif op_action in [Move.STRIKE_2, Move.STRIKE_3]:
                return tuple(2 * ele for ele in mismatch_probs)  # type: ignore
            elif op_action in [Move.STRIKE_1, Move.BALL_8]:
                return mismatch_probs  # type: ignore
            else:
                return (0.1, 0.1, 0.1)

        raise ValueError(
            f'The agent shouldn\'t be doing {agent_action} or '
            f'the opponent shouldn\'t be doing {op_action}'
        )

    def generate_result(
        self, swing_prob, hit_prob, score_prob, state_, op_action
    ) -> Tuple[bool, int, List[int], Optional[EpisodeResult]]:
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
        situation1 = ([0, 0], [0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 2], [2, 3])
        situation2 = ([1, 0], [1, 1])
        situation3 = ([2, 0], [2, 1])

        if state in situation1:
            return self.generate_action_from_tau(tau=self.policy.value, situation=1)
        elif state in situation2:
            return self.generate_action_from_tau(tau=self.policy.value, situation=2)
        elif state in situation3:
            return self.generate_action_from_tau(tau=self.policy.value, situation=3)

        raise ValueError(f'Opponent with policy {self.policy}')

    def generate_action_from_tau(self, tau: int, situation: int) -> Move:
        """
        Return the action given tau and situation.

        Args:
            tau (int): Opponent policy, can be 1, 2, 3, or 4.
            situation (int): States are catogorized into 3 situations, can be 1, 2, or 3.

        Returns:
            Move: The action to take.
        """
        # mu1: probability of the action that is most likely to happen
        action_control_constant_mu1 = 0.6
        # define the probability of other less likely actions
        mu2 = (1 - action_control_constant_mu1) / 16
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
                    options,
                    1,
                    p=[action_control_constant_mu1, 4 * mu2, 4 * mu2, 4 * mu2, mu2, mu2, mu2, mu2],
                )
            elif tau == 2:
                action = np.random.choice(
                    options,
                    1,
                    p=[4 * mu2, action_control_constant_mu1, 4 * mu2, 4 * mu2, mu2, mu2, mu2, mu2],
                )
            elif tau == 3:
                action = np.random.choice(
                    options,
                    1,
                    p=[4 * mu2, 4 * mu2, action_control_constant_mu1, 4 * mu2, mu2, mu2, mu2, mu2],
                )
            elif tau == 4:
                action = np.random.choice(
                    options,
                    1,
                    p=[4 * mu2, 4 * mu2, 4 * mu2, action_control_constant_mu1, mu2, mu2, mu2, mu2],
                )
        # [1, 0], [1, 1]
        elif situation == 2:
            if tau == 1:
                action = np.random.choice(
                    options,
                    1,
                    p=[
                        4 * mu2,
                        1 * mu2,
                        1 * mu2,
                        1 * mu2,
                        action_control_constant_mu1,
                        4 * mu2,
                        4 * mu2,
                        1 * mu2,
                    ],
                )
            elif tau == 2:
                action = np.random.choice(
                    options,
                    1,
                    p=[
                        1 * mu2,
                        4 * mu2,
                        1 * mu2,
                        1 * mu2,
                        4 * mu2,
                        action_control_constant_mu1,
                        1 * mu2,
                        4 * mu2,
                    ],
                )
            elif tau == 3:
                action = np.random.choice(
                    options,
                    1,
                    p=[
                        1 * mu2,
                        1 * mu2,
                        4 * mu2,
                        1 * mu2,
                        4 * mu2,
                        1 * mu2,
                        action_control_constant_mu1,
                        4 * mu2,
                    ],
                )
            elif tau == 4:
                action = np.random.choice(
                    options,
                    1,
                    p=[
                        1 * mu2,
                        1 * mu2,
                        1 * mu2,
                        4 * mu2,
                        1 * mu2,
                        4 * mu2,
                        4 * mu2,
                        action_control_constant_mu1,
                    ],
                )
        # [2, 0], [2, 1]
        elif situation == 3:
            if tau == 1:
                action = np.random.choice(
                    options,
                    1,
                    p=[mu2, mu2, mu2, mu2, action_control_constant_mu1, 4 * mu2, 4 * mu2, 4 * mu2],
                )
            elif tau == 2:
                action = np.random.choice(
                    options,
                    1,
                    p=[mu2, mu2, mu2, mu2, 4 * mu2, action_control_constant_mu1, 4 * mu2, 4 * mu2],
                )
            elif tau == 3:
                action = np.random.choice(
                    options,
                    1,
                    p=[mu2, mu2, mu2, mu2, 4 * mu2, 4 * mu2, action_control_constant_mu1, 4 * mu2],
                )
            elif tau == 4:
                action = np.random.choice(
                    options,
                    1,
                    p=[mu2, mu2, mu2, mu2, 4 * mu2, 4 * mu2, 4 * mu2, action_control_constant_mu1],
                )

        return action[0]


class BprOpponent(Opponent):
    def __init__(self, x=None, y=None):
        super().__init__(x, y)

        self.n_policies = len(self.Policy)
        self._belief = np.ones(self.n_policies) / self.n_policies  # initial as uniform distribution

        # TODO: need to add check for the performance model (#42)
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
        Update the belief according to the utility the agent gets.
        Notice that the utility is from the agent's perspective.

        Args:
            utility (int): The reward the agent gets in a episode.
        """
        # find the currently observed utility in the performance model
        likelihood = (self.performance_model[self.policy.value - 1] == utility).astype(float)
        # posterior (belief) = prior * likelihood (performance model)
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
        ELEVEN = 11

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
                f'Phi should be represented by `PhiOpponent.Phi` class, '
                f'invalid value: {new_phi} ({type(new_phi)})'
            )
        self._phi = new_phi

    def update_policy(self, final_action, final_result) -> None:
        """
        Update the policy of the opponent according to phi and sigma (terminal state).
        See Table IV in the paper for details.
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
            # if (3478 hit) or (1256 out/strike_out) or (78 walk)
            if (
                (
                    final_action in (Move.STRIKE_3, Move.STRIKE_4, Move.BALL_7, Move.BALL_8)
                    and final_result is EpisodeResult.HIT
                )
                or (
                    final_action in (Move.STRIKE_1, Move.STRIKE_2, Move.BALL_5, Move.BALL_6)
                    and final_result in (EpisodeResult.OUT, EpisodeResult.STRIKE_OUT)
                )
                or (
                    final_action in (Move.BALL_7, Move.BALL_8)
                    and final_result is EpisodeResult.WALK
                )
            ):
                self.policy = self.Policy.ONE
            else:
                self.policy = self.Policy.THREE
        # phi 7
        elif self.phi is self.Phi.SEVEN:
            # if (3478 hit) or (1256 out/strike_out) or (78 walk)
            if (
                (
                    final_action in (Move.STRIKE_3, Move.STRIKE_4, Move.BALL_7, Move.BALL_8)
                    and final_result is EpisodeResult.HIT
                )
                or (
                    final_action in (Move.STRIKE_1, Move.STRIKE_2, Move.BALL_5, Move.BALL_6)
                    and final_result in (EpisodeResult.OUT, EpisodeResult.STRIKE_OUT)
                )
                or (
                    final_action in (Move.BALL_7, Move.BALL_8)
                    and final_result is EpisodeResult.WALK
                )
            ):
                self.policy = self.Policy.FOUR
            else:
                self.policy = self.Policy.TWO
        # phi 8
        elif self.phi is self.Phi.EIGHT:
            # if (1256 hit) or (3478 out)
            if (
                final_action in (Move.STRIKE_1, Move.STRIKE_2, Move.BALL_5, Move.BALL_6)
                and final_result is EpisodeResult.HIT
            ) or (
                final_action in (Move.STRIKE_3, Move.STRIKE_4, Move.BALL_7, Move.BALL_8)
                and final_result is EpisodeResult.OUT
            ):
                self.policy = self.Policy.ONE
            else:
                self.policy = self.Policy.THREE
        # phi 9
        elif self.phi is self.Phi.NINE:
            # if hit/walk
            if final_result in (EpisodeResult.HIT, EpisodeResult.WALK):
                self.policy = self.Policy.FOUR
            else:
                self.policy = self.Policy.THREE
        # phi 10 resembles phi 6
        elif self.phi is self.Phi.TEN:
            if (
                # sigma_p1
                final_action in (Move.STRIKE_1, Move.STRIKE_2, Move.BALL_5, Move.BALL_6)
                and final_result is EpisodeResult.HIT
            ) or (
                # sigma_p4 and sigma_p6
                final_action in (Move.STRIKE_3, Move.STRIKE_4, Move.BALL_7, Move.BALL_8)
                and final_result in (EpisodeResult.OUT, EpisodeResult.STRIKE_OUT)
            ):
                self.policy = self.Policy.THREE
            elif (
                (
                    # sigma_p2
                    final_action in (Move.STRIKE_3, Move.STRIKE_4, Move.BALL_7, Move.BALL_8)
                    and final_result is EpisodeResult.HIT
                )
                or (
                    # sigma_p5
                    final_action in (Move.STRIKE_1, Move.STRIKE_2, Move.BALL_5, Move.BALL_6)
                    and final_result is EpisodeResult.STRIKE_OUT
                )
                or (
                    # sigma_p8
                    final_action in (Move.BALL_7, Move.BALL_8)
                    and final_result is EpisodeResult.WALK
                )
            ):
                self.policy = self.Policy.ONE
            elif (
                # sigma_p3
                final_action in (Move.STRIKE_1, Move.STRIKE_2, Move.BALL_5, Move.BALL_6)
                and final_result is EpisodeResult.OUT
            ) or (
                # sigma_p7
                final_action in (Move.BALL_5, Move.BALL_6)
                and final_result is EpisodeResult.WALK
            ):
                self.policy = self.Policy.FOUR

        # phi 11 resembles phi 5
        elif self.phi is self.Phi.ELEVEN:
            if (
                # sigma_p1
                final_action in (Move.STRIKE_1, Move.STRIKE_2, Move.BALL_5, Move.BALL_6)
                and final_result is EpisodeResult.HIT
            ):
                self.policy = self.Policy.THREE
            elif (
                # sigma_p2
                final_action in (Move.STRIKE_3, Move.STRIKE_4, Move.BALL_7, Move.BALL_8)
                and final_result is EpisodeResult.HIT
            ):
                self.policy = self.Policy.ONE
            elif (
                # sigma_p7
                final_action in (Move.BALL_5, Move.BALL_6)
                and final_result is EpisodeResult.WALK
            ):
                self.policy = self.Policy.FOUR
            elif (
                # sigma_p8
                final_action in (Move.BALL_7, Move.BALL_8)
                and final_result is EpisodeResult.WALK
            ):
                self.policy = self.Policy.TWO
            else:
                self.policy = random.choice(list(self.Policy))


class NewPhiNoiseOpponent(PhiOpponent):
    def __init__(self, p_pattern: float, x=None, y=None):
        super().__init__(x, y)
        self.p_pattern = p_pattern
        # randomly choose phi from phi 6 to phi 9
        self._phi = random.choice(list(PhiOpponent.Phi)[5:9])

    def update_policy(self, final_action, final_result) -> None:
        """
        Update the policy of the opponent according to phi and sigma (terminal state).
        """
        if not self.phi:
            raise RuntimeError('Phi is not specified.')

        if random.random() > self.p_pattern:
            self.policy = random.choice(list(self.Policy))
        else:
            super().update_policy(final_action, final_result)
