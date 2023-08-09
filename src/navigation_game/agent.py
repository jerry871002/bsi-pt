import random
from enum import Enum
from typing import Tuple

import numpy as np

from utils import normalize_distribution

from .env import Agent, Location, Move, NavigationGame, PhiOpponent


class BprAgent(Agent):
    class Policy(Enum):
        ONE = 1
        TWO = 2
        THREE = 3
        FOUR = 4
        FIVE = 5

    def __init__(self, x=None, y=None):
        super().__init__(x, y)
        self.n_policies = len(self.Policy)
        self._policy = None
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

    @property
    def policy(self):
        return self._policy

    @policy.setter
    def policy(self, new_policy):
        if not isinstance(new_policy, self.Policy):
            raise ValueError('Policy should be represented by the `BprAgent.Policy` class')
        self._policy = new_policy

    def get_action(self) -> Move:
        x, y = self.get_xy()

        if self.get_xy() in ((0, 4), (2, 4), (7, 4), (12, 4), (14, 4)):
            return Move.STANDBY

        if self._policy is self.Policy.ONE:
            # ==== first possibility ====
            #     * * * * * s
            #     *
            #     *
            #     * * * * * *
            # 1 . 2 . . . . 3 . . . . 4 . 5

            # ==== second possibility ====
            #               s * * * * *
            #                         *
            #                         *
            #               * * * * * *
            # 1 . 2 . . . . 3 . . . . 4 . 5
            if (x, y) == (7, 0):  # choose the first step randomly
                return Move.LEFT if random.random() > 0.5 else Move.RIGHT
            if (
                (x, y) == (7, 3) or
                (x in (2, 12) and y < 3)
            ):
                return Move.DOWN
            if y == 0:
                return Move.LEFT if x < 7 else Move.RIGHT
            if y == 3:
                return Move.RIGHT if x < 7 else Move.LEFT
        elif self._policy is self.Policy.TWO:
            #     * * * * * s
            #     *
            #     *
            #     *
            # 1 . 2 . . . . 3 . . . . 4 . 5
            if ((x, y) == (2, 3)) or (x in (2, 12) and y < 3):
                return Move.DOWN
            else:
                return Move.RIGHT if x < 2 else Move.LEFT
        elif self._policy is self.Policy.THREE:
            #               s * * * * *
            #                         *
            #                         *
            #                         *
            # 1 . 2 . . . . 3 . . . . 4 . 5
            if ((x, y) == (12, 3)) or (x in (2, 12) and y < 3):
                return Move.DOWN
            else:
                return Move.RIGHT if x < 12 else Move.LEFT
        elif self._policy is self.Policy.FOUR:
            # * * * * * * * s
            # *
            # *
            # *
            # 1 . 2 . . . . 3 . . . . 4 . 5
            if ((x, y) == (0, 3)) or (x in (0, 12) and y < 3):
                return Move.DOWN
            else:
                return Move.RIGHT if x < 0 else Move.LEFT
        elif self._policy is self.Policy.FIVE:
            #               s * * * * * * *
            #                             *
            #                             *
            #                             *
            # 1 . 2 . . . . 3 . . . . 4 . 5
            if ((x, y) == (14, 3)) or (x in (2, 14) and y < 3):
                return Move.DOWN
            else:
                return Move.LEFT if x > 14 else Move.RIGHT

        raise ValueError(
            f'BprAgent with policy {self._policy} shouldn\'t be in {self.get_xy()}'
        )


class BprPlusAgent(BprAgent):
    def __init__(self, x=None, y=None):
        super().__init__(x, y)

    def update_belief(self, utility: int):
        # posterior (belief) = prior * likelihood (performance model)
        likelihood = ((self.performance_model[:, self.policy.value-1]) == utility).astype(float)
        belief_unnormalized = likelihood * self.belief / (np.sum(likelihood * self.belief) + 1e-6)
        self.belief = normalize_distribution(belief_unnormalized, 0.01)

    def update_policy(self):
        belief_mul_performance = self.belief @ self.performance_model
        candidates = np.argwhere(belief_mul_performance == np.amax(belief_mul_performance)).flatten().tolist()
        self.policy = list(self.Policy)[random.choice(candidates)]


class DeepBprPlusAgent(BprAgent):
    def __init__(self, x=None, y=None):
        super().__init__(x, y)
        self._tau_hat = np.ones(self.n_policies) / self.n_policies  # initial as uniform distribution

    @property
    def tau_hat(self):
        return self._tau_hat

    @tau_hat.setter
    def tau_hat(self, new_tau_hat):
        if not isinstance(new_tau_hat, np.ndarray) or new_tau_hat.shape != (self.n_policies,):
            raise ValueError(f'New tau_hat should be a numpy array with {self.n_policies} items')
        self._tau_hat = new_tau_hat

    def compute_tau_hat(self, state: Location):
        # opponent model observes the last state of opponent to get a distribution of tau
        if state == (0, 4):
            self.tau_hat = np.array([0, 0., 0., 1., 0.])
        elif state == (2, 4):
            self.tau_hat = np.array([0., 1., 0., 0., 0.])
        elif state == (7, 4):
            self.tau_hat = np.array([1., 0., 0., 0., 0.])
        elif state == (12, 4):
            self.tau_hat = np.array([0., 0., 1., 0., 0.])
        elif state == (14, 4):
            self.tau_hat = np.array([0., 0., 0., 0., 1.])
        else:
            self.tau_hat = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

    def update_belief(self, utility: int):
        # posterior (belief) = prior * likelihood (performance model)
        likelihood = ((self.performance_model[:, self.policy.value-1]) == utility).astype(float)
        belief_unnormalized = self.tau_hat * likelihood * self.belief / (np.sum(self.tau_hat * likelihood * self.belief) + 1e-6)
        self.belief = normalize_distribution(belief_unnormalized, 0.01)

    def update_policy(self):
        belief_mul_performance = self.belief @ self.performance_model
        candidates = np.argwhere(belief_mul_performance == np.amax(belief_mul_performance)).flatten().tolist()
        self.policy = list(self.Policy)[random.choice(candidates)]


class TomAgent(BprAgent):
    def __init__(
        self,
        x=None,
        y=None,
        confidence=0.7,
        l_episode=6,
        win_rate_threshold=0.1,
        adjustment_rate=0.2,
        indicator=1,
        first_order_prediction=1
    ):
        super().__init__(x, y)
        # the belief inherited from parent class is used as zero_order_belief
        self._first_order_belief = np.ones(self.n_policies) / self.n_policies
        self._integrated_belief = np.ones(self.n_policies) / self.n_policies

        # confidence_degree
        self._confidence = confidence

        # fixed legnth of episodes used to update confidence degree
        self.l_episode = l_episode
        self.win_rate_threshold = win_rate_threshold

        # adjustment rate of confidence degree
        self.adjustment_rate = adjustment_rate

        # Fvi indicator, used to update confidence degree
        self.indicator = indicator

        # initialize first order prediction
        self.first_order_prediction = first_order_prediction

    @property
    def first_order_belief(self):
        return self._first_order_belief

    @first_order_belief.setter
    def first_order_belief(self, new_belief):
        if not isinstance(new_belief, np.ndarray) or  new_belief.shape != (self.n_policies,):
            raise ValueError(f'New policy should be a numpy array with {self.n_policies} items')
        self._first_order_belief = new_belief

    @property
    def integrated_belief(self):
        return self._integrated_belief

    @integrated_belief.setter
    def integrated_belief(self, new_belief):
        if not isinstance(new_belief, np.ndarray) or new_belief.shape != (self.n_policies,):
            raise ValueError(f'New policy should be a numpy array with {self.n_policies} items')
        self._integrated_belief = new_belief

    @property
    def confidence(self):
        return self._confidence

    @confidence.setter
    def confidence(self, new_confidence):
        if not 0 <= new_confidence <= 1:
            raise ValueError(f'Confidence should be in the interval [0, 1], invalid value: {new_confidence}')
        self._confidence = new_confidence

    def compute_first_order_prediction(self):
        belief_mul_performance = self.first_order_belief @ np.transpose(self.performance_model)
        candidates = np.argwhere(belief_mul_performance == np.amin(belief_mul_performance)).flatten().tolist()
        self.first_order_prediction = random.choice(candidates) + 1

    def compute_integrated_belief(self):
        for op_policy in range(self.n_policies):
            if op_policy == self.first_order_prediction-1:
                self.integrated_belief[op_policy] = (1 - self.confidence) * self.belief[op_policy] + self.confidence
            else:
                self.integrated_belief[op_policy] = (1 - self.confidence) * self.belief[op_policy]

    def update_policy(self):
        belief_mul_performance = self.integrated_belief @ self.performance_model
        candidates = np.argwhere(belief_mul_performance == np.amax(belief_mul_performance)).flatten().tolist()
        self.policy = list(self.Policy)[random.choice(candidates)]

    def update_belief(self, current_n_episode, rewards):
        reward = rewards[-1]  # for calculating first-order and zero-order belief

        # update first_order_belief
        likelihood_pi = ((self.performance_model[self.first_order_prediction-1]) == reward).astype(float)
        first_order_belief_unnormalized = likelihood_pi * self.first_order_belief / (np.sum(likelihood_pi * self.first_order_belief) + 1e-6)

        # only update first-order belief when not all beliefs are zeros
        if np.sum(first_order_belief_unnormalized) > 0:
            self.first_order_belief = normalize_distribution(
                first_order_belief_unnormalized, 0.01
            )

        # update zero_order_belief
        likelihood_tau = ((self.performance_model[:, self.policy.value-1]) == reward).astype(float)
        belief_unnormalized = likelihood_tau * self.belief / (np.sum(likelihood_tau * self.belief) + 1e-6)
        self.belief = normalize_distribution(belief_unnormalized, 0.01)

        # update confidence degree
        if current_n_episode > self.l_episode:
            win_rate = np.average((np.array(rewards[current_n_episode-self.l_episode: current_n_episode-1]) >= 0).astype(int))
            print(f'win rate is {win_rate}')
            previous_win_rate = np.average((np.array(rewards[current_n_episode-self.l_episode-1: current_n_episode-2]) >= 0).astype(int))
            print(f'previous win rate is {previous_win_rate}')
            if win_rate >= previous_win_rate and win_rate > self.win_rate_threshold:
                self.confidence = ((1 - self.adjustment_rate) * self.confidence + self.adjustment_rate) * self.indicator
            elif win_rate < previous_win_rate and win_rate > self.win_rate_threshold:
                confidence = (self.confidence * np.log(win_rate) / np.log(win_rate - self.win_rate_threshold)) * self.indicator
                self.confidence = confidence if confidence <= 1 else 1
            else :
                self.confidence = self.adjustment_rate * self.indicator
                if self.indicator == 0:
                    self.indicator = 1
                elif self.indicator == 1:
                    self.indicator = 0


class BprOkrAgent(BprAgent):
    def __init__(self, x=None, y=None, l=5, rho=0.1):
        super().__init__(x, y)

        self.rho = rho  # belief weight paramenter

        # initialize intra-episode belief model
        self.intra_belief_model = IntraBeliefModel(n_policies=self.n_policies, l=l)

    @property
    def intra_belief(self):
        """
        In case anyone wants to inspect the intra-episode belief
        Notice that there's no setter for this property
        """
        return self.intra_belief_model.intra_belief

    def reset_intra_belief(self):
        """
        Assign the inter-epidode belief to the intra-episode belief
        """
        self.intra_belief_model.intra_belief = self.belief

    def update_intra_belief(self) -> None:
        """
        Update intra-episode belief using the state-action pairs
        int the experience queue
        """
        self.intra_belief_model.update_intra_belief()

    def update_belief(self, utility: int) -> None:
        """
        Use the following formula to update the belief.
        posterior (belief) = prior * likelihood (performance model)

        Args:
            utility (int): The agent's total rewards in a episode.
        """
        # when the agent switches its policy within an epsode
        # we might obtain an utility thats not in the performance model
        # the following method will give higher probability to the one
        # that is closer to the utility
        # e.g.
        # performance_model(pi2) = [-108, 42, -108, -110], utility = 40
        # abs(performance_model(pi2) - utility) = [148, 2, 148, 150]
        # reciprocal([148, 2, 148, 150]) = [0.0068, 0.5, 0.0068, 0.0067]
        # normalize([0.0068, 0.5, 0.0068, 0.0067]) = [0.013, 0.961, 0.013, 0.01288]
        likelihood = np.reciprocal(
            (np.abs((self.performance_model[:, self.policy.value-1]) - utility) + 1).astype(float)
        )
        likelihood /= np.sum(likelihood)
        belief_unnormalized = likelihood * self.belief / (np.sum(likelihood * self.belief) + 1e-6)
        self.belief = normalize_distribution(belief_unnormalized, 0.01)

    def add_experience_queue(self, state: Location, opponent_action: Move):
        self.intra_belief_model.add_experience_queue(state, opponent_action)

    def clear_experience_queue(self):
        self.intra_belief_model.clear_experience_queue()

    def update_policy(self, integrated_belief: bool = False):
        if integrated_belief:
            self.intra_belief_model.intra_belief = (
                self.rho * self.belief +
                (1 - self.rho) * self.intra_belief_model.intra_belief
            )
            self.intra_belief_model.intra_belief = normalize_distribution(
                self.intra_belief_model.intra_belief, 0.001
            )

        # when belief * performance model has same values within the results
        # `np.argmax` will always choose the one that has the smaller index
        # e.g. tau 2 over tau 3
        # now we fix it to randomly choose between those who have the same values
        belief_mul_performance = self.intra_belief_model.intra_belief @ self.performance_model
        candidates = np.argwhere(belief_mul_performance == np.amax(belief_mul_performance)).flatten().tolist()
        self.policy = list(self.Policy)[random.choice(candidates)]


class BsiBaseAgent(BprAgent):
    def __init__(self, x=None, y=None, l=5):
        super().__init__(x, y)

        # initialize phi belief
        n_tau = len(PhiOpponent.Phi) - 1  # phi 11 is an unknown policy
        self._phi_belief = np.ones(n_tau) / n_tau

        # initialize intra-episode belief model
        # need to use it when calculating observation model
        self.intra_belief_model = IntraBeliefModel(n_policies=self.n_policies, l=l)

        self.op_strategy_model = np.array([
            [1, 0, 0, 0, 0],  # phi 1: opponent always uses tau 1
            [0, 1, 0, 0, 0],  # phi 2: opponent always uses tau 2
            [0, 0, 1, 0, 0],  # phi 3: opponent always uses tau 3
            [0, 0, 0, 1, 0],  # phi 4: opponent always uses tau 4
            [0, 0, 0, 0, 1],  # phi 5: opponent always uses tau 5
            [0.2, 0.2, 0.2, 0.2, 0.2],  # phi 6: opponent always uses random switching
            [0, 0, 0, 0, 0],  # phi 7
            [0, 0, 0, 0, 0],  # phi 8
            [0, 0, 0, 0, 0],  # phi 9
            [0, 0, 0, 0, 0],  # phi 10
        ], dtype=np.float64)

        self.observation_model = None  # need to initialize through `set_observation_model`
        self.state_queue = []

    def set_observation_model(self, env: NavigationGame):
        if not isinstance(env, NavigationGame):
            raise RuntimeError('Please provide a `NavigationGame` envoronment')

        # observation = (agent state, opponent state)
        # each row represents an agent policy pi
        # each column represents an opponent policy tau
        self.observation_model = np.array([
            [(env.G3, env.G3), (env.G3, env.G2), (env.G3, env.G4), (env.G3, env.G1), (env.G3, env.G5)],
            [(env.G2, env.G3), (env.G2, env.G2), (env.G2, env.G4), (env.G2, env.G1), (env.G2, env.G5)],
            [(env.G4, env.G3), (env.G4, env.G2), (env.G4, env.G4), (env.G4, env.G1), (env.G4, env.G5)],
            [(env.G1, env.G3), (env.G1, env.G2), (env.G1, env.G4), (env.G1, env.G1), (env.G1, env.G5)],
            [(env.G5, env.G3), (env.G5, env.G2), (env.G5, env.G4), (env.G5, env.G1), (env.G5, env.G5)]
        ])

        # we have to save them for `update_op_strategy_model`
        self.G1, self.G2, self.G3, self.G4, self.G5 = env.G1, env.G2, env.G3, env.G4, env.G5

    @property
    def phi_belief(self):
        return self._phi_belief

    @phi_belief.setter
    def phi_belief(self, new_phi_belief):
        if not isinstance(new_phi_belief, np.ndarray) or len(new_phi_belief) != len(self.phi_belief):
            raise ValueError(
                f'Phi belief should be a numpy array with {len(self.phi_belief)} elements '
                f'(invalid: {new_phi_belief})'
            )
        self._phi_belief = new_phi_belief

    def update_op_strategy_model(self, sigma):
        agent_location, opponent_location = np.array(sigma).reshape(2, 2).tolist()
        agent_location = tuple(agent_location)
        opponent_location = tuple(opponent_location)

        terminal_state_combination = (
            opponent_location == self.G1 and agent_location == self.G1,
            opponent_location == self.G2 and agent_location == self.G2,
            opponent_location == self.G3 and agent_location == self.G3,
            opponent_location == self.G4 and agent_location == self.G4,
            opponent_location == self.G5 and agent_location == self.G5,
            opponent_location == self.G1 and agent_location != self.G1,
            opponent_location == self.G2 and agent_location != self.G2,
            opponent_location == self.G3 and agent_location != self.G3,
            opponent_location == self.G4 and agent_location != self.G4,
            opponent_location == self.G5 and agent_location != self.G5,
        )

        # phi 7
        if terminal_state_combination[1] or terminal_state_combination[8]:
            self.op_strategy_model[6] = [0, 0, 1, 0, 0]  # tau 3
        elif terminal_state_combination[7]:
            self.op_strategy_model[6] = [1, 0, 0, 0, 0]  # tau 1
        elif any((
            terminal_state_combination[0], terminal_state_combination[2],
            terminal_state_combination[3], terminal_state_combination[4],
            terminal_state_combination[6]
        )):
            self.op_strategy_model[6] = [0, 1, 0, 0, 0]  # tau 2
        elif terminal_state_combination[5]:
            self.op_strategy_model[6] = [0, 0, 0, 1, 0]  # tau 4
        else:
            self.op_strategy_model[6] = [0, 0, 0, 0, 1]  # tau 5

        # phi 8
        if terminal_state_combination[0] or terminal_state_combination[7]:
            self.op_strategy_model[7] = [0, 0, 1, 0, 0]  # tau 3
        elif terminal_state_combination[1] or terminal_state_combination[8]:
            self.op_strategy_model[7] = [0, 0, 0, 0, 1]  # tau 5
        elif terminal_state_combination[2] or terminal_state_combination[9]:
            self.op_strategy_model[7] = [0, 0, 0, 1, 0]  # tau 4
        elif terminal_state_combination[3] or terminal_state_combination[5]:
            self.op_strategy_model[7] = [0, 1, 0, 0, 0]  # tau 2
        elif terminal_state_combination[4] or terminal_state_combination[6]:
            self.op_strategy_model[7] = [1, 0, 0, 0, 0]    # tau 1

        # phi 9
        if any((
            terminal_state_combination[0], terminal_state_combination[1],
            terminal_state_combination[2], terminal_state_combination[9]
        )):
            self.op_strategy_model[8] = [0, 0, 0, 0, 1]
        elif any((
            terminal_state_combination[3], terminal_state_combination[4],
            terminal_state_combination[5]
        )):
            self.op_strategy_model[8] = [0, 0, 0, 1, 0]
        elif terminal_state_combination[6]:
            self.op_strategy_model[8] = [0, 1, 0, 0, 0]
        elif terminal_state_combination[7]:
            self.op_strategy_model[8] = [1, 0, 0, 0, 0]
        elif terminal_state_combination[8]:
            self.op_strategy_model[8] = [0, 0, 1, 0, 0]

        # phi 10
        if terminal_state_combination[0] or terminal_state_combination[8]:
            self.op_strategy_model[9] = [0, 0, 0, 0, 1]
        elif terminal_state_combination[1] or terminal_state_combination[6]:
            self.op_strategy_model[9] = [1, 0, 0, 0, 0]
        elif terminal_state_combination[2] or terminal_state_combination[7]:
            self.op_strategy_model[9] = [0, 0, 1, 0, 0]
        elif terminal_state_combination[3] or terminal_state_combination[9]:
            self.op_strategy_model[9] = [0, 0, 0, 1, 0]
        elif terminal_state_combination[4] or terminal_state_combination[5]:
            self.op_strategy_model[9] = [0, 1, 0, 0, 0]

    def update_phi(self) -> None:
        """
        This method contains the following three steps:
        1. Update the `op_strategy_model` using the previous state's sigma.
        2. Update the `observation_model` using current sigma
        3. Update `phi_belief` using `op_strategy_model` and `observation_model`
        """
        if len(self.state_queue) <= 1:
            return

        sigma_pre, sigma = self.state_queue[-2:]

        self.update_op_strategy_model(sigma_pre)

        observation_model_unnormalized = []
        for observation in self.observation_model[self.policy.value-1]:
            observation_model_unnormalized.append(np.array_equal(observation, np.array(sigma).reshape(2, 2)))
        observation_model = normalize_distribution(observation_model_unnormalized, 0.01)

        phi_belief_unnormalized = observation_model @ self.op_strategy_model.transpose() * self.phi_belief
        self.phi_belief = normalize_distribution(phi_belief_unnormalized, 0.0001)

    def infer_tau(self):
        """
        Find the distribution of next opponent tau with current `phi_belief`.
        Use the inferred tau as the inter-epidose belief and set it as initial intra-episode belief.
        """
        if len(self.state_queue) <= 1:
            return

        sigma = self.state_queue[-1]
        self.update_op_strategy_model(sigma)

        self.belief = self.phi_belief @ self.op_strategy_model

    def add_experience_queue(self, state: Location, opponent_action: Move):
        self.intra_belief_model.add_experience_queue(state, opponent_action)

    def clear_experience_queue(self):
        self.intra_belief_model.clear_experience_queue()

    def add_state_queue(self, state):
        self.state_queue.append(state)


class BsiAgent(BsiBaseAgent):
    def __init__(self, x=None, y=None):
        super().__init__(x, y)

    def update_policy(self):
        # when belief * performance model has same values within the results
        # `np.argmax` will always choose the one that has the smaller index
        # e.g. tau 2 over tau 3
        # now we fix it to randomly choose between those who have the same values
        belief_mul_performance = self.belief @ self.performance_model
        candidates = np.argwhere(belief_mul_performance == np.amax(belief_mul_performance)).flatten().tolist()
        self.policy = list(self.Policy)[random.choice(candidates)]


class BsiPtAgent(BsiBaseAgent):
    def __init__(self, x=None, y=None, l=5, rho=0.1):
        super().__init__(x, y)

        self.rho = rho  # belief weight paramenter

    @property
    def intra_belief(self):
        """
        In case anyone wants to inspect the intra-episode belief
        Notice that there's no setter for this property
        """
        return self.intra_belief_model.intra_belief

    def infer_tau(self):
        """
        Find the distribution of next opponent tau with current `phi_belief`.
        Use the inferred tau as the inter-epidose belief and set it as initial intra-episode belief.
        """
        super().infer_tau()
        self.intra_belief_model.intra_belief = self.belief

    def update_intra_belief(self) -> None:
        """
        Update intra-episode belief using the state-action pairs
        stored in the experience queue
        """
        self.intra_belief_model.update_intra_belief()

    def reset_intra_belief(self) -> None:
        """
        Reset intra-episode belief to uniform distribution.
        """
        self.intra_belief_model.reset_intra_belief()

    def update_policy(self, integrated_belief: bool = False):
        if integrated_belief:
            self.intra_belief_model.intra_belief = (
                self.rho * self.belief +
                (1 - self.rho) * self.intra_belief_model.intra_belief
            )
            self.intra_belief_model.intra_belief = normalize_distribution(
                self.intra_belief_model.intra_belief, 0.0001
            )

        # when belief * performance model has same values within the results
        # `np.argmax` will always choose the one that has the smaller index
        # e.g. tau 2 over tau 3
        # now we fix it to randomly choose between those who have the same values
        belief_mul_performance = self.intra_belief_model.intra_belief @ self.performance_model
        candidates = np.argwhere(belief_mul_performance == np.amax(belief_mul_performance)).flatten().tolist()
        self.policy = list(self.Policy)[random.choice(candidates)]


class IntraBeliefModel:
    def __init__(self, n_policies, l=5):
        self.n_policies = n_policies

        self.experience_queue = []
        self.l = l  # length of experience queue

        self._intra_belief = np.ones(self.n_policies) / self.n_policies  # initial as uniform distribution

    @property
    def intra_belief(self):
        return self._intra_belief

    @intra_belief.setter
    def intra_belief(self, new_belief):
        if not isinstance(new_belief, np.ndarray) or new_belief.shape != (self.n_policies,):
            raise ValueError(f'Belief should be a numpy array with {self.n_policies} items')
        self._intra_belief = new_belief

    def update_intra_belief(self) -> None:
        """
        Update intra-episode belief using experinece queue.
        See equation (8) in the OKR paper.
        """
        p_q_tau = self.prob_experience_queue()
        self.intra_belief = (
            (p_q_tau * self.intra_belief) / np.sum(p_q_tau * self.intra_belief)
        )
        self.intra_belief = normalize_distribution(self.intra_belief, 0.001)

    def reset_intra_belief(self) -> None:
        self.intra_belief = np.ones(self.n_policies) / self.n_policies

    def prob_experience_queue(self) -> np.ndarray:
        """
        Calculate the probability of generating this experience queue using each tau.
        See equation (9) in the OKR paper.

        Returns:
            np.ndarray: The probability of generating this experience queue using each tau.
        """
        sum_of_logs = np.zeros((self.n_policies,))
        for state, action in self.experience_queue[-self.l:]:
            sum_of_logs += np.log(0.1 + np.array(self.opponent_model(state, action)))
        exp_sum_of_logs = np.exp(sum_of_logs)
        return exp_sum_of_logs / np.sum(exp_sum_of_logs)

    def add_experience_queue(self, state: Location, opponent_action: Move):
        self.experience_queue.append((state, opponent_action))

    def clear_experience_queue(self):
        self.experience_queue.clear()

    def opponent_model(self, state: Location, action: Move) -> Tuple[int, int, int, int, int]:
        """
        Return if each policy is possible for this state-action pair.

        Args:
            state (Location): The location the opponent is at.
            action (Move): Tha action the opponent is doing.

        Returns:
            Tuple[int, int, int, int, int]: Each element indicate a tau, 1 means possible and 0 means not possible.
        """
        if state == (7, 8):
            if action == Move.UP:
                return (1, 1, 1, 0, 0)
            elif action == Move.LEFT:
                return (0, 0, 0, 1, 0)
            elif action == Move.RIGHT:
                return (0, 0, 0, 0, 1)
        elif state == (7, 7):
            if action == Move.UP:
                return (1, 0, 0, 0, 0)
            elif action == Move.LEFT:
                return (0, 1, 0, 0, 0)
            elif action == Move.RIGHT:
                return (0, 0, 1, 0, 0)
        elif state == (4, 6):
            if action == Move.LEFT:
                return (0, 1, 0, 1, 0)
        elif state == (3, 6):
            if action == Move.UP:
                return (0, 1, 0, 0, 0)
            elif action == Move.LEFT:
                return (0, 0, 0, 1, 0)
        elif state == (10, 6):
            if action == Move.RIGHT:
                return (0, 0, 1, 0, 1)
        elif state == (11, 6):
            if action == Move.UP:
                return (0, 0, 1, 0, 0)
            elif action == Move.RIGHT:
                return (0, 0, 0, 0, 1)
        elif state in ((7, 6), (7, 5), (7, 4)):
            # only Policy.ONE
            return (1, 0, 0, 0, 0)
        elif state in ((6, 7), (5, 7), (5, 6), (3, 5), (2, 5), (2, 4)):
            # only Policy.TWO
            return (0, 1, 0, 0, 0)
        elif state in ((8, 7), (9, 7), (9, 6), (11, 5), (12, 5), (12, 4)):
            # only Policy.THREE
            return (0, 0, 1, 0, 0)
        elif state in ((6, 8), (5, 8), (4, 8), (4, 7), (2, 6), (1, 6), (0, 6), (0, 5), (0, 4)):
            # only Policy.FOUR
            return (0, 0, 0, 1, 0)
        elif state in ((8, 8), (9, 8), (10, 8), (10, 7), (12, 6), (13, 6), (14, 6), (14, 5), (14, 4)):
            # only Policy.FIVE
            return (0, 0, 0, 0, 1)
        return (0, 0, 0, 0, 0)
