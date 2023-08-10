import random
from enum import Enum
from typing import Tuple

import numpy as np
from utils import normalize_distribution

from .env import Agent, Move, PhiOpponent, State


class BprAgent(Agent):
    class Policy(Enum):
        ONE = 1
        TWO = 2
        THREE = 3
        FOUR = 4

    def __init__(self, x=None, y=None):
        super().__init__(x, y)
        self.n_policies = len(self.Policy)
        self._policy = None
        self._belief = np.ones(self.n_policies) / self.n_policies  # initial as uniform distribution

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

    def get_action(self, state) -> Move:
        if not isinstance(self._policy, self.Policy):
            raise ValueError(f'BprAgent with invalid policy: {self._policy}')

        # see Table 2 in the paper
        # state = (strike, ball)
        applicable_states = ([0, 0], [0, 1], [0, 2], [0, 3], [1, 2], [1, 3])
        if self._policy is self.Policy.ONE:
            if state in applicable_states:
                return Move.STRIKE_1
        elif self._policy is self.Policy.TWO:
            if state in applicable_states:
                return Move.STRIKE_2
        elif self._policy is self.Policy.THREE:
            if state in applicable_states:
                return Move.STRIKE_3
        elif self._policy is self.Policy.FOUR:
            if state in applicable_states:
                return Move.STRIKE_4
        elif self._policy is self.Policy.FIVE:
            if state in ([0, 2], [0, 3]):
                return Move.STRIKE_1

        return Move.STRIKE_ALL


class BprPlusAgent(BprAgent):
    def __init__(self, x=None, y=None):
        super().__init__(x, y)

    def update_belief(self, utility: int):
        # posterior (belief) = prior * likelihood (performance model)
        likelihood = np.reciprocal(
            (np.abs((self.performance_model[:, self.policy.value - 1]) - utility) + 1).astype(float)
        )
        likelihood /= np.sum(likelihood)
        belief_unnormalized = likelihood * self.belief / (np.sum(likelihood * self.belief) + 1e-6)
        self.belief = normalize_distribution(belief_unnormalized, 0.001)

    def update_policy(self):
        # update policy based on belief and performance model
        belief_mul_performance = self.belief @ self.performance_model
        candidates = (
            np.argwhere(belief_mul_performance == np.amax(belief_mul_performance))
            .flatten()
            .tolist()
        )
        self.policy = list(self.Policy)[random.choice(candidates)]


class DeepBprPlusAgent(BprAgent):
    def __init__(self, x=None, y=None):
        super().__init__(x, y)
        # initial as uniform distribution
        self._tau_hat = np.ones(self.n_policies) / self.n_policies
        # initialize intra-episode belief model
        # Deep-BPR+ doesn't have a intra-belief model by definition
        # but we can utilize the experience queue + opponent_model in the intra-belief model
        # to update the belief of Deep-BPR+ after each episode
        self.intra_belief_model = IntraBeliefModel(n_policies=self.n_policies, l=5)

    @property
    def tau_hat(self):
        return self._tau_hat

    @tau_hat.setter
    def tau_hat(self, new_tau_hat):
        if not isinstance(new_tau_hat, np.ndarray) or new_tau_hat.shape != (self.n_policies,):
            raise ValueError(f'New tau_hat should be a numpy array with {self.n_policies} items')
        self._tau_hat = new_tau_hat

    def reset_intra_belief(self):
        """
        Assign an initial value to the intra-episode belief
        """
        self.intra_belief_model.intra_belief = [0.25, 0.25, 0.25, 0.25]

    def add_experience_queue(self, state: State, opponent_action: Move):
        self.intra_belief_model.add_experience_queue(state, opponent_action)

    def clear_experience_queue(self):
        self.intra_belief_model.clear_experience_queue()

    def compute_tau_hat(self):
        # use the intra_belief calculated after each episode to represent tau_hat
        self.intra_belief_model.update_intra_belief()
        self.tau_hat = self.intra_belief_model.intra_belief

    def update_belief(self, utility: int):
        # posterior (belief) = prior * likelihood (performance model)
        likelihood = np.reciprocal(
            (np.abs((self.performance_model[:, self.policy.value - 1]) - utility) + 1).astype(float)
        )
        likelihood /= np.sum(likelihood)
        belief_unnormalized = (
            self.tau_hat
            * likelihood
            * self.belief
            / (np.sum(self.tau_hat * likelihood * self.belief) + 1e-6)
        )
        self.belief = normalize_distribution(belief_unnormalized, 0.001)

    def update_policy(self):
        belief_mul_performance = self.belief @ self.performance_model
        candidates = (
            np.argwhere(belief_mul_performance == np.amax(belief_mul_performance))
            .flatten()
            .tolist()
        )
        self.policy = list(self.Policy)[random.choice(candidates)]


class BprOkrAgent(BprAgent):
    def __init__(self, x=None, y=None, l=3, rho=0.1):
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
            (np.abs((self.performance_model[:, self.policy.value - 1]) - utility) + 1).astype(float)
        )
        likelihood /= np.sum(likelihood)
        belief_unnormalized = likelihood * self.belief / (np.sum(likelihood * self.belief) + 1e-6)
        self.belief = normalize_distribution(belief_unnormalized, 0.001)

    def add_experience_queue(self, state: State, opponent_action: Move):
        self.intra_belief_model.add_experience_queue(state, opponent_action)

    def clear_experience_queue(self):
        self.intra_belief_model.clear_experience_queue()

    def update_policy(self, integrated_belief: bool = False):
        if integrated_belief:
            self.intra_belief_model.intra_belief = (
                self.rho * self.belief + (1 - self.rho) * self.intra_belief_model.intra_belief
            )
            self.intra_belief_model.intra_belief = normalize_distribution(
                self.intra_belief_model.intra_belief, 0.001
            )

        # when belief * performance model has same values within the results
        # `np.argmax` will always choose the one that has the smaller index
        # e.g. tau 2 over tau 3
        # now we fix it to randomly choose between those who have the same values
        belief_mul_performance = self.intra_belief_model.intra_belief @ self.performance_model
        candidates = (
            np.argwhere(belief_mul_performance == np.amax(belief_mul_performance))
            .flatten()
            .tolist()
        )
        self.policy = list(self.Policy)[random.choice(candidates)]


class BsiBaseAgent(BprAgent):
    def __init__(self, x=None, y=None, l=3):
        super().__init__(x, y)

        # initialize phi belief
        n_tau = len(PhiOpponent.Phi) - 3  # there are three phi uknown to the agent
        self._phi_belief = np.ones(n_tau) / n_tau

        # initialize intra-episode belief model
        # need to use it when calculating observation model
        self.intra_belief_model = IntraBeliefModel(n_policies=self.n_policies, l=l)

        # all 4 phi opponents starts the first episode by randomly choose a policy tau
        self.op_strategy_model = np.array(
            [
                [1, 0, 0, 0],  # phi 1: opponent always uses tau 1
                [0, 1, 0, 0],  # phi 2: opponent always uses tau 2
                [0, 0, 1, 0],  # phi 3: opponent always uses tau 3
                [0, 0, 0, 1],  # phi 4: opponent always uses tau 4
                [0.25, 0.25, 0.25, 0.25],  # phi 5: opponent always uses random switching
                [0, 0, 0, 0],  # phi 6
                [0, 0, 0, 0],  # phi 7
                [0, 0, 0, 0],  # phi 8
                [0, 0, 0, 0],  # phi 9
            ],
            dtype=np.float64,
        )

        self.observation_model = None  # need to initialize through `set_observation_model`
        self.sigma_queue = []

    @property
    def phi_belief(self):
        return self._phi_belief

    @phi_belief.setter
    def phi_belief(self, new_phi_belief):
        if not isinstance(new_phi_belief, np.ndarray) or len(new_phi_belief) != len(
            self.phi_belief
        ):
            raise ValueError(
                f'Phi belief should be a numpy array with {len(self.phi_belief)} elements '
                f'(invalid: {new_phi_belief})'
            )
        self._phi_belief = new_phi_belief

    def update_op_strategy_model(self, sigma):
        # self.op_strategy_model[0] -> probabilities of 4 tau for phi 1
        # self.op_strategy_model[1] -> probabilities of 4 tau for phi 2
        # self.op_strategy_model[2] -> probabilities of 4 tau for phi 3
        # ... etc
        final_state, final_action, final_reward = sigma

        # phi 6
        # if (3478 hit/walk) or (1256 out/strike_out) or (78 walk)
        if (
            final_action in [Move.STRIKE_3, Move.STRIKE_4, Move.BALL_7, Move.BALL_8]
            and final_reward > 0
        ) or (
            final_action in [Move.STRIKE_1, Move.STRIKE_2, Move.BALL_5, Move.BALL_6]
            and final_reward <= 0
        ):
            self.op_strategy_model[5] = [1, 0, 0, 0]
        else:
            self.op_strategy_model[5] = [0, 0, 1, 0]
        # phi 7
        # if (3478 hit/walk) or (1256 out/strike_out) or (78 walk)
        if (
            final_action in [Move.STRIKE_3, Move.STRIKE_4, Move.BALL_7, Move.BALL_8]
            and final_reward > 0
        ) or (
            final_action in [Move.STRIKE_1, Move.STRIKE_2, Move.BALL_5, Move.BALL_6]
            and final_reward <= 0
        ):
            self.op_strategy_model[6] = [0, 0, 0, 1]
        else:
            self.op_strategy_model[6] = [0, 1, 0, 0]
        # phi 8
        # if (1256 hit) or (1256 stirke out)
        if (
            final_action in [Move.STRIKE_1, Move.STRIKE_2, Move.BALL_5, Move.BALL_6]
            and final_reward > 0
            and final_state[1] < 4
        ) or (
            final_action in [Move.STRIKE_3, Move.STRIKE_4, Move.BALL_7, Move.BALL_8]
            and final_reward <= 0
            and final_state[0] < 3
        ):
            self.op_strategy_model[7] = [1, 0, 0, 0]
        else:
            self.op_strategy_model[7] = [0, 0, 1, 0]
        # phi 9
        # if hit/walk
        if final_reward > 0:
            self.op_strategy_model[8] = [0, 0, 0, 1]
        else:
            self.op_strategy_model[8] = [0, 0, 1, 0]
        # phi 10 & phi 11 & phi 12 is not in the strategy model

    def update_phi(self) -> None:
        """
        This method contains the following three steps:
        1. Update the `op_strategy_model` using the previous state's sigma.
        2. Update the `observation_model` using current sigma
        3. Update `phi_belief` using `op_strategy_model` and `observation_model`
        """
        if len(self.sigma_queue) <= 1:
            return

        sigma_pre = self.sigma_queue[-2]

        self.update_op_strategy_model(sigma_pre)

        # directly use the observation model in intra_belief_model instead of
        # using the terminal state as the parameter
        # TODO: explain why
        observation_model = self.intra_belief_model.prob_experience_queue()
        phi_belief_unnormalized = (
            observation_model @ self.op_strategy_model.transpose() * self.phi_belief
        )
        self.phi_belief = normalize_distribution(phi_belief_unnormalized, 0.0001)

    def infer_tau(self):
        """
        Find the distribution of next opponent tau with current `phi_belief`.
        Use the inferred tau as the inter-epidose belief and set it as initial intra-episode belief.
        """
        if len(self.sigma_queue) <= 1:
            return

        sigma = self.sigma_queue[-1]
        self.update_op_strategy_model(sigma)

        self.belief = self.phi_belief @ self.op_strategy_model

    def add_experience_queue(self, state: State, opponent_action: Move):
        self.intra_belief_model.add_experience_queue(state, opponent_action)

    def clear_experience_queue(self):
        self.intra_belief_model.clear_experience_queue()

    def add_terminal_state_queue(self, sigma):
        self.sigma_queue.append(sigma)


class BsiAgent(BsiBaseAgent):
    def __init__(self, x=None, y=None):
        super().__init__(x, y)

    def update_policy(self):
        # when belief * performance model has same values within the results
        # `np.argmax` will always choose the one that has the smaller index
        # e.g. tau 2 over tau 3
        # now we fix it to randomly choose between those who have the same values
        belief_mul_performance = self.belief @ self.performance_model
        candidates = (
            np.argwhere(belief_mul_performance == np.amax(belief_mul_performance))
            .flatten()
            .tolist()
        )
        self.policy = list(self.Policy)[random.choice(candidates)]


class BsiPtAgent(BsiBaseAgent):
    def __init__(self, x=None, y=None, l=3, rho=0.1):
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

    def infer_tau2(self):
        """
        Use the uniform tau as initial intra-episode belief.
        """
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
                self.rho * self.belief + (1 - self.rho) * self.intra_belief_model.intra_belief
            )
            self.intra_belief_model.intra_belief = normalize_distribution(
                self.intra_belief_model.intra_belief, 0.01
            )

        # when belief * performance model has same values within the results
        # `np.argmax` will always choose the one that has the smaller index
        # e.g. tau 2 over tau 3
        # now we fix it to randomly choose between those who have the same values
        belief_mul_performance = self.intra_belief_model.intra_belief @ self.performance_model
        candidates = (
            np.argwhere(belief_mul_performance == np.amax(belief_mul_performance))
            .flatten()
            .tolist()
        )
        self.policy = list(self.Policy)[random.choice(candidates)]


class IntraBeliefModel:
    def __init__(self, n_policies, l=3):
        self.n_policies = n_policies
        self.experience_queue = []
        self.l = l  # length of experience queue
        self._intra_belief = (
            np.ones(self.n_policies) / self.n_policies
        )  # initial as uniform distribution

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
        self.intra_belief = (p_q_tau * self.intra_belief) / np.sum(p_q_tau * self.intra_belief)
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
        for state, action in self.experience_queue[-self.l :]:
            sum_of_logs += np.log(0.1 + np.array(self.opponent_model(state, action)))
        exp_sum_of_logs = np.exp(sum_of_logs)
        return exp_sum_of_logs / np.sum(exp_sum_of_logs)

    def add_experience_queue(self, state: State, opponent_action: Move):
        self.experience_queue.append((state, opponent_action))

    def clear_experience_queue(self):
        self.experience_queue.clear()

    def opponent_model(self, state: State, opponent_action: Move) -> Tuple[int, int, int, int]:
        """
        Return if each policy is possible for this state-action pair.

        Args:
            state (State): Current strike/ball count.
            opponent_action (Move): Tha action the opponent is doing.

        Returns:
            Tuple[int, int, int, int]: Each element indicate the probability of a tau.
        """

        # mu1: probability of the action that is most likely to happen
        action_control_constant_mu1 = 0.6
        # define the probability of other less likely actions
        mu2 = (1 - action_control_constant_mu1) / 16

        # each row represents a tau, each column represents a possible opponent action
        situation1 = np.array(
            [
                [action_control_constant_mu1, 4 * mu2, 4 * mu2, 4 * mu2, mu2, mu2, mu2, mu2],
                [4 * mu2, action_control_constant_mu1, 4 * mu2, 4 * mu2, mu2, mu2, mu2, mu2],
                [4 * mu2, 4 * mu2, action_control_constant_mu1, 4 * mu2, mu2, mu2, mu2, mu2],
                [4 * mu2, 4 * mu2, 4 * mu2, action_control_constant_mu1, mu2, mu2, mu2, mu2],
            ]
        )
        situation2 = np.array(
            [
                [4 * mu2, mu2, mu2, mu2, action_control_constant_mu1, 4 * mu2, 4 * mu2, mu2],
                [mu2, 4 * mu2, mu2, mu2, 4 * mu2, action_control_constant_mu1, mu2, 4 * mu2],
                [mu2, mu2, 4 * mu2, mu2, 4 * mu2, mu2, action_control_constant_mu1, 4 * mu2],
                [mu2, mu2, mu2, 4 * mu2, mu2, 4 * mu2, 4 * mu2, action_control_constant_mu1],
            ]
        )
        situation3 = np.array(
            [
                [mu2, mu2, mu2, mu2, action_control_constant_mu1, 4 * mu2, 4 * mu2, 4 * mu2],
                [mu2, mu2, mu2, mu2, 4 * mu2, action_control_constant_mu1, 4 * mu2, 4 * mu2],
                [mu2, mu2, mu2, mu2, 4 * mu2, 4 * mu2, action_control_constant_mu1, 4 * mu2],
                [mu2, mu2, mu2, mu2, 4 * mu2, 4 * mu2, 4 * mu2, action_control_constant_mu1],
            ]
        )

        # situation 1
        if state in ([0, 0], [0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 2], [2, 3]):
            return tuple(situation1[:, opponent_action.value - 1])  # type: ignore
        # situation 2
        elif state in ([1, 0], [1, 1]):
            return tuple(situation2[:, opponent_action.value - 1])  # type: ignore
        # situation 3
        elif state in ([2, 0], [2, 1]):
            return tuple(situation3[:, opponent_action.value - 1])  # type: ignore

        return (0, 0, 0, 0)
