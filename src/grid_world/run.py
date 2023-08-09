import argparse
import random
from typing import Dict, Tuple

import numpy as np

from .agent import (
    BprAgent,
    BprOkrAgent,
    BprPlusAgent,
    BsiAgent,
    BsiPtAgent,
    DeepBprPlusAgent,
    TomAgent,
)
from .env import GridWorld, Location, NewPhiNoiseOpponent, NewPhiOpponent, Opponent, PhiOpponent


def run_bpr_plus(args: argparse.Namespace, **kwargs) -> Dict:
    setup_initial_policy(args)

    # setup the agent
    agent = BprPlusAgent(**kwargs)
    agent.policy = BprAgent.Policy(args.agent_policy)

    env = setup_environment(args, agent)

    rewards = []
    win_records = []
    for i in range(args.num_episodes):
        print(f'\n========== Start episode {i+1} ==========')
        print(f'Agent belief is {agent.belief}')
        print(f'Agent policy is {agent.policy}, Opponent policy is {env.opponent.policy}')

        total_reward = 0
        state = env.reset()
        while True:
            if args.print_map:
                print('----- MAP ------')
                env.show()
                print('----- MAP ------')

            done, reward, state, actions = env.step(agent.get_action(state[2:]))
            total_reward += reward

            if args.print_action:
                print(f'(agent action, opponent action) is {actions}')

            if done:
                if reward > 0:
                    win_records.append(True)
                else:
                    win_records.append(False)

                rewards.append(total_reward)
                print(f'Total reward of this episode is {total_reward}')
                break

        agent.update_belief(total_reward)
        agent.update_policy()
        opponent_update_policy(
            args=args,
            env=env,
            episode=i,
            reward=total_reward,
            ternimal_state=(state[:2], state[2:]),
        )

    print(f'Rewards: {rewards}')
    return {'rewards': rewards, 'win_records': win_records}


def run_deep_bpr_plus(args: argparse.Namespace, **kwargs) -> Dict:
    setup_initial_policy(args)

    # setup the agent
    agent = DeepBprPlusAgent(**kwargs)
    agent.policy = BprAgent.Policy(args.agent_policy)

    env = setup_environment(args, agent)

    rewards = []
    win_records = []
    for i in range(args.num_episodes):
        print(f'\n========== Start episode {i+1} ==========')
        print(f'Agent belief is {agent.belief}')
        print(f'Agent policy is {agent.policy}, Opponent policy is {env.opponent.policy}')

        total_reward = 0
        state = env.reset()
        state_batch = [state[2:]]  # record the states of the opponent to calculate tau_hat
        while True:
            if args.print_map:
                print('------- Map ---------')
                env.show()
                print('---------------------')

            done, reward, state, actions = env.step(agent.get_action(state[2:]))
            total_reward += reward

            if args.print_action:
                print(f'(agent action, opponent action) is {actions}')

            if done:
                if reward > 0:
                    win_records.append(True)
                else:
                    win_records.append(False)

                rewards.append(total_reward)
                print(f'Total reward of this episode is {total_reward}')
                break

            if state[2:] != state_batch[-1]:
                state_batch.append(state[2:])

        agent.compute_tau_hat(state_batch)  # use opponent model to estimate tau hat
        agent.update_belief(total_reward)  # use the episodic return to update belief
        agent.update_policy()
        opponent_update_policy(
            args=args,
            env=env,
            episode=i,
            reward=total_reward,
            ternimal_state=(state[:2], state[2:]),
        )

    print(f'Rewards: {rewards}')
    return {'rewards': rewards, 'win_records': win_records}


def run_tom(args: argparse.Namespace, **kwargs) -> Dict:
    setup_initial_policy(args)

    # setup the agent
    agent = TomAgent(**kwargs)
    agent.policy = BprAgent.Policy(args.agent_policy)

    env = setup_environment(args, agent)

    rewards = []
    win_records = []
    for i in range(args.num_episodes):
        print(f'\n========== Start episode {i+1} ==========')
        print(f'Agent zero order belief is {agent.belief}')
        print(f'Agent first order belief is {agent.first_order_belief}')
        print(f'Agent first order prediction is {agent.first_order_prediction}')
        print(f'Agent confidence degree is {agent.confidence}')
        print(f'Agent integrated belief is {agent.integrated_belief}')
        print(f'Agent policy is {agent.policy}, Opponent policy is {env.opponent.policy}')

        total_reward = 0
        state = env.reset()
        while True:
            if args.print_map:
                print('----- MAP ------')
                env.show()
                print('----- MAP ------')

            done, reward, state, actions = env.step(agent.get_action(state[2:]))
            total_reward += reward

            if args.print_action:
                print(f'(agent action, opponent action) is {actions}')

            if done:
                if reward > 0:
                    win_records.append(True)
                else:
                    win_records.append(False)

                rewards.append(total_reward)
                print(f'Total reward of this episode is {total_reward}')
                break

        # compute the agent's zero-order belief
        agent.update_belief(current_n_episode=i + 1, rewards=rewards)

        # compute the first-order opponent policy prediction
        agent.compute_first_order_prediction()

        # integrate the first-order predicted policy with zero-order belief
        agent.compute_integrated_belief()

        # select the optimal policy
        agent.update_policy()

        opponent_update_policy(
            args=args,
            env=env,
            episode=i,
            reward=total_reward,
            ternimal_state=(state[:2], state[2:]),
        )

    print(f'Rewards: {rewards}')
    return {'rewards': rewards, 'win_records': win_records}


def run_bpr_okr(args: argparse.Namespace, **kwargs) -> Dict:
    setup_initial_policy(args)

    # setup the agent
    agent = BprOkrAgent(**kwargs)
    agent.policy = BprAgent.Policy(args.agent_policy)

    env = setup_environment(args, agent)

    rewards = []
    win_records = []
    for i in range(args.num_episodes):
        print(f'\n========== Start episode {i+1} ==========')
        print(f'Agent Inter-episode belief is {agent.belief}')
        print(f'Agent initial intra belief is {agent.intra_belief}')
        print(f'Agent policy is {agent.policy}, Opponent policy is {env.opponent.policy}')

        total_reward = 0
        state = env.reset()
        while True:
            if args.print_map:
                print('----- MAP ------')
                env.show()
                print('----- MAP ------')

            done, reward, state_, actions = env.step(agent.get_action(state[2:]))
            total_reward += reward

            if args.print_action:
                print(f'(agent action, opponent action) is {actions}')

            # update performance model
            start_state = state_ if not done else None
            performance_model = env.generate_performance_model(start_state=start_state)
            agent.performance_model = performance_model

            agent.add_experience_queue(
                state[2:], actions[1]
            )  # only add the opponent's location and action to the experience queue
            agent.update_intra_belief()
            agent.update_policy(integrated_belief=True)  # compute the current integrated belief

            if done:
                if reward > 0:
                    win_records.append(True)
                else:
                    win_records.append(False)

                rewards.append(total_reward)
                print(f'Total reward of this episode is {total_reward}')
                break

            print(f'(Step {env.steps}) Intra-belief is updated to {agent.intra_belief}')
            print(
                f'(Step {env.steps}) Agent policy is {agent.policy}, '
                f'Opponent policy is {env.opponent.policy}'
            )

            state = state_

        # update performance model to initial state
        performance_model = env.generate_performance_model()

        agent.belief = agent.intra_belief
        # agent.update_belief(total_reward)  # use the episodic return to update belief
        agent.clear_experience_queue()  # empty the queue
        agent.reset_intra_belief()  # assign the inter-epsode belief to the intra-episode beleif
        agent.update_policy()
        opponent_update_policy(
            args=args,
            env=env,
            episode=i,
            reward=total_reward,
            ternimal_state=(state_[:2], state_[2:]),
        )

    print(f'Rewards: {rewards}')
    return {'rewards': rewards, 'win_records': win_records}


def run_bsi(args: argparse.Namespace, **kwargs) -> Dict:
    setup_initial_policy(args)

    # setup the agent
    agent = BsiAgent(**kwargs)
    agent.policy = BprAgent.Policy(args.agent_policy)

    env = setup_environment(args, agent)
    agent.set_observation_model(env=env)

    rewards = []
    phi_beliefs = []
    win_records = []
    policy_preds = []
    for i in range(args.num_episodes):
        print(f'\n========== Start episode {i+1} ==========')
        print(f'Agent Phi belief is {agent.phi_belief}')
        print(f'Agent initial belief is {agent.belief}')
        print(f'Agent policy is {agent.policy}, Opponent policy is {env.opponent.policy}')

        phi_beliefs.append(agent.phi_belief)

        total_reward = 0
        state = env.reset()
        while True:
            if args.print_map:
                print('----- MAP ------')
                env.show()
                print('----- MAP ------')

            done, reward, state_, actions = env.step(agent.get_action(state[2:4]))
            total_reward += reward

            if args.print_action:
                print(f'(agent action, opponent action) is {actions}')

            # only add the opponent's location and action to the experience queue
            agent.add_experience_queue(state[2:4], actions[1])

            if done:
                if reward > 0:
                    win_records.append(True)
                else:
                    win_records.append(False)

                rewards.append(total_reward)
                print(f'Total reward of this episode is {total_reward}')
                break

            state = state_

        if np.argmax(agent.belief) + 1 == env.opponent.policy.value:
            policy_preds.append(True)
        else:
            policy_preds.append(False)

        agent.add_state_queue(state_)
        agent.update_phi()
        agent.infer_tau()
        agent.update_policy()
        agent.clear_experience_queue()  # empty the queue

        opponent_update_policy(
            args=args,
            env=env,
            episode=i,
            reward=total_reward,
            ternimal_state=(state_[:2], state_[2:4]),
        )

        if i == 0:
            print('[BSI] Randomly choose a policy at the beginning of episode 2')
            agent.policy = agent.Policy(random.randint(1, 5))

    print(f'Rewards: {rewards}')

    corresponding_phi = None
    if isinstance(env.opponent, NewPhiOpponent) or isinstance(env.opponent, NewPhiNoiseOpponent):
        corresponding_phi = env.opponent.corresponding_phi

    return {
        'rewards': rewards,
        'phi_beliefs': phi_beliefs,
        'win_records': win_records,
        'policy_preds': policy_preds,
        'corresponding_phi': corresponding_phi,
    }


def run_bsi_pt(args: argparse.Namespace, **kwargs) -> Dict:
    setup_initial_policy(args)

    # setup the agent
    agent = BsiPtAgent(**kwargs)
    agent.policy = BprAgent.Policy(args.agent_policy)

    env = setup_environment(args, agent)
    agent.set_observation_model(env=env)

    rewards = []
    phi_beliefs = []
    win_records = []
    policy_preds = []
    for i in range(args.num_episodes):
        print(f'\n========== Start episode {i+1} ==========')
        print(f'Agent Phi belief is {agent.phi_belief}')
        print(f'Agent initial intra belief is {agent.belief}')
        print(f'Agent policy is {agent.policy}, Opponent policy is {env.opponent.policy}')

        phi_beliefs.append(agent.phi_belief)

        total_reward = 0
        state = env.reset()
        while True:
            if args.print_map:
                print('----- MAP ------')
                env.show()
                print('----- MAP ------')

            done, reward, state_, actions = env.step(agent.get_action(state[2:4]))
            total_reward += reward

            if args.print_action:
                print(f'(agent action, opponent action) is {actions}')

            # only add the opponent's location and action to the experience queue
            agent.add_experience_queue(state[2:4], actions[1])

            agent.update_intra_belief()

            # compute the current integrated belief
            agent.update_policy(integrated_belief=False)

            if done:
                if reward > 0:
                    win_records.append(True)
                else:
                    win_records.append(False)

                rewards.append(total_reward)
                print(f'Total reward of this episode is {total_reward}')
                break

            print(f'(Step {env.steps}) Intra-belief is updated to {agent.intra_belief}')
            print(
                f'(Step {env.steps}) Agent policy is {agent.policy}, '
                f'Opponent policy is {env.opponent.policy}'
            )

            state = state_

        if np.argmax(agent.intra_belief) + 1 == env.opponent.policy.value:
            policy_preds.append(True)
        else:
            policy_preds.append(False)

        agent.add_state_queue(state_)
        agent.update_phi()
        agent.infer_tau()
        agent.update_policy()
        agent.clear_experience_queue()  # empty the queue

        opponent_update_policy(
            args=args,
            env=env,
            episode=i,
            reward=total_reward,
            ternimal_state=(state_[:2], state_[2:4]),
        )

        if i == 0:
            print('[BSI-PT] Randomly choose a policy at the beginning of episode 2')
            agent.policy = agent.Policy(random.randint(1, 5))

    print(f'Rewards: {rewards}')

    corresponding_phi = None
    if isinstance(env.opponent, NewPhiOpponent) or isinstance(env.opponent, NewPhiNoiseOpponent):
        corresponding_phi = env.opponent.corresponding_phi

    return {
        'rewards': rewards,
        'phi_beliefs': phi_beliefs,
        'win_records': win_records,
        'policy_preds': policy_preds,
        'corresponding_phi': corresponding_phi,
    }


def setup_environment(args: argparse.Namespace, agent: BprAgent) -> GridWorld:
    env = GridWorld(
        agent=agent,
        bpr_opponent=args.bpr_opponent,
        phi_opponent=args.phi_opponent,
        new_phi_opponent=args.new_phi_opponent,
        new_phi_noise_opponent=args.new_phi_noise_opponent,
        q=args.q_distance,
        p_pattern=args.p_pattern,
    )

    # setup initial policy
    env.opponent.policy = Opponent.Policy(args.op_policy)
    if args.phi_opponent:
        env.opponent.phi = PhiOpponent.Phi(args.phi)

    # setup performance model
    performance_model = env.generate_performance_model()
    env.opponent.performance_model = performance_model
    agent.performance_model = performance_model

    return env


def opponent_update_policy(
    args: argparse.Namespace,
    env: GridWorld,
    episode: int,
    reward: int,
    ternimal_state: Tuple[Location, Location],
) -> None:
    if args.bpr_opponent:
        env.opponent.update_belief(reward)
        env.opponent.update_policy()
        print(f'Opponent belief updated to {env.opponent.belief}')
    elif args.phi_opponent or args.new_phi_opponent or args.new_phi_noise_opponent:
        env.opponent.update_policy(ternimal_state)
    elif (episode + 1) % args.episode_reset == 0:  # random switch opponent
        candidate = list(Opponent.Policy)
        candidate.remove(env.opponent.policy)
        env.opponent.policy = random.choice(candidate)


def setup_initial_policy(args: argparse.Namespace) -> None:
    if args.op_policy == -1:  # if user do not set it manually
        args.op_policy = random.randint(1, 5)

    if args.agent_policy == -1:  # if user do not set it manually
        args.agent_policy = random.randint(1, 5)

    if args.phi_opponent and args.phi <= 5:
        args.op_policy = args.phi  # the first five tau always use the same policy
