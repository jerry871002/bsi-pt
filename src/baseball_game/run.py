import argparse
import random
from typing import Dict, List, Tuple

import numpy as np
from scipy.special import kl_div, rel_entr

from utils import normalize_distribution

# from .env import (BaseballGame, NewPhiNoiseOpponent,
#                   NewPhiOpponent, Opponent, PhiOpponent)

from .agent import (
    BprAgent,
    BprOkrAgent,
    BprPlusAgent,
    BsiAgent,
    BsiPtAgent,
    DeepBprPlusAgent,
    # TomAgent
)
from .env import BaseballGame, Opponent, PhiOpponent


def run_bpr_plus(args: argparse.Namespace, **kwargs) -> Dict:
    np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
    setup_initial_policy(args)

    # setup the agent
    agent = BprPlusAgent(**kwargs)
    agent.policy = BprAgent.Policy(args.agent_policy)

    env = setup_environment(args, agent)

    rewards = []
    win_records = []
    policy_preds = []
    for i in range(args.num_episodes):
        # print(f'\n========== Start episode {i+1} ==========')
        # print(f'Agent belief is {agent.belief}')
        # print(f'Agent policy is {agent.policy}, Opponent policy is {env.opponent.policy}')

        total_reward = 0
        env.reset()
        while True:
            # print(f'--- step {env.steps} ---')
            # print(f'current state [strike, ball] = {env.state}')
            done, reward, state_, actions, episode_result = env.step(agent.get_action(env.state))
            total_reward += reward

            if args.print_map:
                # print('------- Map ---------')
                env.show(actions)
                # print('-------------------')

            if args.print_action:
                print(f'(agent action, opponent action) is {actions}')

            if done:
                if reward > 0:
                    win_records.append(True)
                else:
                    win_records.append(False)

                rewards.append(total_reward)
                # print(f'episode result: {episode_result}, episode reward: {total_reward}')
                break

            env.state = state_

            if (
                not args.bpr_opponent
                and not args.phi_opponent
                and not args.new_phi_noise_opponent
                and args.episode_reset < 0
            ):
                opponent_update_policy(
                    args=args,
                    env=env,
                    episode=i,
                    final_reward=total_reward,
                    final_action=actions[1],
                    final_result=episode_result,
                )

        # policy prediction accuracy of the first two episodes is not applicable
        if np.all(agent.belief == agent.belief[0]):
            if random.randint(1, agent.n_policies) == env.opponent.policy.value:
                policy_preds.append(True)
            else:
                policy_preds.append(False)
        else:
            if np.argmax(agent.belief) + 1 == env.opponent.policy.value:
                policy_preds.append(True)
            else:
                policy_preds.append(False)

        agent.update_belief(total_reward)
        agent.update_policy()

        # opponent_update_policy(args=args, env=env, episode=i, reward=total_reward, ternimal_state=(state[:2], state[2:]))
        opponent_update_policy(
            args=args,
            env=env,
            episode=i,
            final_reward=total_reward,
            final_action=actions[1],
            final_result=episode_result,
        )

    '''print(f'Rewards: {rewards}')
    print(f'Hit rate (AVG) = {env.hit_count/(env.hit_count+env.strike_out_count+env.hit_out_count)}')
    print(f'On base percentage (OBP) = {(env.walk_count+env.hit_count)/(env.walk_count+env.hit_count+env.strike_out_count+env.hit_out_count)}')

    print(f'hit and score {env.hit_count}')
    print(f'strike out {env.strike_out_count}')
    print(f'hit and out {env.hit_out_count} ')
    print(f'walk count {env.walk_count}')'''

    return {
        'rewards': rewards,
        'win_records': win_records,
        'policy_preds': policy_preds,
        'hit_count': env.hit_count,
        'strike_out_count': env.strike_out_count,
        'hit_out_count': env.hit_out_count,
        'walk_count': env.walk_count,
    }


def run_deep_bpr_plus(args: argparse.Namespace, **kwargs) -> Dict:
    np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
    setup_initial_policy(args)

    # setup the agent
    agent = DeepBprPlusAgent(**kwargs)
    agent.policy = BprAgent.Policy(args.agent_policy)

    env = setup_environment(args, agent)

    rewards = []
    win_records = []
    policy_preds = []
    for i in range(args.num_episodes):
        # print(f'\n========== Start episode {i+1} ==========')
        # print(f'Agent belief is {agent.belief}')
        # print(f'Agent policy is {agent.policy}, Opponent policy is {env.opponent.policy}')

        total_reward = 0
        env.reset()
        while True:
            # print(f'--- step {env.steps} ---')
            # print(f'current state [strike, ball] = {env.state}')
            done, reward, state_, actions, episode_result = env.step(agent.get_action(env.state))
            total_reward += reward

            if args.print_map:
                # print('------- Map ---------')
                env.show(actions)
                # print('---------------------')

            if args.print_action:
                print(f'(agent action, opponent action) is {actions}')

            if done:
                if reward > 0:
                    win_records.append(True)
                else:
                    win_records.append(False)

                rewards.append(total_reward)
                # print(f'episode result: {episode_result}, episode reward: {total_reward}')
                break

            agent.add_experience_queue(
                env.state, actions[1]
            )  # only add the opponent's location and action to the experience queue
            env.state = state_

            if (
                not args.bpr_opponent
                and not args.phi_opponent
                and not args.new_phi_noise_opponent
                and args.episode_reset < 0
            ):  # intra episode random switch opponent
                opponent_update_policy(
                    args=args,
                    env=env,
                    episode=i,
                    final_reward=total_reward,
                    final_action=actions[1],
                    final_result=episode_result,
                )

        # policy prediction accuracy of the first two episodes is not applicable
        if np.all(agent.belief == agent.belief[0]):
            if random.randint(1, agent.n_policies) == env.opponent.policy.value:
                policy_preds.append(True)
            else:
                policy_preds.append(False)
        else:
            if np.argmax(agent.belief) + 1 == env.opponent.policy.value:
                policy_preds.append(True)
            else:
                policy_preds.append(False)

        agent.compute_tau_hat()  # use opponent model + experience_queue to estimate tau hat
        agent.update_belief(total_reward)  # use the episodic return to update belief
        agent.update_policy()

        # opponent_update_policy(args=args, env=env, episode=i, reward=total_reward, ternimal_state=(state[:2], state[2:]))
        opponent_update_policy(
            args=args,
            env=env,
            episode=i,
            final_reward=total_reward,
            final_action=actions[1],
            final_result=episode_result,
        )

    '''print(f'Rewards: {rewards}')
    print(f'Hit rate (AVG) = {env.hit_count/(env.hit_count+env.strike_out_count+env.hit_out_count)}')
    print(f'On base percentage (OBP) = {(env.walk_count+env.hit_count)/(env.walk_count+env.hit_count+env.strike_out_count+env.hit_out_count)}')

    print(f'hit and score {env.hit_count}')
    print(f'strike out {env.strike_out_count}')
    print(f'hit and out {env.hit_out_count} ')
    print(f'walk count {env.walk_count}')'''

    return {
        'rewards': rewards,
        'win_records': win_records,
        'policy_preds': policy_preds,
        'hit_count': env.hit_count,
        'strike_out_count': env.strike_out_count,
        'hit_out_count': env.hit_out_count,
        'walk_count': env.walk_count,
    }


def run_bpr_okr(args: argparse.Namespace, **kwargs) -> Dict:
    np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
    setup_initial_policy(args)

    # setup the agent
    agent = BprOkrAgent(**kwargs)
    agent.policy = BprAgent.Policy(args.agent_policy)

    env = setup_environment(args, agent)

    rewards = []
    win_records = []
    policy_preds = []
    step_0_policy_preds = []  # indicates whether the agent can choose the write policy before the episode starts
    step_1_policy_preds = []
    step_2_policy_preds = []
    step_3_policy_preds = []
    step_4_policy_preds = []
    step_5_policy_preds = []
    step_6_policy_preds = []
    for i in range(args.num_episodes):

        if i == 0:
            agent.update_policy()

        '''print(f'\n========== Start episode {i+1} ==========')
        print(f'Agent Inter-episode belief is {agent.belief}')
        print(f'Agent initial intra belief is {agent.intra_belief}')
        print(f'Agent policy is {agent.policy}, Opponent policy is {env.opponent.policy}')'''

        # policy prediction accuracy of the first two episodes is not applicable
        if np.all(agent.intra_belief == agent.intra_belief[0]):  # agent's belief is still uniform
            if random.randint(1, agent.n_policies) == env.opponent.policy.value:
                step_0_policy_preds.append(True)
            else:
                step_0_policy_preds.append(False)
        else:
            if np.argmax(agent.intra_belief) + 1 == env.opponent.policy.value:
                step_0_policy_preds.append(True)
            else:
                step_0_policy_preds.append(False)

        total_reward = 0
        env.reset()
        while True:
            # print(f'--- step {env.steps} ---')
            # print(f'current state [strike, ball] = {env.state}')

            done, reward, state_, actions, episode_result = env.step(agent.get_action(env.state))
            total_reward += reward

            if args.print_map:
                # print('----- MAP ------')
                env.show(actions)
                # print('----- MAP ------')

            if args.print_action:
                print(f'(agent action, opponent action) is {actions}')

            agent.add_experience_queue(
                env.state, actions[1]
            )  # only add the opponent's location and action to the experience queue
            agent.update_intra_belief()
            agent.update_policy(integrated_belief=True)  # compute the current integrated belief
            env.state = state_

            if env.steps == 1:
                if np.argmax(agent.intra_belief) + 1 == env.opponent.policy.value:
                    step_1_policy_preds.append(True)
                else:
                    step_1_policy_preds.append(False)
            elif env.steps == 2:
                if np.argmax(agent.intra_belief) + 1 == env.opponent.policy.value:
                    step_2_policy_preds.append(True)
                else:
                    step_2_policy_preds.append(False)
            elif env.steps == 3:
                if np.argmax(agent.intra_belief) + 1 == env.opponent.policy.value:
                    step_3_policy_preds.append(True)
                else:
                    step_3_policy_preds.append(False)
            elif env.steps == 4:
                if np.argmax(agent.intra_belief) + 1 == env.opponent.policy.value:
                    step_4_policy_preds.append(True)
                else:
                    step_4_policy_preds.append(False)
            elif env.steps == 5:
                if np.argmax(agent.intra_belief) + 1 == env.opponent.policy.value:
                    step_5_policy_preds.append(True)
                else:
                    step_5_policy_preds.append(False)
            elif env.steps == 6:
                if np.argmax(agent.intra_belief) + 1 == env.opponent.policy.value:
                    step_6_policy_preds.append(True)
                else:
                    step_6_policy_preds.append(False)

            if done:
                if reward > 0:
                    win_records.append(True)
                else:
                    win_records.append(False)
                rewards.append(total_reward)

                '''if env.steps < 6:
                    step_6_policy_preds.append(None)
                if env.steps < 5:
                    step_5_policy_preds.append(None)
                if env.steps < 4:
                    step_4_policy_preds.append(None)
                if env.steps < 3:
                    step_3_policy_preds.append(None)
                if env.steps < 2:
                    step_2_policy_preds.append(None)
                if env.steps < 1:
                    step_1_policy_preds.append(None)'''

                # print(f'episode result: {episode_result}, episode reward: {total_reward}')
                break

            if (
                not args.bpr_opponent
                and not args.phi_opponent
                and not args.new_phi_noise_opponent
                and args.episode_reset < 0
            ):  # intra episode random switch opponent
                opponent_update_policy(
                    args=args,
                    env=env,
                    episode=i,
                    final_reward=total_reward,
                    final_action=actions[1],
                    final_result=episode_result,
                )

            # print(f'(Step {env.steps}) Intra-belief is updated to {agent.intra_belief}')
            # print(f'(Step {env.steps}) Agent policy is {agent.policy}, Opponent policy is {env.opponent.policy}')

        # policy prediction accuracy of the first two episodes is not applicable
        '''if np.all(agent.intra_belief == agent.intra_belief[0]):
            if random.randint(1, agent.n_policies) == env.opponent.policy.value:
                policy_preds.append(True)
            else:
                policy_preds.append(False)
        else:
            if np.argmax(agent.intra_belief) + 1 == env.opponent.policy.value:
                policy_preds.append(True)
            else:
                policy_preds.append(False)'''

        if np.all(agent.intra_belief == agent.intra_belief[0]):
            if random.randint(1, agent.n_policies) == env.opponent.policy.value:
                policy_preds.append(True)
            else:
                policy_preds.append(False)
        else:
            if env.steps == 1:
                policy_preds.append(step_0_policy_preds[i])
                step_2_policy_preds.append(step_1_policy_preds[i])
                step_3_policy_preds.append(step_1_policy_preds[i])
                step_4_policy_preds.append(step_1_policy_preds[i])
                step_5_policy_preds.append(step_1_policy_preds[i])
                step_6_policy_preds.append(step_1_policy_preds[i])
            elif env.steps == 2:
                policy_preds.append(step_1_policy_preds[i])
                step_3_policy_preds.append(step_2_policy_preds[i])
                step_4_policy_preds.append(step_2_policy_preds[i])
                step_5_policy_preds.append(step_2_policy_preds[i])
                step_6_policy_preds.append(step_2_policy_preds[i])
            elif env.steps == 3:
                policy_preds.append(step_2_policy_preds[i])
                step_4_policy_preds.append(step_3_policy_preds[i])
                step_5_policy_preds.append(step_3_policy_preds[i])
                step_6_policy_preds.append(step_3_policy_preds[i])
            elif env.steps == 4:
                policy_preds.append(step_3_policy_preds[i])
                step_5_policy_preds.append(step_4_policy_preds[i])
                step_6_policy_preds.append(step_4_policy_preds[i])
            elif env.steps == 5:
                policy_preds.append(step_4_policy_preds[i])
                step_6_policy_preds.append(step_5_policy_preds[i])
            elif env.steps == 6:
                policy_preds.append(step_5_policy_preds[i])

        agent.belief = agent.intra_belief
        # agent.update_belief(total_reward)  # use the episodic return to update belief
        agent.clear_experience_queue()  # empty the queue
        agent.reset_intra_belief()  # assign the inter-epsode belief to the intra-episode beleif
        agent.update_policy()

        # opponent_update_policy(args=args, env=env, episode=i, reward=total_reward, ternimal_state=(state_[:2], state_[2:]))
        opponent_update_policy(
            args=args,
            env=env,
            episode=i,
            final_reward=total_reward,
            final_action=actions[1],
            final_result=episode_result,
        )

    '''print(f'Rewards: {rewards}')
    print(f'Hit rate (AVG) = {env.hit_count/(env.hit_count+env.strike_out_count+env.hit_out_count)}')
    print(f'On base percentage (OBP) = {(env.walk_count+env.hit_count)/(env.walk_count+env.hit_count+env.strike_out_count+env.hit_out_count)}')

    print(f'hit and score {env.hit_count}')
    print(f'strike out {env.strike_out_count}')
    print(f'hit and out {env.hit_out_count} ')
    print(f'walk count {env.walk_count}')'''

    return {
        'rewards': rewards,
        'win_records': win_records,
        'policy_preds': policy_preds,
        'step_0_policy_preds': step_0_policy_preds,
        'step_1_policy_preds': step_1_policy_preds,
        'step_2_policy_preds': step_2_policy_preds,
        'step_3_policy_preds': step_3_policy_preds,
        'step_4_policy_preds': step_4_policy_preds,
        'step_5_policy_preds': step_5_policy_preds,
        'step_6_policy_preds': step_6_policy_preds,
        'hit_count': env.hit_count,
        'strike_out_count': env.strike_out_count,
        'hit_out_count': env.hit_out_count,
        'walk_count': env.walk_count,
    }


def run_bsi(args: argparse.Namespace, **kwargs) -> Dict:
    np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
    setup_initial_policy(args)
    # setup the agent
    agent = BsiAgent(**kwargs)
    agent.policy = BprAgent.Policy(args.agent_policy)
    env = setup_environment(args, agent)

    # agent.set_observation_model(env=env)
    rewards = []
    phi_beliefs = []
    win_records = []
    policy_preds = []
    step_0_policy_preds = []  # indicates whether the agent can choose the write policy before the episode starts
    step_1_policy_preds = []
    step_2_policy_preds = []
    step_3_policy_preds = []
    step_4_policy_preds = []
    step_5_policy_preds = []
    step_6_policy_preds = []
    kl_divergences = {
        'rel_entr(real, belief)': [],
        'rel_entr(belief, real)': [],
        'kl_div(real, belief)': [],
        'kl_div(belief, real)': [],
    }
    for i in range(args.num_episodes):

        if i <= 1:
            agent.update_policy()

        '''print(f'\n========== Start episode {i+1} ==========')
        print(f'Agent Phi belief is {agent.phi_belief}')
        print(f'Agent initial belief is {agent.belief}')
        print(f'Agent policy is {agent.policy}, Opponent policy is {env.opponent.policy}')'''

        phi_beliefs.append(agent.phi_belief)

        # policy prediction accuracy of the first two episodes is not applicable
        if np.all(agent.belief == agent.belief[0]):  # agent's belief is still uniform
            if random.randint(1, agent.n_policies) == env.opponent.policy.value:
                step_0_policy_preds.append(True)
            else:
                step_0_policy_preds.append(False)
        else:
            if np.argmax(agent.belief) + 1 == env.opponent.policy.value:
                step_0_policy_preds.append(True)
            else:
                step_0_policy_preds.append(False)

        total_reward = 0
        env.reset()
        while True:
            # print(f'--- step {env.steps} ---')
            # print(f'current state [strike, ball] = {env.state}')
            done, reward, state_, actions, episode_result = env.step(agent.get_action(env.state))
            total_reward += reward

            if args.print_map:
                # print('----- MAP ------')
                env.show(actions)
                # print('----- MAP ------')

            if args.print_action:
                print(f'(agent action, opponent action) is {actions}')

            # add the state-action pair of the opponent to experience queue (action of opponent is the observable o')
            agent.add_experience_queue(env.state, actions[1])

            if done:
                if reward > 0:
                    win_records.append(True)
                else:
                    win_records.append(False)
                rewards.append(total_reward)

                # print(f'episode result: {episode_result}, episode reward: {total_reward}')

                break

            env.state = state_

            if (
                not args.bpr_opponent
                and not args.phi_opponent
                and not args.new_phi_noise_opponent
                and args.episode_reset < 0
            ):  # intra episode random switch opponent
                opponent_update_policy(
                    args=args,
                    env=env,
                    episode=i,
                    final_reward=total_reward,
                    final_action=actions[1],
                    final_result=episode_result,
                )

        # policy prediction accuracy of the first two episodes is not applicable
        if np.all(agent.belief == agent.belief[0]):
            if random.randint(1, agent.n_policies) == env.opponent.policy.value:
                policy_preds.append(True)
            else:
                policy_preds.append(False)
        else:
            if np.argmax(agent.belief) + 1 == env.opponent.policy.value:
                policy_preds.append(True)
            else:
                policy_preds.append(False)

        # if i == 0:
        # print(f'BSI chooses policy base on uniform belief at the biginning of episode 2')
        # agent.update_policy()

        append_kl_divergences(kl_divergences, real_policy=env.opponent.policy.value, belief=agent.belief)
        agent.add_terminal_state_queue(state_ + list(actions[1]) + [reward])
        agent.update_phi()
        agent.infer_tau()

        agent.update_policy()

        agent.clear_experience_queue()  # empty the queue

        # opponent_update_policy(args=args, env=env, episode=i, final_reward=total_reward, ternimal_state=(state_[:2], state_[2:4]))
        opponent_update_policy(
            args=args,
            env=env,
            episode=i,
            final_reward=total_reward,
            final_action=actions[1],
            final_result=episode_result,
        )

    '''print(f'Rewards: {rewards}')
    print(f'Hit rate (AVG) = {env.hit_count/(env.hit_count+env.strike_out_count+env.hit_out_count)}')
    print(f'On base percentage (OBP) = {(env.walk_count+env.hit_count)/(env.walk_count+env.hit_count+env.strike_out_count+env.hit_out_count)}')

    print(f'hit and score {env.hit_count}')
    print(f'strike out {env.strike_out_count}')
    print(f'hit and out {env.hit_out_count} ')
    print(f'walk count {env.walk_count}')'''

    corresponding_phi = None
    # if isinstance(env.opponent, NewPhiOpponent) or isinstance(env.opponent, NewPhiNoiseOpponent):
    #     corresponding_phi = env.opponent.corresponding_phi

    return {
        'rewards': rewards,
        'phi_beliefs': phi_beliefs,
        'win_records': win_records,
        'kl_divergences': kl_divergences,
        'policy_preds': policy_preds,
        'step_0_policy_preds': step_0_policy_preds,
        'step_1_policy_preds': step_1_policy_preds,
        'step_2_policy_preds': step_2_policy_preds,
        'step_3_policy_preds': step_3_policy_preds,
        'step_4_policy_preds': step_4_policy_preds,
        'step_5_policy_preds': step_5_policy_preds,
        'step_6_policy_preds': step_6_policy_preds,
        'corresponding_phi': corresponding_phi,
        'hit_count': env.hit_count,
        'strike_out_count': env.strike_out_count,
        'hit_out_count': env.hit_out_count,
        'walk_count': env.walk_count,
    }


def run_bsi_pt(args: argparse.Namespace, **kwargs) -> Dict:
    np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
    setup_initial_policy(args)

    # setup the agent
    agent = BsiPtAgent(**kwargs)
    agent.policy = BprAgent.Policy(args.agent_policy)

    env = setup_environment(args, agent)
    # agent.set_observation_model(env=env)

    rewards = []
    phi_beliefs = []
    win_records = []
    policy_preds = []
    step_0_policy_preds = []  # indicates whether the agent can choose the write policy before the episode starts
    step_1_policy_preds = []
    step_2_policy_preds = []
    step_3_policy_preds = []
    step_4_policy_preds = []
    step_5_policy_preds = []
    step_6_policy_preds = []
    kl_divergences = {
        'rel_entr(real, belief)': [],
        'rel_entr(belief, real)': [],
        'kl_div(real, belief)': [],
        'kl_div(belief, real)': [],
    }
    for i in range(args.num_episodes):

        if i <= 1:
            agent.update_policy()

        '''print(f'\n========== Start episode {i+1} ==========')
        print(f'Agent Phi belief is {agent.phi_belief}')
        print(f'Agent initial intra belief is {agent.belief}')
        print(f'Agent policy is {agent.policy}, Opponent policy is {env.opponent.policy}')'''

        phi_beliefs.append(agent.phi_belief)

        # policy prediction accuracy of the first two episodes is not applicable
        if np.all(agent.intra_belief == agent.intra_belief[0]):  # agent's belief is still uniform
            if random.randint(1, agent.n_policies) == env.opponent.policy.value:
                step_0_policy_preds.append(True)
            else:
                step_0_policy_preds.append(False)
        else:
            if np.argmax(agent.intra_belief) + 1 == env.opponent.policy.value:
                step_0_policy_preds.append(True)
            else:
                step_0_policy_preds.append(False)

        total_reward = 0
        env.reset()
        while True:
            # print(f'--- step {env.steps} ---')
            # print(f'current state [strike, ball] = {env.state}')

            done, reward, state_, actions, episode_result = env.step(agent.get_action(env.state))
            total_reward += reward

            if args.print_map:
                # print('----- MAP ------')
                env.show(actions)
                # print('----- MAP ------')

            if args.print_action:
                print(f'(agent action, opponent action) is {actions}')

            # only add the opponent's location and action o' to the experience queue
            agent.add_experience_queue(env.state, actions[1])

            agent.update_intra_belief()

            # compute the current integrated belief
            agent.update_policy(integrated_belief=False)

            if env.steps == 1:
                if np.argmax(agent.intra_belief) + 1 == env.opponent.policy.value:
                    step_1_policy_preds.append(True)
                else:
                    step_1_policy_preds.append(False)
            elif env.steps == 2:
                if np.argmax(agent.intra_belief) + 1 == env.opponent.policy.value:
                    step_2_policy_preds.append(True)
                else:
                    step_2_policy_preds.append(False)
            elif env.steps == 3:
                if np.argmax(agent.intra_belief) + 1 == env.opponent.policy.value:
                    step_3_policy_preds.append(True)
                else:
                    step_3_policy_preds.append(False)
            elif env.steps == 4:
                if np.argmax(agent.intra_belief) + 1 == env.opponent.policy.value:
                    step_4_policy_preds.append(True)
                else:
                    step_4_policy_preds.append(False)
            elif env.steps == 5:
                if np.argmax(agent.intra_belief) + 1 == env.opponent.policy.value:
                    step_5_policy_preds.append(True)
                else:
                    step_5_policy_preds.append(False)
            elif env.steps == 6:
                if np.argmax(agent.intra_belief) + 1 == env.opponent.policy.value:
                    step_6_policy_preds.append(True)
                else:
                    step_6_policy_preds.append(False)

            if done:
                if reward > 0:
                    win_records.append(True)
                else:
                    win_records.append(False)
                rewards.append(total_reward)
                '''if env.steps < 6:
                    step_6_policy_preds.append(None)
                if env.steps < 5:
                    step_5_policy_preds.append(None)
                if env.steps < 4:
                    step_4_policy_preds.append(None)
                if env.steps < 3:
                    step_3_policy_preds.append(None)
                if env.steps < 2:
                    step_2_policy_preds.append(None)
                if env.steps < 1:
                    step_1_policy_preds.append(None)'''

                # print(f'episode result: {episode_result}, episode reward: {total_reward}')

                break
            env.state = state_
            if (
                not args.bpr_opponent
                and not args.phi_opponent
                and not args.new_phi_noise_opponent
                and args.episode_reset < 0
            ):  # intra episode random switch opponent
                opponent_update_policy(
                    args=args,
                    env=env,
                    episode=i,
                    final_reward=total_reward,
                    final_action=actions[1],
                    final_result=episode_result,
                )

            # print(f'(Step {env.steps}) Intra-belief is updated to {agent.intra_belief}')
            # print(f'(Step {env.steps}) Agent policy is {agent.policy}, Opponent policy is {env.opponent.policy}')

        # print(step_0_policy_preds,step_1_policy_preds)
        # policy prediction accuracy of the first two episodes is not applicable
        '''if np.all(agent.intra_belief == agent.intra_belief[0]):
            if random.randint(1, agent.n_policies) == env.opponent.policy.value:
                policy_preds.append(True)
            else:
                policy_preds.append(False)
        else:
            if np.argmax(agent.intra_belief) + 1 == env.opponent.policy.value:
                policy_preds.append(True)
            else:
                policy_preds.append(False)'''

        if np.all(agent.intra_belief == agent.intra_belief[0]):
            if random.randint(1, agent.n_policies) == env.opponent.policy.value:
                policy_preds.append(True)
            else:
                policy_preds.append(False)
        else:
            if env.steps == 1:
                policy_preds.append(step_0_policy_preds[i])
                step_2_policy_preds.append(step_1_policy_preds[i])
                step_3_policy_preds.append(step_1_policy_preds[i])
                step_4_policy_preds.append(step_1_policy_preds[i])
                step_5_policy_preds.append(step_1_policy_preds[i])
                step_6_policy_preds.append(step_1_policy_preds[i])
            elif env.steps == 2:
                policy_preds.append(step_1_policy_preds[i])
                step_3_policy_preds.append(step_2_policy_preds[i])
                step_4_policy_preds.append(step_2_policy_preds[i])
                step_5_policy_preds.append(step_2_policy_preds[i])
                step_6_policy_preds.append(step_2_policy_preds[i])
            elif env.steps == 3:
                policy_preds.append(step_2_policy_preds[i])
                step_4_policy_preds.append(step_3_policy_preds[i])
                step_5_policy_preds.append(step_3_policy_preds[i])
                step_6_policy_preds.append(step_3_policy_preds[i])
            elif env.steps == 4:
                policy_preds.append(step_3_policy_preds[i])
                step_5_policy_preds.append(step_4_policy_preds[i])
                step_6_policy_preds.append(step_4_policy_preds[i])
            elif env.steps == 5:
                policy_preds.append(step_4_policy_preds[i])
                step_6_policy_preds.append(step_5_policy_preds[i])
            elif env.steps == 6:
                policy_preds.append(step_5_policy_preds[i])

        # if i == 0:
        # print(f'BSI choose policy base on uniform belief at the biginning of episode 2')
        # agent.update_policy()

        append_kl_divergences(kl_divergences, real_policy=env.opponent.policy.value, belief=agent.intra_belief)
        agent.add_terminal_state_queue(state_ + list(actions[1]) + [reward])
        agent.update_phi()
        agent.infer_tau()

        agent.update_policy()

        agent.clear_experience_queue()  # empty the queue

        # opponent_update_policy(args=args, env=env, episode=i, reward=total_reward, ternimal_state=(state_[:2], state_[2:4]))
        opponent_update_policy(
            args=args,
            env=env,
            episode=i,
            final_reward=total_reward,
            final_action=actions[1],
            final_result=episode_result,
        )

    '''print(f'Rewards: {rewards}')
    print(f'Hit rate (AVG) = {env.hit_count/(env.hit_count+env.strike_out_count+env.hit_out_count)}')
    print(f'On base percentage (OBP) = {(env.walk_count+env.hit_count)/(env.walk_count+env.hit_count+env.strike_out_count+env.hit_out_count)}')

    print(f'hit and score {env.hit_count}')
    print(f'strike out {env.strike_out_count}')
    print(f'hit and out {env.hit_out_count} ')
    print(f'walk count {env.walk_count}')'''

    corresponding_phi = None
    # if isinstance(env.opponent, NewPhiOpponent) or isinstance(env.opponent, NewPhiNoiseOpponent):
    #     corresponding_phi = env.opponent.corresponding_phi

    return {
        'rewards': rewards,
        'phi_beliefs': phi_beliefs,
        'win_records': win_records,
        'kl_divergences': kl_divergences,
        'policy_preds': policy_preds,
        'step_0_policy_preds': step_0_policy_preds,
        'step_1_policy_preds': step_1_policy_preds,
        'step_2_policy_preds': step_2_policy_preds,
        'step_3_policy_preds': step_3_policy_preds,
        'step_4_policy_preds': step_4_policy_preds,
        'step_5_policy_preds': step_5_policy_preds,
        'step_6_policy_preds': step_6_policy_preds,
        'corresponding_phi': corresponding_phi,
        'hit_count': env.hit_count,
        'strike_out_count': env.strike_out_count,
        'hit_out_count': env.hit_out_count,
        'walk_count': env.walk_count,
    }


def run_uniformA(args: argparse.Namespace, **kwargs) -> Dict:
    np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
    setup_initial_policy(args)

    # setup the agent
    agent = BsiPtAgent(**kwargs)
    agent.policy = BprAgent.Policy(args.agent_policy)

    env = setup_environment(args, agent)
    # agent.set_observation_model(env=env)

    rewards = []
    phi_beliefs = []
    win_records = []
    policy_preds = []
    step_0_policy_preds = []  # indicates whether the agent can choose the write policy before the episode starts
    step_1_policy_preds = []
    step_2_policy_preds = []
    step_3_policy_preds = []
    step_4_policy_preds = []
    step_5_policy_preds = []
    step_6_policy_preds = []
    kl_divergences = {
        'rel_entr(real, belief)': [],
        'rel_entr(belief, real)': [],
        'kl_div(real, belief)': [],
        'kl_div(belief, real)': [],
    }
    for i in range(args.num_episodes):

        agent.infer_tau2()
        agent.update_policy()

        '''print(f'\n========== Start episode {i+1} ==========')
        print(f'Agent Phi belief is {agent.phi_belief}')
        print(f'Agent initial intra belief is {agent.belief}')
        print(f'Agent policy is {agent.policy}, Opponent policy is {env.opponent.policy}')
        
        print('11111',agent.intra_belief)'''

        phi_beliefs.append(agent.phi_belief)

        # policy prediction accuracy of the first two episodes is not applicable
        if np.all(agent.intra_belief == agent.intra_belief[0]):  # agent's belief is still uniform
            if random.randint(1, agent.n_policies) == env.opponent.policy.value:
                step_0_policy_preds.append(True)
            else:
                step_0_policy_preds.append(False)
        else:
            if np.argmax(agent.intra_belief) + 1 == env.opponent.policy.value:
                step_0_policy_preds.append(True)
            else:
                step_0_policy_preds.append(False)

        total_reward = 0
        env.reset()
        while True:
            # print(f'--- step {env.steps} ---')
            # print(f'current state [strike, ball] = {env.state}')

            done, reward, state_, actions, episode_result = env.step(agent.get_action(env.state))
            total_reward += reward

            if args.print_map:
                # print('----- MAP ------')
                env.show(actions)
                # print('----- MAP ------')

            if args.print_action:
                print(f'(agent action, opponent action) is {actions}')

            # only add the opponent's location and action o' to the experience queue
            agent.add_experience_queue(env.state, actions[1])

            agent.update_intra_belief()

            # compute the current integrated belief
            agent.update_policy(integrated_belief=False)

            if env.steps == 1:
                if np.argmax(agent.intra_belief) + 1 == env.opponent.policy.value:
                    step_1_policy_preds.append(True)
                else:
                    step_1_policy_preds.append(False)
            elif env.steps == 2:
                if np.argmax(agent.intra_belief) + 1 == env.opponent.policy.value:
                    step_2_policy_preds.append(True)
                else:
                    step_2_policy_preds.append(False)
            elif env.steps == 3:
                if np.argmax(agent.intra_belief) + 1 == env.opponent.policy.value:
                    step_3_policy_preds.append(True)
                else:
                    step_3_policy_preds.append(False)
            elif env.steps == 4:
                if np.argmax(agent.intra_belief) + 1 == env.opponent.policy.value:
                    step_4_policy_preds.append(True)
                else:
                    step_4_policy_preds.append(False)
            elif env.steps == 5:
                if np.argmax(agent.intra_belief) + 1 == env.opponent.policy.value:
                    step_5_policy_preds.append(True)
                else:
                    step_5_policy_preds.append(False)
            elif env.steps == 6:
                if np.argmax(agent.intra_belief) + 1 == env.opponent.policy.value:
                    step_6_policy_preds.append(True)
                else:
                    step_6_policy_preds.append(False)

            if done:
                if reward > 0:
                    win_records.append(True)
                else:
                    win_records.append(False)
                rewards.append(total_reward)
                '''if env.steps < 6:
                    step_6_policy_preds.append(None)
                if env.steps < 5:
                    step_5_policy_preds.append(None)
                if env.steps < 4:
                    step_4_policy_preds.append(None)
                if env.steps < 3:
                    step_3_policy_preds.append(None)
                if env.steps < 2:
                    step_2_policy_preds.append(None)
                if env.steps < 1:
                    step_1_policy_preds.append(None)'''

                # print(f'episode result: {episode_result}, episode reward: {total_reward}')

                break
            env.state = state_
            if (
                not args.bpr_opponent
                and not args.phi_opponent
                and not args.new_phi_noise_opponent
                and args.episode_reset < 0
            ):  # intra episode random switch opponent
                opponent_update_policy(
                    args=args,
                    env=env,
                    episode=i,
                    final_reward=total_reward,
                    final_action=actions[1],
                    final_result=episode_result,
                )

            # print(f'(Step {env.steps}) Intra-belief is updated to {agent.intra_belief}')
            # print(f'(Step {env.steps}) Agent policy is {agent.policy}, Opponent policy is {env.opponent.policy}')

        # print(step_0_policy_preds,step_1_policy_preds)
        # policy prediction accuracy of the first two episodes is not applicable
        '''if np.all(agent.intra_belief == agent.intra_belief[0]):
            if random.randint(1, agent.n_policies) == env.opponent.policy.value:
                policy_preds.append(True)
            else:
                policy_preds.append(False)
        else:
            if np.argmax(agent.intra_belief) + 1 == env.opponent.policy.value:
                policy_preds.append(True)
            else:
                policy_preds.append(False)'''

        if np.all(agent.intra_belief == agent.intra_belief[0]):
            if random.randint(1, agent.n_policies) == env.opponent.policy.value:
                policy_preds.append(True)
            else:
                policy_preds.append(False)
        else:
            if env.steps == 1:
                policy_preds.append(step_0_policy_preds[i])
                step_2_policy_preds.append(step_1_policy_preds[i])
                step_3_policy_preds.append(step_1_policy_preds[i])
                step_4_policy_preds.append(step_1_policy_preds[i])
                step_5_policy_preds.append(step_1_policy_preds[i])
                step_6_policy_preds.append(step_1_policy_preds[i])
            elif env.steps == 2:
                policy_preds.append(step_1_policy_preds[i])
                step_3_policy_preds.append(step_2_policy_preds[i])
                step_4_policy_preds.append(step_2_policy_preds[i])
                step_5_policy_preds.append(step_2_policy_preds[i])
                step_6_policy_preds.append(step_2_policy_preds[i])
            elif env.steps == 3:
                policy_preds.append(step_2_policy_preds[i])
                step_4_policy_preds.append(step_3_policy_preds[i])
                step_5_policy_preds.append(step_3_policy_preds[i])
                step_6_policy_preds.append(step_3_policy_preds[i])
            elif env.steps == 4:
                policy_preds.append(step_3_policy_preds[i])
                step_5_policy_preds.append(step_4_policy_preds[i])
                step_6_policy_preds.append(step_4_policy_preds[i])
            elif env.steps == 5:
                policy_preds.append(step_4_policy_preds[i])
                step_6_policy_preds.append(step_5_policy_preds[i])
            elif env.steps == 6:
                policy_preds.append(step_5_policy_preds[i])

        '''append_kl_divergences(kl_divergences, real_policy=env.opponent.policy.value, belief=agent.intra_belief)
        agent.add_terminal_state_queue(state_+list(actions[1])+[reward])
        agent.update_phi()
        agent.infer_tau()

        agent.update_policy()'''

        agent.clear_experience_queue()  # empty the queue

        # opponent_update_policy(args=args, env=env, episode=i, reward=total_reward, ternimal_state=(state_[:2], state_[2:4]))
        opponent_update_policy(
            args=args,
            env=env,
            episode=i,
            final_reward=total_reward,
            final_action=actions[1],
            final_result=episode_result,
        )

    '''print(f'Rewards: {rewards}')
    print(f'Hit rate (AVG) = {env.hit_count/(env.hit_count+env.strike_out_count+env.hit_out_count)}')
    print(f'On base percentage (OBP) = {(env.walk_count+env.hit_count)/(env.walk_count+env.hit_count+env.strike_out_count+env.hit_out_count)}')

    print(f'hit and score {env.hit_count}')
    print(f'strike out {env.strike_out_count}')
    print(f'hit and out {env.hit_out_count} ')
    print(f'walk count {env.walk_count}')'''

    corresponding_phi = None
    # if isinstance(env.opponent, NewPhiOpponent) or isinstance(env.opponent, NewPhiNoiseOpponent):
    #     corresponding_phi = env.opponent.corresponding_phi

    return {
        'rewards': rewards,
        'phi_beliefs': phi_beliefs,
        'win_records': win_records,
        'kl_divergences': kl_divergences,
        'policy_preds': policy_preds,
        'step_0_policy_preds': step_0_policy_preds,
        'step_1_policy_preds': step_1_policy_preds,
        'step_2_policy_preds': step_2_policy_preds,
        'step_3_policy_preds': step_3_policy_preds,
        'step_4_policy_preds': step_4_policy_preds,
        'step_5_policy_preds': step_5_policy_preds,
        'step_6_policy_preds': step_6_policy_preds,
        'corresponding_phi': corresponding_phi,
        'hit_count': env.hit_count,
        'strike_out_count': env.strike_out_count,
        'hit_out_count': env.hit_out_count,
        'walk_count': env.walk_count,
    }


def setup_environment(args: argparse.Namespace, agent: BprAgent) -> BaseballGame:
    env = BaseballGame(
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
    env.opponent.PERFORMANCE_MODEL = performance_model
    agent.PERFORMANCE_MODEL = performance_model

    return env


def opponent_update_policy(
    args: argparse.Namespace,
    env: BaseballGame,
    episode: int,
    final_reward: int,
    final_action,  # opponent's actual final action o'
    final_result,
) -> None:
    if args.bpr_opponent:
        env.opponent.update_belief(final_reward)
        env.opponent.update_policy()
        print(f'Opponent belief updated to {env.opponent.belief}')
    elif args.phi_opponent or args.new_phi_noise_opponent:
        # given the final result (hit, out, BB) and the final action (o'), opponent can deternmine its next policy
        env.opponent.update_policy(final_action, final_result)
    elif args.episode_reset > 0 and (episode + 1) % args.episode_reset == 0:  # random switch opponent
        candidate = list(Opponent.Policy)
        candidate.remove(env.opponent.policy)
        env.opponent.policy = random.choice(candidate)
    elif (
        args.episode_reset < 0 and env.steps % (-args.episode_reset) == 0 and env.steps / (-args.episode_reset) == 1
    ):  # intra episode random switch opponent: # intra episode random switch
        candidate = list(Opponent.Policy)
        candidate.remove(env.opponent.policy)
        env.opponent.policy = random.choice(candidate)
        print(f'!!opponent intra switch to {env.opponent.policy} at step {env.steps}!!')
    elif args.episode_reset < 0 and final_result is not None:
        candidate = list(Opponent.Policy)
        candidate.remove(env.opponent.policy)
        env.opponent.policy = random.choice(candidate)
        print(f'intra switch opponent switch policy to {env.opponent.policy} when episode ends')


def setup_initial_policy(args: argparse.Namespace) -> None:
    if args.op_policy == -1:  # if user does not set it manually
        args.op_policy = random.randint(1, 4)

    if args.agent_policy == -1:  # if user does not set it manually
        args.agent_policy = random.randint(1, 4)

    if args.phi_opponent and args.phi <= 4:
        args.op_policy = args.phi  # the first four tau always use the same policy


def append_kl_divergences(kl_divergences: Dict[str, List], real_policy: int, belief: np.ndarray) -> None:
    PI_NUM = 4

    real_distribution = np.zeros(PI_NUM)
    real_distribution[real_policy - 1] = 1
    real_distribution = normalize_distribution(real_distribution, 0.001)
    kl_divergences['rel_entr(real, belief)'].append(sum(rel_entr(real_distribution, belief)))
    kl_divergences['rel_entr(belief, real)'].append(sum(rel_entr(belief, real_distribution)))
    kl_divergences['kl_div(real, belief)'].append(sum(kl_div(real_distribution, belief)))
    kl_divergences['kl_div(belief, real)'].append(sum(kl_div(belief, real_distribution)))
