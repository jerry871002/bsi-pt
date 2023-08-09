import argparse
import random
from typing import Dict

import numpy as np

from .agent import BprAgent, BprOkrAgent, BprPlusAgent, BsiAgent, BsiPtAgent, DeepBprPlusAgent
from .env import BaseballGame, Opponent, PhiOpponent


def run_bpr_plus(args: argparse.Namespace, **kwargs) -> Dict:
    # (maybe?) print win rate to the second decimal
    np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
    # set agent and opponent policy to a random number if user does not specify
    # also set the opponent policy of the phi opponent with the first four tau
    setup_initial_policy(args)

    # setup the agent
    agent = BprPlusAgent(**kwargs)
    agent.policy = BprAgent.Policy(args.agent_policy)

    env = setup_environment(args, agent)

    rewards = []
    win_records = []
    policy_preds = []
    for i in range(args.num_episodes):
        total_reward = 0
        env.reset()
        while True:
            done, reward, state_, actions, episode_result = env.step(agent.get_action(env.state))
            total_reward += reward

            if args.print_map:
                env.show(actions)

            if args.print_action:
                print(f'(agent action, opponent action) is {actions}')

            if done:  # the episode has finished
                if reward > 0:
                    win_records.append(True)
                else:
                    win_records.append(False)

                rewards.append(total_reward)
                break

            # change to the next state
            env.state = state_

            # intra episode random switch opponent can switch policy wtihin an episode
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
            # use random number as the prediction accuracy
            if random.randint(1, agent.n_policies) == env.opponent.policy.value:
                policy_preds.append(True)
            else:
                policy_preds.append(False)
        # after the second episode, record whether the policy with highest belief
        # is the opponent's actual policy
        else:
            if np.argmax(agent.belief) + 1 == env.opponent.policy.value:
                policy_preds.append(True)
            else:
                policy_preds.append(False)

        agent.update_belief(total_reward)  # update belief with episodic return
        agent.update_policy()

        # update opponent policy according to different op type
        opponent_update_policy(
            args=args,
            env=env,
            episode=i,
            final_reward=total_reward,
            final_action=actions[1],
            final_result=episode_result,
        )

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
    # set agent and opponent policy to a random number if user does not specify
    # also set the opponent policy of the phi opponent with the first four tau
    setup_initial_policy(args)

    # setup the agent
    agent = DeepBprPlusAgent(**kwargs)
    agent.policy = BprAgent.Policy(args.agent_policy)

    env = setup_environment(args, agent)

    rewards = []
    win_records = []
    policy_preds = []
    for i in range(args.num_episodes):
        total_reward = 0
        env.reset()
        while True:
            done, reward, state_, actions, episode_result = env.step(agent.get_action(env.state))
            total_reward += reward

            if args.print_map:
                env.show(actions)

            if args.print_action:
                print(f'(agent action, opponent action) is {actions}')

            if done:  # the episode has finished
                if reward > 0:
                    win_records.append(True)
                else:
                    win_records.append(False)

                rewards.append(total_reward)
                break

            # add the opponent's location and action to the experience queue
            agent.add_experience_queue(env.state, actions[1])
            # change to the next state
            env.state = state_

            # intra episode random switch opponent can switch policy wtihin an episode
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

        agent.compute_tau_hat()  # use opponent model + experience_queue to estimate tau hat
        agent.update_belief(total_reward)  # update belief with episodic return
        agent.update_policy()

        # update opponent policy according to different op type
        opponent_update_policy(
            args=args,
            env=env,
            episode=i,
            final_reward=total_reward,
            final_action=actions[1],
            final_result=episode_result,
        )

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
    # indicates whether the agent can choose the right policy before the episode starts
    step_0_policy_preds = []
    # record the prediction accuracy of each step as respective arrays
    step_1_policy_preds = []
    step_2_policy_preds = []
    step_3_policy_preds = []
    step_4_policy_preds = []
    step_5_policy_preds = []
    step_6_policy_preds = []
    for i in range(args.num_episodes):
        # choose the initial policy
        if i == 0:
            agent.update_policy()

        # record the accyracy of step 0 (before the episode starts)
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
            done, reward, state_, actions, episode_result = env.step(agent.get_action(env.state))
            total_reward += reward

            if args.print_map:
                env.show(actions)

            if args.print_action:
                print(f'(agent action, opponent action) is {actions}')

            # add the opponent's location and action to the experience queue
            agent.add_experience_queue(env.state, actions[1])
            agent.update_intra_belief()
            # choose the next policy using integrated belief (combination of inter & intra belief)
            agent.update_policy(integrated_belief=True)
            # switch to the next state
            env.state = state_

            # record the accuracy of each step
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
                break

            # intra episode random switch opponent can switch policy within episodes
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

        # use the intra belief of the last episode as the new inter belief
        agent.belief = agent.intra_belief
        agent.clear_experience_queue()  # empty the queue
        agent.reset_intra_belief()  # assign the inter-epsode belief to the intra-episode beleif
        agent.update_policy()

        opponent_update_policy(
            args=args,
            env=env,
            episode=i,
            final_reward=total_reward,
            final_action=actions[1],
            final_result=episode_result,
        )

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

    rewards = []
    phi_beliefs = []
    win_records = []
    policy_preds = []
    # indicates whether the agent can choose the right policy before the episode starts
    step_0_policy_preds = []
    for i in range(args.num_episodes):
        # need at least two episodes to calculate the first phi belief
        # don't neet to update phi at the first two episodes
        if i <= 1:
            agent.update_policy()

        phi_beliefs.append(agent.phi_belief)

        # policy prediction accuracy of the first two episodes is not applicable
        # FIXME: this condition doesn't mean the distribution is uniform
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
            done, reward, state_, actions, episode_result = env.step(agent.get_action(env.state))
            total_reward += reward

            if args.print_map:
                env.show(actions)

            if args.print_action:
                print(f'(agent action, opponent action) is {actions}')

            # add the state-action pair of the opponent to experience queue
            # (action of opponent is the observable o')
            agent.add_experience_queue(env.state, actions[1])

            if done:
                if reward > 0:
                    win_records.append(True)
                else:
                    win_records.append(False)
                rewards.append(total_reward)

                break
            # switch to the next state
            env.state = state_

            # intra episode random switch opponent can switch policy within episodes
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

        agent.add_terminal_state_queue(state_ + list(actions[1]) + [reward])
        agent.update_phi()
        agent.infer_tau()
        agent.update_policy()
        agent.clear_experience_queue()  # empty the queue
        opponent_update_policy(
            args=args,
            env=env,
            episode=i,
            final_reward=total_reward,
            final_action=actions[1],
            final_result=episode_result,
        )

    corresponding_phi = None

    return {
        'rewards': rewards,
        'phi_beliefs': phi_beliefs,
        'win_records': win_records,
        'policy_preds': policy_preds,
        'step_0_policy_preds': step_0_policy_preds,
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

    rewards = []
    phi_beliefs = []
    win_records = []
    policy_preds = []
    # indicates whether the agent can choose the right policy before the episode starts
    step_0_policy_preds = []
    # record the prediction accuracy of each step as respective arrays
    step_1_policy_preds = []
    step_2_policy_preds = []
    step_3_policy_preds = []
    step_4_policy_preds = []
    step_5_policy_preds = []
    step_6_policy_preds = []
    for i in range(args.num_episodes):
        # need at least two episodes to calculate the first phi belief
        # don't neet to update phi at the first two episodes
        if i <= 1:
            agent.update_policy()

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
            done, reward, state_, actions, episode_result = env.step(agent.get_action(env.state))
            total_reward += reward
            if args.print_map:
                env.show(actions)
            if args.print_action:
                print(f'(agent action, opponent action) is {actions}')

            # add the opponent's location and action o' to the experience queue
            agent.add_experience_queue(env.state, actions[1])
            # bsi-pt updates the intra_belief after every step in an episode
            agent.update_intra_belief()

            # compute the current integrated belief
            agent.update_policy(integrated_belief=False)

            # check the accuracy of each step according to the intra belief
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
                break

            # switch to the next state
            env.state = state_

            # intra episode random switch opponent can switch policy within episodes
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

        agent.add_terminal_state_queue(state_ + list(actions[1]) + [reward])
        agent.update_phi()
        agent.infer_tau()
        agent.update_policy()
        agent.clear_experience_queue()  # empty the queue
        opponent_update_policy(
            args=args,
            env=env,
            episode=i,
            final_reward=total_reward,
            final_action=actions[1],
            final_result=episode_result,
        )
    corresponding_phi = None
    return {
        'rewards': rewards,
        'phi_beliefs': phi_beliefs,
        'win_records': win_records,
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

    rewards = []
    win_records = []
    policy_preds = []
    # indicates whether the agent can choose the right policy before the episode starts
    step_0_policy_preds = []
    # record the prediction accuracy of each step as respective arrays
    step_1_policy_preds = []
    step_2_policy_preds = []
    step_3_policy_preds = []
    step_4_policy_preds = []
    step_5_policy_preds = []
    step_6_policy_preds = []
    for i in range(args.num_episodes):
        # use uniform tau as inital intra belief at the beginning of each episode
        agent.infer_tau2()
        agent.update_policy()

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
            done, reward, state_, actions, episode_result = env.step(agent.get_action(env.state))
            total_reward += reward

            if args.print_map:
                env.show(actions)

            if args.print_action:
                print(f'(agent action, opponent action) is {actions}')

            # add the opponent's location and action o' to the experience queue
            agent.add_experience_queue(env.state, actions[1])

            agent.update_intra_belief()

            # compute the current integrated belief
            agent.update_policy(integrated_belief=False)

            # check the accuracy of each step according to the intra belief
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
                break

            # switch to the next state
            env.state = state_

            # intra episode random switch opponent can switch policy within episodes
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

        '''
        agent.add_terminal_state_queue(state_+list(actions[1])+[reward])
        agent.update_phi()
        agent.infer_tau()

        agent.update_policy()'''

        agent.clear_experience_queue()  # empty the queue
        opponent_update_policy(
            args=args,
            env=env,
            episode=i,
            final_reward=total_reward,
            final_action=actions[1],
            final_result=episode_result,
        )
    corresponding_phi = None
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
        # given the final result (hit, out, BB) and the final action (o')
        # opponent can deternmine its next policy
        env.opponent.update_policy(final_action, final_result)
    elif (
        args.episode_reset > 0 and (episode + 1) % args.episode_reset == 0
    ):  # random switch opponent
        candidate = list(Opponent.Policy)
        candidate.remove(env.opponent.policy)
        env.opponent.policy = random.choice(candidate)
    elif (
        args.episode_reset < 0
        and env.steps % (-args.episode_reset) == 0
        and env.steps / (-args.episode_reset) == 1
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
