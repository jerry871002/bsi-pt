from pathlib import Path
from typing import List, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils import load_pickle


def plot_episodic_rewards(
    pickle_file: Union[Path, str],
    save_fig: bool = False,
    filename: Optional[Union[Path, str]] = None,
    show_fig: bool = False,
) -> pd.DataFrame:
    if save_fig and filename is None:
        raise RuntimeError('Please provide a filename to save the figure')

    data = load_pickle(pickle_file)

    df = pd.DataFrame()

    plt.figure(figsize=(15, 6))
    for agent, results in data.items():
        rewards = []
        for result in results:
            rewards.append(result['rewards'])

        avg_reward = np.average(rewards, axis=0)

        plt.plot(avg_reward, '--', label=agent)

        df[agent] = avg_reward

    plt.legend()
    plt.title(f'Episodic Rewards: {pickle_file}')
    plt.xlabel('Episodes')
    plt.ylabel('Episodic Rewards')

    if save_fig:
        plt.savefig(filename)

    if show_fig:
        plt.show()

    return df


def plot_cumulative_rewards(
    pickle_file: Union[Path, str],
    save_fig: bool = False,
    filename: Optional[Union[Path, str]] = None,
    show_fig: bool = False,
) -> pd.DataFrame:
    if save_fig and filename is None:
        raise RuntimeError('Please provide a filename to save the figure')

    data = load_pickle(pickle_file)

    df = pd.DataFrame()

    plt.figure(figsize=(15, 6))
    for agent, results in data.items():
        rewards = []
        for result in results:
            rewards.append(result['rewards'])

        avg_reward = np.average(rewards, axis=0)
        cumulative_reward = np.cumsum(avg_reward)

        plt.plot(cumulative_reward, '--', label=agent)

        df[agent] = cumulative_reward

    plt.legend()
    plt.title(f'Cumulative Rewards: {pickle_file}')
    plt.xlabel('Episodes')
    plt.ylabel('Cumulative Rewards')

    if save_fig:
        plt.savefig(filename)

    if show_fig:
        plt.show()

    return df


def plot_phi_beliefs(
    pickle_file: Union[Path, str],
    agent: str,
    save_fig: bool = False,
    filename: Optional[Union[Path, str]] = None,
    show_fig: bool = False,
) -> pd.DataFrame:
    if save_fig and filename is None:
        raise RuntimeError('Please provide a filename to save the figure')

    data = load_pickle(pickle_file)

    results = data[agent]

    phi_beliefs = []
    for result in results:
        phi_beliefs.append(result['phi_beliefs'])

    avg_phi_belief = np.average(phi_beliefs, axis=0).transpose()

    df = pd.DataFrame()

    plt.figure(figsize=(15, 6))
    for i, phi_belief in enumerate(avg_phi_belief, start=1):
        plt.plot(phi_belief, '--', label=f'phi_{i}')

        df[f'phi_{i}'] = phi_belief

    plt.legend()
    plt.title(f'Phi Belief: {pickle_file} ({agent})')
    plt.xlabel('Episodes')
    plt.ylabel('Phi Belief')
    plt.ylim([0, 1.1])

    if save_fig:
        plt.savefig(filename)

    if show_fig:
        plt.show()

    return df


def plot_specific_phi_beliefs(
    pickle_files: Sequence[Union[Path, str]],
    labels: Sequence[str],
    agent: str,
    phi_num: int,
    save_fig: bool = False,
    filename: Optional[Union[Path, str]] = None,
    show_fig: bool = False,
) -> pd.DataFrame:
    if save_fig and filename is None:
        raise RuntimeError('Please provide a filename to save the figure')

    if len(pickle_files) != len(labels):
        raise ValueError(
            'Please provide the same number of pickle files and labels, '
            f'invalid: {len(pickle_files)} pickle files and {len(labels)} labels'
        )

    df = pd.DataFrame()

    plt.figure(figsize=(15, 6))
    for pickle_file, label in zip(pickle_files, labels):
        data = load_pickle(pickle_file)
        results = data[agent]

        phi_beliefs = []
        for result in results:
            phi_beliefs.append(result['phi_beliefs'])

        phi_belief = np.average(phi_beliefs, axis=0).transpose()[phi_num - 1]
        plt.plot(phi_belief, '--', label=label)

        df[label] = phi_belief

    plt.legend()
    plt.title(f'Phi {phi_num} Belief ({agent})')
    plt.xlabel('Episodes')
    plt.ylabel(f'Phi {phi_num} Belief')
    plt.ylim([0, 1.1])

    if save_fig:
        plt.savefig(filename)

    if show_fig:
        plt.show()

    return df


def plot_win_rates(
    pickle_dir: Union[Path, str],
    num_runs: int,
    num_episodes: int,
    agent: str,
    save_fig: bool = False,
    filename: Optional[Union[Path, str]] = None,
    show_fig: bool = False,
) -> pd.DataFrame:
    if save_fig and filename is None:
        raise RuntimeError('Please provide a filename to save the figure')

    df = pd.DataFrame()

    plt.figure(figsize=(15, 6))
    for i in range(1, 12):
        pickle_file = Path(pickle_dir) / f'op_{i}_phi_{num_runs}_runs_{num_episodes}_episodes.pkl'

        try:
            win_rates = collect_win_rate(pickle_file, agent)
        except FileNotFoundError:
            print(f'File for phi {i} does not exist, ignore in win rate plot')
            continue

        plt.plot(win_rates, '--', label=f'phi_{i}')

        df[f'phi_{i}'] = win_rates

    plt.legend()
    plt.title(f'Win Rate Against Different Phi Opponent ({agent})')
    plt.xlabel('Episodes')
    plt.ylabel('Win Rate')
    plt.ylim([0, 1.1])

    if save_fig:
        plt.savefig(filename)

    if show_fig:
        plt.show()

    return df


def plot_policy_pred_acc(
    pickle_file: Union[Path, str],
    agent: str,
    save_fig: bool = False,
    filename: Optional[Union[Path, str]] = None,
    show_fig: bool = False,
) -> pd.DataFrame:
    """
    Plot the policy prediction accuracy (without intra markers) of an agent.
    This function is specifically for BSI-PT and Bayes-OKR in the baseball environment.

    x-axis: episodes
    y-axis: policy prediction accuracy
    """
    if save_fig and filename is None:
        raise RuntimeError('Please provide a filename to save the figure')

    df = pd.DataFrame()
    policy_preds = collect_policy_prediction(pickle_file, agent)
    plt.figure(figsize=(15, 6))
    plt.plot(policy_preds, '--')
    df['policy_pred'] = policy_preds

    plt.legend()
    plt.title(f'Policy Prediction Accuracy (ACC), {agent}')
    plt.xlabel('Episodes')
    plt.ylabel('Policy Prediction Accuracy (ACC)')
    plt.ylim([0, 1.1])

    if save_fig:
        plt.savefig(filename)

    if show_fig:
        plt.show()

    return df


def plot_phi_belief_wrt_corr_phi_q(
    pickle_dir: Union[Path, str],
    num_runs: int,
    num_episodes: int,
    agent: str,
    q_end: int,
    q_start: int = 0,
    save_fig: bool = False,
    filename: Optional[Union[Path, str]] = None,
    show_fig: bool = False,
) -> pd.DataFrame:
    if save_fig and filename is None:
        raise RuntimeError('Please provide a filename to save the figure')

    df = pd.DataFrame()

    plt.figure(figsize=(15, 6))
    colortype = ('r', 'b', 'k', 'g', 'c', 'm', 'y', 'gray', 'lime', 'indigo')
    for q in range(q_start, q_end + 1):
        # plot belief of correspounding phi
        pickle_file = (
            Path(pickle_dir) / f'op_new_phi_{q}_q_{num_runs}_runs_{num_episodes}_episodes.pkl'
        )

        phi_belief_wrt_corr_phi = collect_phi_belief_wrt_corr_phi(pickle_file, agent)

        plt.plot(phi_belief_wrt_corr_phi, marker='o', color=colortype[q], label=f'q_{q}')

        df[f'q_{q}'] = phi_belief_wrt_corr_phi

        # plot belief of phi 6
        data = load_pickle(pickle_file)
        results = data[agent]

        phi_beliefs = []
        for result in results:
            phi_beliefs.append(result['phi_beliefs'])

        phi_belief = np.average(phi_beliefs, axis=0).transpose()[5]  # belief of phi 6
        plt.plot(phi_belief, marker='+', color=colortype[q], label=f'q_{q}_phi6')

        df[f'q_{q}_phi6'] = phi_belief

    plt.legend()
    plt.title(f'Averaged Belief of the Corresponding Phi ({agent})')
    plt.xlabel('Episodes')
    plt.ylabel('Belief')
    plt.ylim([0, 1.1])

    if save_fig:
        plt.savefig(filename)

    if show_fig:
        plt.show()

    return df


def plot_phi_belief_wrt_corr_phi_p(
    pickle_dir: Union[Path, str],
    num_runs: int,
    num_episodes: int,
    agent: str,
    save_fig: bool = False,
    filename: Optional[Union[Path, str]] = None,
    show_fig: bool = False,
) -> pd.DataFrame:
    if save_fig and filename is None:
        raise RuntimeError('Please provide a filename to save the figure')

    df = pd.DataFrame()

    plt.figure(figsize=(15, 6))
    colortype = ('r', 'b', 'k', 'g', 'c', 'm', 'y', 'gray', 'lime', 'indigo')
    p_patterns = (0.2, 0.4, 0.6, 0.8, 1)
    for i, p_pattern in enumerate(p_patterns):
        # plot belief of correspounding phi
        pickle_file = (
            Path(pickle_dir)
            / f'op_new_phi_{p_pattern}_p_{num_runs}_runs_{num_episodes}_episodes.pkl'
        )

        phi_belief_wrt_corr_phi = collect_phi_belief_wrt_corr_phi(pickle_file, agent)

        plt.plot(phi_belief_wrt_corr_phi, marker='o', color=colortype[i], label=f'p_{p_pattern}')

        df[f'p_{p_pattern}'] = phi_belief_wrt_corr_phi

        # plot belief of phi 6
        data = load_pickle(pickle_file)
        results = data[agent]

        phi_beliefs = []
        for result in results:
            phi_beliefs.append(result['phi_beliefs'])

        phi_belief = np.average(phi_beliefs, axis=0).transpose()[5]  # belief of phi 6
        plt.plot(phi_belief, marker='+', color=colortype[i], label=f'p_{p_pattern}_phi6')

        df[f'p_{p_pattern}_phi6'] = phi_belief

    plt.legend()
    plt.title(f'Averaged Belief of the Corresponding Phi ({agent})')
    plt.xlabel('Episodes')
    plt.ylabel('Belief')
    plt.ylim([0, 1.1])

    if save_fig:
        plt.savefig(filename)

    if show_fig:
        plt.show()

    return df


def plot_winning_percentage(
    pickle_file: Union[Path, str],
    agents: List[str],
    save_fig: bool = False,
    filename: Optional[Union[Path, str]] = None,
    show_fig: bool = False,
) -> pd.DataFrame:
    """
    Plot the winning percentage of the agents

    x-axis: episodes
    y-axis: winning percentage
    each line: an agent
    """
    if save_fig and filename is None:
        raise RuntimeError('Please provide a filename to save the figure')

    df = pd.DataFrame()

    plt.figure(figsize=(15, 6))
    for agent in agents:
        win_rate = collect_win_rate(pickle_file, agent)
        plt.plot(win_rate, '--', label=agent)
        df[agent] = win_rate

    plt.legend()
    plt.title('Winning percentage (WP) of BSI-PT and the BPR variants')
    plt.xlabel('Episodes')
    plt.ylabel('Winning percentage (WP)')
    plt.ylim([0.2, 0.6])

    if save_fig:
        plt.savefig(filename)

    if show_fig:
        plt.show()

    return df


def plot_policy_pred_acc_with_intra(
    pickle_file: Union[Path, str],
    agent: str,
    save_fig: bool = False,
    filename: Optional[Union[Path, str]] = None,
    show_fig: bool = False,
):
    """
    Plot the policy prediction accuracy (including intra markers) of an agent.
    This function is specifically for BSI-PT and Bayes-OKR in the baseball environment.

    x-axis: episodes
    y-axis: policy prediction accuracy
    """
    if save_fig and filename is None:
        raise RuntimeError('Please provide a filename to save the figure')

    df = pd.DataFrame()
    policy_pred_acc_with_intra = collect_policy_prediction_with_intra(pickle_file, agent)
    num_episodes = len(policy_pred_acc_with_intra[0])

    STEPS_PER_EPISODE = 4
    acc_with_intra = np.array([policy_pred_acc_with_intra[i] for i in range(STEPS_PER_EPISODE)])
    acc_with_intra = acc_with_intra.flatten(order='F')

    plt.figure(figsize=(15, 6))
    plt.plot(acc_with_intra, '--', color='grey')
    df['acc_with_intra'] = acc_with_intra
    # plot the intra-episode markers
    for step in range(STEPS_PER_EPISODE):
        plt.scatter(
            np.arange(num_episodes) * STEPS_PER_EPISODE + step,
            policy_pred_acc_with_intra[step],
            label=f'step {step}',
            zorder=10,
        )

    plt.legend()
    plt.grid()
    plt.title(f'Policy Prediction Accuracy (ACC) with Intra-Episode, {agent}')
    plt.xlabel('Episodes')
    plt.xticks(np.arange(num_episodes) * STEPS_PER_EPISODE, np.arange(num_episodes) + 1)
    plt.ylabel('Policy Prediction Accuracy (ACC)')
    plt.ylim([0.0, 1.1])

    if save_fig:
        plt.savefig(filename)

    if show_fig:
        plt.show()

    return df


def collect_win_rate(pickle_file: Union[Path, str], agent: str) -> np.ndarray:
    data = load_pickle(pickle_file)

    results = data[agent]

    win_records = []
    for result in results:
        win_records.append(result['win_records'])

    return np.sum(np.array(win_records).astype(int), axis=0) / len(win_records)


def collect_policy_prediction(pickle_file: Union[Path, str], agent: str) -> np.ndarray:
    data = load_pickle(pickle_file)

    results = data[agent]

    policy_preds = []
    for result in results:
        policy_preds.append(result['policy_preds'])

    return np.sum(np.array(policy_preds).astype(int), axis=0) / len(policy_preds)


def collect_policy_prediction_with_intra(pickle_file: Union[Path, str], agent: str) -> np.ndarray:
    """
    This function is specifically for BSI-PT and Bayes-OKR in the baseball environment.
    """
    data = load_pickle(pickle_file)

    results = data[agent]

    policy_preds_intra = {
        0: [],
        1: [],
        2: [],
        3: [],
        4: [],
        5: [],
        6: [],
    }
    for result in results:
        for step in policy_preds_intra.keys():
            policy_preds_intra[step].append(result[f'step_{step}_policy_preds'])

    return {
        step: np.sum(np.array(policy_preds_intra[step]).astype(int), axis=0)
        / len(policy_preds_intra[step])
        for step in policy_preds_intra.keys()
    }


def collect_phi_belief_wrt_corr_phi(pickle_file: Union[Path, str], agent: str) -> np.ndarray:
    data = load_pickle(pickle_file)

    results = data[agent]

    beliefs = []
    for result in results:
        corresponding_phi = result['corresponding_phi']
        beliefs.append([belief[corresponding_phi - 1] for belief in result['phi_beliefs']])

    return np.average(beliefs, axis=0)
