import argparse
from itertools import product
from pathlib import Path
from typing import Tuple

import pandas as pd
from plot import (
    plot_phi_beliefs,
    plot_policy_pred_acc,
    plot_win_rates,
)
from run import positive_int
from run_experiment import run_experiment


def set_opponent_type(args: argparse.Namespace, opponent_type: str) -> None:
    if opponent_type not in ['phi', 'new-phi', 'new-phi-noise', 'bpr']:
        raise ValueError(
            f'opponent_type must be phi, new-phi, new-phi-noise, or bpr, but got {opponent_type}'
        )

    args.new_phi_noise_opponent = False
    args.phi_opponent = False
    args.bpr_opponent = False
    args.new_phi_opponent = False

    if opponent_type == 'phi':
        args.phi_opponent = True
    elif opponent_type == 'new-phi':
        args.new_phi_opponent = True
    elif opponent_type == 'new-phi-noise':
        args.new_phi_noise_opponent = True
    elif opponent_type == 'bpr':
        args.bpr_opponent = True


def experiment_one(args: argparse.Namespace) -> None:
    """
    BSI-PT agent against phi opponent

    Plots:
        1. Phi belief
    """
    set_opponent_type(args, 'phi')
    args.q_distance = 0
    args.p_pattern = 1
    args.agents = ['bsi-pt']
    args.data_dir = Path('data/exp_one')

    # we are only interested in phi_5, phi_6, phi_10, and phi_11
    phi_range = (5, 6, 10, 11)

    for phi in phi_range:
        args.phi = phi
        run_experiment(args)

    pickle_dir = args.data_dir / args.scenario
    fig_dir, csv_dir = make_fig_csv_dir('exp_one', args.scenario)

    # plot phi belief
    for agent, phi in product(args.agents, phi_range):
        pickle_file = (
            pickle_dir / f'op_{phi}_phi_{args.num_runs}_runs_{args.num_episodes}_episodes.pkl'
        )
        df = plot_phi_beliefs(
            pickle_file=pickle_file,
            agent=agent,
            save_fig=True,
            filename=fig_dir / f'exp1_{args.scenario}_{agent}_belief_omega{phi}.png',
        )
        save_as_csv(df, csv_dir / f'exp1_{args.scenario}_{agent}_belief_omega{phi}.csv')


def experiment_two(args: argparse.Namespace) -> None:
    """
    BSI-PT and BPR-OKR agents against phi opponent

    Plots:
        1. Phi belief
        2. Accuracy
        3. Win percentage
    """
    set_opponent_type(args, 'phi')
    args.q_distance = 0
    args.p_pattern = 1
    args.agents = ['bsi-pt', 'bpr-okr']
    args.data_dir = Path('data/exp_two')

    # we are only interested in phi_5, phi_6, phi_10, and phi_11
    phi_range = (5, 6, 10, 11)

    for phi in phi_range:
        args.phi = phi
        run_experiment(args)

    pickle_dir = args.data_dir / args.scenario
    fig_dir, csv_dir = make_fig_csv_dir('exp_two', args.scenario)

    for agent in args.agents:
        # plot accuracy
        # TODO: collect accuracy data for bpr-okr
        if 'bsi' in agent:
            df = plot_policy_pred_acc(
                pickle_dir=pickle_dir,
                num_runs=args.num_runs,
                num_episodes=args.num_episodes,
                agent=agent,
                save_fig=True,
                filename=fig_dir / f'exp2_{args.scenario}_{agent}_acc.png',
            )
            save_as_csv(df, csv_dir / f'exp2_{args.scenario}_{agent}_acc.csv')

        # plot win rate
        df = plot_win_rates(
            pickle_dir=pickle_dir,
            num_runs=args.num_runs,
            num_episodes=args.num_episodes,
            agent=agent,
            save_fig=True,
            filename=fig_dir / f'exp2_{args.scenario}_{agent}_wr.png',
        )
        save_as_csv(df, csv_dir / f'exp2_{args.scenario}_{agent}_wr.csv')


def experiment_three(args: argparse.Namespace) -> None:
    """
    BSI-PT, BPR-OKR, D-BPR+, BPR+ agents against phi-noise opponent (epsilon: 0, 0.2, 0.5, and 1)
    epsilon in the paper is the complement of p_pattern in the code,
    i.e. epsilon = 1 - p_pattern

    Plots:
        1. Accuracy
        2. Win percentage
    """
    set_opponent_type(args, 'new-phi-noise')
    args.q_distance = 0
    args.agents = ['bpr+', 'deep-bpr+', 'bpr-okr', 'bsi-pt']
    args.data_dir = Path('data/exp_three')

    epsilons = (0, 0.2, 0.5, 1)
    for epsilon in epsilons:
        p_pattern = 1 - epsilon
        args.p_pattern = p_pattern
        run_experiment(args)

    pickle_dir = args.data_dir / args.scenario
    fig_dir, csv_dir = make_fig_csv_dir('exp_three', args.scenario)

    for agent in args.agents:
        # plot accuracy
        # TODO: collect accuracy data for other agents
        if 'bsi' in agent:
            df = plot_policy_pred_acc(
                pickle_dir=pickle_dir,
                num_runs=args.num_runs,
                num_episodes=args.num_episodes,
                agent=agent,
                save_fig=True,
                filename=fig_dir / f'exp3_{args.scenario}_{agent}_acc.png',
            )
            save_as_csv(df, csv_dir / f'exp3_{args.scenario}_{agent}_acc.csv')

        # plot win rate
        # FIXME: plot_win_rates is currently looking for phi opponent not new-phi-noise opponent
        df = plot_win_rates(
            pickle_dir=pickle_dir,
            num_runs=args.num_runs,
            num_episodes=args.num_episodes,
            agent=agent,
            save_fig=True,
            filename=fig_dir / f'exp3_{args.scenario}_{agent}_wr.png',
        )
        save_as_csv(df, csv_dir / f'exp3_{args.scenario}_{agent}_wr.csv')


def save_as_csv(df: pd.DataFrame, filename: Path):
    df.insert(loc=0, column='episode', value=df.index + 1)
    df.to_csv(filename, index=False, sep=' ', float_format='%.4f')


def make_fig_csv_dir(name: str, scenario: str) -> Tuple[Path, Path]:
    fig_dir = Path(f'fig/{name}/{scenario}')
    fig_dir.mkdir(parents=True, exist_ok=True)

    csv_dir = Path(f'csv/{name}/{scenario}')
    csv_dir.mkdir(parents=True, exist_ok=True)

    return fig_dir, csv_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run the experiments in our paper, save the results (pkl and csv) and plot them'
    )
    parser.add_argument(
        '-e',
        '--exp-nums',
        type=int,
        nargs='*',
        choices=range(1, 4),
        default=range(1, 4),
        help='the experiment(s) you would like to run',
    )
    parser.add_argument(
        '-s',
        '--scenarios',
        type=str,
        nargs='*',
        choices=('grid', 'nav', 'soccer'),
        default=('grid', 'nav', 'soccer'),
        help='the scenario(s) you would like to run on',
    )
    parser.add_argument(
        '-r',
        '--num-runs',
        type=positive_int,
        help='number of total runs',
    )
    parser.add_argument(
        '-n',
        '--num-episodes',
        type=positive_int,
        help='number of episodes in each run',
    )
    parser.add_argument(
        '-p',
        '--print-map',
        action='store_true',
        help='print the map of the game while episodes are played',
    )
    parser.add_argument(
        '-pa',
        '--print-action',
        action='store_true',
        help='print the action of agent and opponent in each step',
    )

    args = parser.parse_args()
    # always use multi-processing to save time
    # turn it to False if you need to debug
    args.multi_processing = True

    experiments = {
        1: experiment_one,
        2: experiment_two,
        3: experiment_three,
    }

    for exp_num, scenario in product(args.exp_nums, args.scenarios):
        args.scenario = scenario
        experiments[exp_num](args)
