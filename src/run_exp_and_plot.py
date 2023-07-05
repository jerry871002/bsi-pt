"""
There are four experiments to choose from
1. BSI, BSI-PT against phi opponent
    Plots:
        1. Phi belief
        2. KL divergence
        3. Win rate
2. All agents against phi opponents
    Plot: cumulative rewards
3. All agents against random switch opponents (switch interval: 1, 3 and 10)
    Plot: episodic rewards
4. All agents against BPR opponent
    Plot: cumulative rewards
"""

import argparse
from itertools import product
from pathlib import Path
from typing import Tuple

import pandas as pd

from plot import (plot_cumulative_rewards, plot_episodic_rewards,
                  plot_kl_divergences, plot_phi_belief_wrt_corr_phi_q,
                  plot_phi_belief_wrt_corr_phi_p, plot_phi_beliefs,
                  plot_policy_pred_acc, plot_specific_phi_beliefs,
                  plot_win_rates)
from run import positive_int
from run_experiment import run_experiment


def experiment_one(args: argparse.Namespace) -> None:
    args.new_phi_noise_opponent = False
    args.phi_opponent = True
    args.bpr_opponent = False
    args.new_phi_opponent = False
    args.q_distance = 0
    args.p_pattern=1
    args.agents = ['bsi', 'bsi-pt']
    args.data_dir = Path('data/exp_one')

    N_PHI = 11

    for phi in range(6, N_PHI + 1):
        args.phi = phi
        run_experiment(args)

    pickle_dir = args.data_dir / args.scenario
    fig_dir, csv_dir = make_fig_csv_dir('exp_one', args.scenario)

    # plot phi belief
    for agent in args.agents:
        for phi in range(6, N_PHI + 1):
            pickle_file = pickle_dir / f'op_{phi}_phi_{args.num_runs}_runs_{args.num_episodes}_episodes.pkl'
            df = plot_phi_beliefs(
                pickle_file=pickle_file, agent=agent,
                save_fig=True, filename=fig_dir / f'exp1_{args.scenario}_{agent}_belief_omega{phi}.png'
            )
            save_as_csv(df, csv_dir / f'exp1_{args.scenario}_{agent}_belief_omega{phi}.csv')

    # plot KL divergence
    '''for agent in args.agents:
        df = plot_kl_divergences(
            pickle_dir=pickle_dir, num_runs=args.num_runs, num_episodes=args.num_episodes,
            agent=agent, save_fig=True, filename=fig_dir / f'exp1_{args.scenario}_{agent}_kl.png'
        )
        save_as_csv(df, csv_dir / f'exp1_{args.scenario}_{agent}_kl.csv')'''

    # plot win rate
    for agent in args.agents:
        df = plot_win_rates(
            pickle_dir=pickle_dir, num_runs=args.num_runs, num_episodes=args.num_episodes,
            agent=agent, save_fig=True, filename=fig_dir / f'exp1_{args.scenario}_{agent}_wr.png'
        )
        save_as_csv(df, csv_dir / f'exp1_{args.scenario}_{agent}_wr.csv')

    # policy prediction accuracy
    '''for agent in args.agents:
        df = plot_policy_pred_acc(
            pickle_dir=pickle_dir, num_runs=args.num_runs, num_episodes=args.num_episodes,
            agent=agent, save_fig=True, filename=fig_dir / f'exp1_{args.scenario}_{agent}_acc.png'
        )
        save_as_csv(df, csv_dir / f'exp1_{args.scenario}_{agent}_acc.csv')'''

def experiment_two(args: argparse.Namespace) -> None:
    args.new_phi_noise_opponent = False
    args.phi_opponent = True
    args.bpr_opponent = False
    args.new_phi_opponent = False
    args.q_distance = 0
    args.p_pattern=1
    args.agents = ['bpr+', 'deep-bpr+', 'tom', 'bpr-okr', 'bsi', 'bsi-pt']
    args.data_dir = Path('data/exp_two')

    # plot every phi in [PHI_START, PHI_END]
    # e.g. PHI_START = 5 and PHI_END = 8 -> plot phi 5, 6, 7, and 8
    PHI_START = 6
    PHI_END = 11

    for phi in range(PHI_START, PHI_END + 1):
        args.phi = phi
        run_experiment(args)

    pickle_dir = args.data_dir / args.scenario
    fig_dir, csv_dir = make_fig_csv_dir('exp_two', args.scenario)

    # plot cumulative rewards
    for phi in range(PHI_START, PHI_END + 1):
        pickle_file = pickle_dir / f'op_{phi}_phi_{args.num_runs}_runs_{args.num_episodes}_episodes.pkl'
        df = plot_cumulative_rewards(
            pickle_file=pickle_file, save_fig=True,
            filename=fig_dir / f'exp2_{args.scenario}_utility_omega{phi}.png'
        )
        save_as_csv(df, csv_dir / f'exp2_{args.scenario}_utility_omega{phi}.csv')

def experiment_three(args: argparse.Namespace) -> None:
    args.new_phi_noise_opponent = False
    args.phi_opponent = False
    args.bpr_opponent = False
    args.new_phi_opponent = False
    args.q_distance = 0
    args.p_pattern=1
    args.agents = ['bpr+', 'deep-bpr+', 'tom', 'bpr-okr', 'bsi', 'bsi-pt']
    args.data_dir = Path('data/exp_three')

    random_switch_intervals = [1,2, 3, 4,5,6,7,8]

    for interval in random_switch_intervals:
        args.episode_reset = interval
        run_experiment(args)

    pickle_dir = args.data_dir / args.scenario
    fig_dir, csv_dir = make_fig_csv_dir('exp_three', args.scenario)

    # plot episodic rewards
    for interval in random_switch_intervals:
        pickle_file = pickle_dir / f'op_{interval}_random_{args.num_runs}_runs_{args.num_episodes}_episodes.pkl'
        #df = plot_episodic_rewards(
        df = plot_cumulative_rewards(
            pickle_file=pickle_file, save_fig=True,
            filename=fig_dir / f'exp3_{args.scenario}_utility_interval{interval}.png'
        )
        save_as_csv(df, csv_dir / f'exp3_{args.scenario}_utility_interval{interval}.csv')

    # plot phi belief of phi 6 (random switch opponent)
    for agent in ('bsi', 'bsi-pt'):
        pickle_files = [
            pickle_dir / f'op_{interval}_random_{args.num_runs}_runs_{args.num_episodes}_episodes.pkl'
            for interval in random_switch_intervals
        ]
        labels = [
            f'interval_{interval}' for interval in random_switch_intervals
        ]
        df = plot_specific_phi_beliefs(
            pickle_files=pickle_files, labels=labels, agent=agent, phi_num=6,
            save_fig=True, filename=fig_dir / f'exp3_{args.scenario}_{agent}_belief_omega6.png'
        )
        save_as_csv(df, csv_dir / f'exp3_{args.scenario}_{agent}_belief_omega6.csv')

def experiment_four(args: argparse.Namespace) -> None:
    args.new_phi_noise_opponent = False
    args.bpr_opponent = True
    args.phi_opponent = False
    args.new_phi_opponent = False
    args.q_distance = 0
    args.p_pattern=1
    args.agents = ['bpr+', 'deep-bpr+', 'tom', 'bpr-okr', 'bsi', 'bsi-pt']
    args.data_dir = Path('data/exp_four')

    run_experiment(args)

    pickle_dir = args.data_dir / args.scenario
    fig_dir, csv_dir = make_fig_csv_dir('exp_four', args.scenario)

    # plot cumulative rewards
    pickle_file = pickle_dir / f'op_bpr_{args.num_runs}_runs_{args.num_episodes}_episodes.pkl'
    df = plot_cumulative_rewards(
        pickle_file=pickle_file, save_fig=True,
        filename=fig_dir / f'exp4_{args.scenario}_utility_BPR.png'
    )
    save_as_csv(df, csv_dir / f'exp4_{args.scenario}_utility_BPR.csv')

    # plot phi belief
    for agent in ('bsi', 'bsi-pt'):
            df = plot_phi_beliefs(
                pickle_file=pickle_file, agent=agent,
                save_fig=True, filename=fig_dir / f'exp4_{args.scenario}_{agent}_belief.png'
            )
            save_as_csv(df, csv_dir / f'exp4_{args.scenario}_{agent}_belief.csv')

def experiment_five(args: argparse.Namespace) -> None:
    args.new_phi_noise_opponent = False
    args.new_phi_opponent = True
    args.bpr_opponent = False
    args.phi_opponent = False
    args.p_pattern=1
    args.agents = ['bpr+', 'deep-bpr+', 'tom', 'bpr-okr', 'bsi', 'bsi-pt']
    args.data_dir = Path('data/exp_five')

    if args.scenario == 'grid':
        q_max = 5
    elif args.scenario == 'nav':
        q_max = 6
    elif args.scenario == 'soccer':
        q_max = 4

    for q_distance in range(q_max + 1):
        args.q_distance = q_distance
        run_experiment(args)

    pickle_dir = args.data_dir / args.scenario
    fig_dir, csv_dir = make_fig_csv_dir('exp_five', args.scenario)

    # plot cumulative rewards
    for q_distance in range(q_max + 1):
        pickle_file = pickle_dir / f'op_new_phi_{q_distance}_q_{args.num_runs}_runs_{args.num_episodes}_episodes.pkl'
        df = plot_cumulative_rewards(
            pickle_file=pickle_file, save_fig=True,
            filename=fig_dir / f'exp5_{args.scenario}_utility_q{q_distance}.png'
        )
        save_as_csv(df, csv_dir / f'exp5_{args.scenario}_utility_q{q_distance}.csv')

    # plot phi belief with respect to corresponding phi
    for agent in ('bsi', 'bsi-pt'):
        df = plot_phi_belief_wrt_corr_phi_q(
            pickle_dir=pickle_dir, num_runs=args.num_runs, num_episodes=args.num_episodes, q_end=4,  # originally q_end=q_max
            agent=agent, save_fig=True, filename=fig_dir / f'exp5_{args.scenario}_{agent}_belief.png'
        )
        save_as_csv(df, csv_dir / f'exp5_{args.scenario}_{agent}_belief.csv')

def experiment_six(args: argparse.Namespace) -> None:
    args.new_phi_noise_opponent = True
    args.new_phi_opponent = False
    args.bpr_opponent = False
    args.phi_opponent = False
    args.q_distance = 0
    args.agents = ['bpr+', 'deep-bpr+', 'tom', 'bpr-okr', 'bsi', 'bsi-pt']
    args.data_dir = Path('data/exp_five')

    p_patterns = (0.2, 0.4, 0.6, 0.8, 1)
    for p_pattern in p_patterns:
        args.p_pattern = p_pattern
        run_experiment(args)

    pickle_dir = args.data_dir / args.scenario
    fig_dir, csv_dir = make_fig_csv_dir('exp_six', args.scenario)

    # plot cumulative rewards
    for p_pattern in p_patterns:
        pickle_file = pickle_dir / f'op_new_phi_{p_pattern}_p_{args.num_runs}_runs_{args.num_episodes}_episodes.pkl'
        df = plot_cumulative_rewards(
            pickle_file=pickle_file, save_fig=True,
            filename=fig_dir / f'exp6_{args.scenario}_utility_p_{p_pattern}.png'
        )
        save_as_csv(df, csv_dir / f'exp6_{args.scenario}_utility_p_{p_pattern}.csv')

    # plot phi belief with respect to corresponding phi
    for agent in ('bsi', 'bsi-pt'):
        df = plot_phi_belief_wrt_corr_phi_p(
            pickle_dir=pickle_dir, num_runs=args.num_runs, num_episodes=args.num_episodes,
            agent=agent, save_fig=True, filename=fig_dir / f'exp6_{args.scenario}_{agent}_belief.png'
        )
        save_as_csv(df, csv_dir / f'exp6_{args.scenario}_{agent}_belief.csv')

def save_as_csv(df: pd.DataFrame, filename: Path):
    df.insert(loc=0, column='episode', value=df.index+1)
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
        '-e', '--exp-nums', type=int, nargs='*',
        choices=range(1, 7), default=range(1, 7),
        help='the experiment(s) you would like to run'
    )
    parser.add_argument(
        '-s', '--scenarios', type=str, nargs='*',
        choices=('grid', 'nav', 'soccer'), default=('grid', 'nav', 'soccer'),
        help='the scenario(s) you would like to run on'
    )
    parser.add_argument(
        '-r', '--num-runs', type=positive_int, default=10,
        help='number of total runs, default value is 10'
    )
    parser.add_argument(
        '-n', '--num-episodes', type=positive_int, default=1,
        help='number of episodes in each run, default value is 1'
    )
    parser.add_argument(
        '-p', '--print-map', action='store_true',
        help='print the map of the game while episodes are played'
    )
    parser.add_argument(
        '-pa', '--print-action', action='store_true',
        help='print the action of agent and opponent in each step'
    )

    args = parser.parse_args()
    args.multi_processing = False  # always use multi-processing to save time

    experiments = {
        1: experiment_one,
        2: experiment_two,
        3: experiment_three,
        4: experiment_four,
        5: experiment_five,
        6: experiment_six
    }

    for exp_num, scenario in product(args.exp_nums, args.scenarios):
        args.scenario = scenario
        experiments[exp_num](args)
