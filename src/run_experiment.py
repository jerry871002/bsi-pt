import argparse
import multiprocessing
import random
from pathlib import Path

from run import positive_int
from utils import store_as_pickle


def run_experiment(args: argparse.Namespace) -> None:
    if args.scenario == 'grid':
        from grid_world.agent import BprAgent
        from grid_world.env import Opponent
        from grid_world.run import (
            run_bpr_okr,
            run_bpr_plus,
            run_bsi,
            run_bsi_pt,
            run_deep_bpr_plus,
            run_tom,
        )
    elif args.scenario == 'nav':
        from navigation_game.agent import BprAgent  # type: ignore
        from navigation_game.env import Opponent  # type: ignore
        from navigation_game.run import (
            run_bpr_okr,
            run_bpr_plus,
            run_bsi,
            run_bsi_pt,
            run_deep_bpr_plus,
            run_tom,
        )
    elif args.scenario == 'soccer':
        from soccer_game.agent import BprAgent  # type: ignore
        from soccer_game.env import Opponent  # type: ignore
        from soccer_game.run import (
            run_bpr_okr,
            run_bpr_plus,
            run_bsi,
            run_bsi_pt,
            run_deep_bpr_plus,
            run_tom,
        )
    elif args.scenario == 'baseball':
        from baseball_game.agent import BprAgent  # type: ignore
        from baseball_game.env import Opponent  # type: ignore
        from baseball_game.run import (
            run_bpr_okr,
            run_bpr_plus,
            run_bsi,
            run_bsi_pt,
            run_deep_bpr_plus,
            run_tom,
        )

    run_functions = {
        'bpr+': run_bpr_plus,
        'deep-bpr+': run_deep_bpr_plus,
        'tom': run_tom,
        'bpr-okr': run_bpr_okr,
        'bsi': run_bsi,
        'bsi-pt': run_bsi_pt,
    }

    # baseball game has different number of policies than the existing scenarios
    # need to pass the number of policies to the worker function
    args.agent_policy_range = len(BprAgent.Policy)
    args.opponent_policy_range = len(Opponent.Policy)

    # if the user specifies which agents to run
    # only keep those that are specified
    if args.agents:
        for agent in list(run_functions.keys()):
            if agent not in args.agents:
                del run_functions[agent]

    if args.multi_processing:  # multi-processing mode
        # we can use a `Manager` object to share data between processes
        # source: https://docs.python.org/3/library/multiprocessing.html#sharing-state-between-processes
        with multiprocessing.Manager() as manager:
            # by default `os.cpu_count()` is used as the number of worker processes in the pool
            with multiprocessing.Pool() as pool:
                result = manager.dict({agent: manager.list() for agent in run_functions.keys()})

                for agent, run_function in run_functions.items():
                    for i in range(args.num_runs):
                        pool.apply_async(func=worker, args=(run_function, args, agent, result, i))

                pool.close()
                pool.join()

            for agent, rewards in result.items():
                result[agent] = list(rewards)  # type: ignore

            result = dict(result)  # type: ignore
    else:  # normal mode (single process)
        result = {agent: list() for agent in run_functions.keys()}  # type: ignore
        for agent, run_function in run_functions.items():
            print(f'========== Agent {agent} ==========')
            for i in range(args.num_runs):
                worker(run_function, args, agent, result, i)

    # write experiment result into file
    filename = 'op_'
    if args.bpr_opponent:
        filename += 'bpr_'
    elif args.phi_opponent:
        filename += f'{args.phi}_phi_'
    elif args.new_phi_opponent:
        filename += f'new_phi_{args.q_distance}_q_'
    elif args.new_phi_noise_opponent:
        filename += f'new_phi_{args.p_pattern}_p_'
    else:
        filename += f'{args.episode_reset}_random_'
    filename += f'{args.num_runs}_runs_{args.num_episodes}_episodes.pkl'

    pickle_dir = args.data_dir / args.scenario
    pickle_dir.mkdir(parents=True, exist_ok=True)
    store_as_pickle(result, pickle_dir / filename)

    print(f'[run_experiment result]\n{result}')


def worker(run_function, args, agent, return_dict, run_i):
    # randomly select agent and opponent's policy
    args.agent_policy = random.randint(1, args.agent_policy_range)
    args.op_policy = random.randint(1, args.opponent_policy_range)

    if not args.multi_processing:
        print(f'========== Start run {run_i+1} ==========')

    rewards = run_function(args)
    return_dict[agent].append(rewards)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run multiple runs of the experiment in a given scenario with an agent type'
    )
    parser.add_argument(
        'scenario',
        type=str,
        choices=('grid', 'nav', 'soccer', 'baseball'),
        help='the scenario you would like to run on',
    )
    parser.add_argument(
        '-r',
        '--num-runs',
        type=positive_int,
        default=10,
        help='number of total runs, default value is 10',
    )
    parser.add_argument(
        '-n',
        '--num-episodes',
        type=positive_int,
        default=1,
        help='number of episodes in each run, default value is 1',
    )
    parser.add_argument(
        '-a',
        '--agents',
        type=str,
        nargs='*',
        choices=('bpr+', 'deep-bpr+', 'tom', 'bpr-okr', 'bsi', 'bsi-pt'),
        help='the agents you want to run in the experiment, if not set, all agents will be run',
    )
    parser.add_argument(
        '-ph',
        '--phi',
        type=int,
        choices=range(1, 12),
        default=1,
        help=(
            'enter 1~11 to choose the phi used by the opponent, '
            'this value will only be used if `--phi-opponent` is also set, default value is 1'
        ),
    )
    parser.add_argument(
        '-e',
        '--episode-reset',
        type=positive_int,
        default=10,
        help='number of episodes before the opponent switches its policy, default value is 10',
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
    parser.add_argument(
        '-m',
        '--multi-processing',
        action='store_true',
        help='whether to compute using multi-processing',
    )
    parser.add_argument(
        '-d',
        '--data-dir',
        type=Path,
        default=Path('data/'),
        help='where to store the result pickle files',
    )
    parser.add_argument(
        '-q',
        '--q-distance',
        type=int,
        default=0,
        help=(
            'distance between existing phi and new phi, '
            'this value will only be used if `--new-phi-opponent` is set, default value is 0'
        ),
    )
    parser.add_argument(
        '-pat',
        '--p-pattern',
        type=float,
        default=1,
        help=(
            'probability of using policy in existing phi, '
            'this value will only be used if `--new-phi-noise-opponent` is set, default value is 1'
        ),
    )

    opponent_setting_group = parser.add_mutually_exclusive_group()
    opponent_setting_group.add_argument(
        '-b',
        '--bpr-opponent',
        action='store_true',
        help='whether the environment uses a BPR opponent',
    )
    opponent_setting_group.add_argument(
        '-po',
        '--phi-opponent',
        action='store_true',
        help=(
            'whether the environment uses a Phi opponent, '
            'use `--phi` to specify the type of phi to use'
        ),
    )
    opponent_setting_group.add_argument(
        '-np',
        '--new-phi-opponent',
        action='store_true',
        help=(
            'whether the environment uses a new Phi opponent, '
            'use `--q-distance` to specify the probability of using policy in existing phi'
        ),
    )
    opponent_setting_group.add_argument(
        '-nnp',
        '--new-phi-noise-opponent',
        action='store_true',
        help=(
            'whether the environment uses a new Phi noise opponent, '
            'use `--p-pattern` to control the probability of randomly choosing policy'
        ),
    )
    args = parser.parse_args()

    run_experiment(args)
