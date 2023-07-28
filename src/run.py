import argparse


def run(args: argparse.Namespace) -> None:
    if args.scenario == 'grid':
        from grid_world.run import (run_bpr_okr, run_bpr_plus, run_bsi,
                                    run_bsi_pt, run_deep_bpr_plus, run_tom)
    elif args.scenario == 'nav':
        from navigation_game.run import (run_bpr_okr, run_bpr_plus, run_bsi,
                                         run_bsi_pt, run_deep_bpr_plus,
                                         run_tom)
    elif args.scenario == 'soccer':
        from soccer_game.run import (run_bpr_okr, run_bpr_plus, run_bsi,
                                     run_bsi_pt, run_deep_bpr_plus, run_tom)
    elif args.scenario == 'baseball':
        from baseball_game.run import (run_bpr_okr, run_bpr_plus, run_bsi,
                                       run_bsi_pt, run_deep_bpr_plus)

    if args.agent == 'bpr+':
        run_bpr_plus(args)
    elif args.agent == 'deep-bpr+':
        run_deep_bpr_plus(args)
    elif args.agent == 'tom':
        run_tom(args)
    elif args.agent == 'bpr-okr':
        run_bpr_okr(args)
    elif args.agent == 'bsi':
        run_bsi(args)
    elif args.agent == 'bsi-pt':
        run_bsi_pt(args)

def positive_int(value: str) -> int:
    if int(value) <= 0:
        raise argparse.ArgumentTypeError(f'{value} is an invalid positive int value')
    return int(value)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the experiment in a given scenario with an agent type')
    parser.add_argument(
        'scenario', type=str, choices=('grid', 'nav', 'soccer', 'baseball'),
        help='the scenario you would like to run on'
    )
    parser.add_argument(
        'agent', type=str, choices=('bpr+', 'deep-bpr+', 'tom', 'bpr-okr', 'bsi', 'bsi-pt'),
        help='type of agent you would like to use'
    )
    parser.add_argument(
        '-o', '--op-policy', type=int, choices=range(1, 6), default=-1,
        help=(
            'enter 1~5 to choose the initial policy (tau) used by the opponent, '
            'randomly choose one if not set'
        )
    )
    parser.add_argument(
        '-a', '--agent-policy', type=int, choices=range(1, 6), default=-1,
        help=(
            'enter 1~5 to choose the initial policy (pi) used by the agent, '
            'randomly choose one if not set'
        )
    )
    parser.add_argument(
        '-ph', '--phi', type=int, choices=range(1, 12), default=1,
        help=(
            'enter 1~11 to choose the phi used by the opponent, '
            'this value will only be used if `--phi-opponent` is also set, default value is 1'
        )
    )
    parser.add_argument(
        '-e', '--episode-reset', type=positive_int, default=10,
        help='number of episodes before the opponent switches its policy, default value is 10'
    )
    parser.add_argument(
        '-p', '--print-map', action='store_true',
        help='print the map of the game while episodes are played'
    )
    parser.add_argument(
        '-pa', '--print-action', action='store_true',
        help='print the action of agent and opponent in each step'
    )
    parser.add_argument(
        '-n', '--num-episodes', type=positive_int, default=1,
        help='enter the number of episodes you want to run, default value is 1'
    )
    parser.add_argument(
        '-q', '--q-distance', type=int, default=0,
        help=(
            'distance between existing phi and new phi, '
            'this value will only be used if `--new-phi-opponent` is also set, default value is 0'
        )
    )
    parser.add_argument(
        '-pat', '--p-pattern', type=float, default=1,
        help=(
            'probability of using policy in existing phi, '
            'this value will only be used if `--new-phi-noise-opponent` is also set, default value is 1'
        )
    )

    opponent_setting_group = parser.add_mutually_exclusive_group()
    opponent_setting_group.add_argument(
        '-b', '--bpr-opponent', action='store_true',
        help='whether the environment uses a BPR opponent'
    )
    opponent_setting_group.add_argument(
        '-po', '--phi-opponent', action='store_true',
        help=(
            'whether the environment uses a Phi opponent, '
            'you should also use `--phi` to specify the type of phi to use'
        )
    )
    opponent_setting_group.add_argument(
        '-np', '--new-phi-opponent', action='store_true',
        help=(
            'whether the environment uses a new Phi opponent, '
            'you should use `--q-distance` to specify the probability of using policy in existing phi'
        )
    )
    opponent_setting_group.add_argument(
        '-nnp', '--new-phi-noise-opponent', action='store_true',
        help=(
            'whether the environment uses a new Phi noise opponent, '
            'you should use `--p-pattern` to control the probability of randomly choosing policy'
        )
    )

    args = parser.parse_args()

    run(args)
