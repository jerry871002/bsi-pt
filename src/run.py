import argparse
import importlib


def run(args: argparse.Namespace) -> None:
    scenario_modules = {
        'grid': 'grid_world.run',
        'nav': 'navigation_game.run',
        'soccer': 'soccer_game.run',
        'baseball': 'baseball_game.run',
    }

    if args.scenario in scenario_modules:
        run_module = importlib.import_module(scenario_modules[args.scenario])
        run_bpr_okr = run_module.run_bpr_okr
        run_bpr_plus = run_module.run_bpr_plus
        run_bsi = run_module.run_bsi
        run_bsi_pt = run_module.run_bsi_pt
        run_deep_bpr_plus = run_module.run_deep_bpr_plus
        run_tom = run_module.run_tom
    else:
        raise ValueError(f"Unsupported scenario: {scenario}")

    agent_functions = {
        'bpr+': run_bpr_plus,
        'deep-bpr+': run_deep_bpr_plus,
        'tom': run_tom,
        'bpr-okr': run_bpr_okr,
        'bsi': run_bsi,
        'bsi-pt': run_bsi_pt,
    }

    if args.agent in agent_functions:
        agent_functions[args.agent](args)
    else:
        raise ValueError(f"Unsupported agent type: {agent}")


def positive_int(value: str) -> int:
    if int(value) <= 0:
        raise argparse.ArgumentTypeError(f'{value} is an invalid positive int value')
    return int(value)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run the experiment in a given scenario with an agent type'
    )
    parser.add_argument(
        'scenario',
        type=str,
        choices=('grid', 'nav', 'soccer', 'baseball'),
        help='the scenario you would like to run on',
    )
    parser.add_argument(
        'agent',
        type=str,
        choices=('bpr+', 'deep-bpr+', 'tom', 'bpr-okr', 'bsi', 'bsi-pt'),
        help='type of agent you would like to use',
    )
    parser.add_argument(
        '-o',
        '--op-policy',
        type=int,
        choices=range(1, 6),
        default=-1,
        help=(
            'enter 1~5 to choose the initial policy (tau) used by the opponent, '
            'randomly choose one if not set'
        ),
    )
    parser.add_argument(
        '-a',
        '--agent-policy',
        type=int,
        choices=range(1, 6),
        default=-1,
        help=(
            'enter 1~5 to choose the initial policy (pi) used by the agent, '
            'randomly choose one if not set'
        ),
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
        '-n',
        '--num-episodes',
        type=positive_int,
        default=1,
        help='enter the number of episodes you want to run, default value is 1',
    )
    parser.add_argument(
        '-q',
        '--q-distance',
        type=int,
        default=0,
        help=(
            'distance between existing phi and new phi, '
            'this value will only be used if `--new-phi-opponent` is also set, default value is 0'
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

    run(args)
