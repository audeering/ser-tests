import argparse
import importlib
import os
import sys
import typing

import audeer


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CURRENT_DIR, 'common'))
from common import (  # noqa: E402
    Impacts,
    model_name,
    print_impact_summary,
    print_impact_title,
)

from test import (  # noqa: E402
    available_tests,
    get_model_interface,
    run_tests,
)


def main():
    r"""Load a baseline model and a candidate and analyse the impact."""
    args = parse_arguments()
    condition = args.condition
    device = args.device
    impact_test = args.impact
    model_baseline_id = args.baseline
    model_candidate_id = args.candidate
    max_signal_length_baseline = args.max_signal_length_baseline
    max_signal_length_candidate = args.max_signal_length_candidate
    tuning_baseline = {}
    tuning_candidate = {}

    # Check if test is available or if we should list tests
    impact_tests = available_impact_tests(condition)
    if impact_test == 'list':
        print(f'Available impact tests are: {", ".join(impact_tests)}')
        return
    if impact_test is not None:
        if impact_test not in impact_tests:
            raise ValueError(
                f"The selected impact test '{impact_test}' is not available. "
                f"Possible impact tests are: {', '.join(impact_tests)}."
            )
        impact_tests = [impact_test]

    # Assign tuning parameters if set
    if max_signal_length_baseline is not None:
        tuning_baseline['max_signal_length'] = max_signal_length_baseline
    if max_signal_length_candidate is not None:
        tuning_candidate['max_signal_length'] = max_signal_length_candidate

    model_baseline = get_model_interface(
        model_baseline_id, tuning_baseline,
        device, condition
    )

    model_candidate = get_model_interface(
        model_candidate_id, tuning_candidate,
        device, condition
    )

    # Check if tests have been run for both models
    # and if not, run them
    for model in [model_baseline, model_candidate]:
        result_dir = os.path.join(
            CURRENT_DIR,
            'docs',
            'results',
            'test',
            condition,
            model_name(model),
        )
        if not os.path.exists(result_dir):
            print(f'Model {model_name(model)} has no test results yet.')
            print(
                f'Executing tests for {model_name(model)} '
                'before impact analysis.'
            )
            tests = available_tests(condition)
            run_tests(model, condition, tests)

    # Set positive/negative/neutral/skipped test counter to 0
    Impacts.reset()

    print_impact_title(
        model_name(model_baseline),
        model_name(model_candidate),
        condition
    )

    # Execute all requested tests
    for impact_test in impact_tests:
        module = importlib.import_module(f'common.{impact_test}')
        run = getattr(module, 'run')
        run(model_baseline, model_candidate, condition)

    print_impact_summary(Impacts)


def available_impact_tests(condition: str) -> typing.List[str]:
    r"""List all available impact tests.

    Args:
        condition: ``'arousal'``, ``'dominance'``,
            ``'valence'``, or ``'emotion'``
    Returns:
        list of test names

    """
    impact_test_dir = os.path.join(CURRENT_DIR, 'impact', condition)
    tests = audeer.list_file_names(impact_test_dir, filetype='yaml')
    return [audeer.basename_wo_ext(t) for t in tests]


def parse_arguments():
    parser = argparse.ArgumentParser(description='Model comparisons')
    parser.add_argument(
        'condition',
        choices=['arousal', 'dominance', 'emotion', 'valence'],
        help='Emotion dimension or categories.',
    )
    parser.add_argument(
        '--baseline',
        metavar='MODEL_BASELINE_ID',
        type=str,
        help=(
            'ONNX model ID of the baseline model.'
        ),
        required=True
    )
    parser.add_argument(
        '--candidate',
        metavar='MODEL_CANDIDATE_IDS',
        type=str,
        help=(
            'ONNX model ID of the candidate to compare the baseline to.'
        ),
        required=True
    )
    parser.add_argument(
        '--device',
        metavar='DEVICE',
        type=str,
        default='cpu',
        help=(
            'Device to run the impact analysis on. '
            'E.g. "cpu", "cuda:2". '
            'If "cpu" is slected, it will use 8 workers.'
        ),
    )
    parser.add_argument(
        '--max-signal-length-baseline',
        metavar='DURATION_IN_SEC_BASELINE',
        type=int,
        default=None,
        help=(
            'Maximum length of input signal for the baseline model '
            'in seconds. '
            'If the input signal is longer, '
            'only its last samples '
            'matching the provided duration will be used. '
            'This will add a `-{max-input-length}s` '
            'to the model ID in the tests.'
        ),
    )
    parser.add_argument(
        '--max-signal-length-candidate',
        metavar='DURATION_IN_SEC_CANDIDATE',
        type=int,
        default=None,
        help=(
            'Maximum length of input signal for the candidate model '
            'in seconds. '
            'If the input signal is longer, '
            'only its last samples '
            'matching the provided duration will be used. '
            'This will add a `-{max-input-length}s` '
            'to the model ID in the tests.'
        ),
    )
    parser.add_argument(
        '--impact',
        metavar='IMPACT',
        type=str,
        default=None,
        help=(
            "Select an impact analysis. "
            "Use 'list' to see all available impact analyses. "
            "If not provided, "
            "it executes all impact analyses."
        ),
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
