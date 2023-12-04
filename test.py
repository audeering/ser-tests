import argparse
import importlib
import os
import sys
import typing

import numpy as np
import pandas as pd

import audeer
import audinterface
import audmodel
import audobject
import audonnx

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CURRENT_DIR, 'common'))
from common import (  # noqa: E402
    CATEGORY_LABELS,
    model_information,
    model_name,
    print_title,
    print_summary,
    Tests,
)
from common.random_model import (  # noqa: E402
    RandomUniformCategorical,
    RandomGaussian,
)


def main():
    r"""Load a model and execute the tests."""
    args = parse_arguments()
    condition = args.condition
    device = args.device
    test = args.test
    model_id = args.model
    max_signal_length = args.max_signal_length
    # Check if test is available or if we should list tests
    tests = available_tests(condition)
    if test == 'list':
        print(f'Available tests are: {", ".join(tests)}')
        return
    if test is not None:
        if test not in tests:
            raise ValueError(
                f"The selected test '{test}' is not available. "
                f"Possible tests are: {', '.join(tests)}."
            )
        tests = [test]

    if model_id is None:
        model_ids, tuning_params = available_models(condition)
    else:
        model_ids = [model_id]
        tuning_params = [{}]

    for model_id, tuning_param in zip(model_ids, tuning_params):

        # Fix dtypes of tuning parameters as we load from CSV
        if 'max_signal_length' in tuning_param:
            tuning_param['max_signal_length'] = int(
                tuning_param['max_signal_length']
            )

        # Assign tuning parameters
        # if not loaded from cache
        # and if assigned
        if (
                'max_signal_length' not in tuning_param
                and max_signal_length is not None
        ):
            tuning_param['max_signal_length'] = max_signal_length

        # Load model
        model = get_model_interface(model_id, tuning_param, device, condition)
        run_tests(model, condition, tests)


def available_models(
        condition: str
) -> typing.Tuple[typing.List[str], typing.List[typing.Dict]]:
    r"""List all previously tested models for the given condition.

    Args:
        condition: ``'arousal'``, ``'dominance'``,
            ``'valence'``, or ``'emotion'``

    Returns:
        list of model IDs

    """
    result_dir = os.path.join(
        CURRENT_DIR,
        f'docs/results/test/{condition}/',
    )
    model_names = audeer.list_dir_names(result_dir, basenames=True)
    model_ids = []
    tuning_params = []
    for _model_name in model_names:
        model_id = _model_name
        tuning_param = {}
        if len(_model_name.split('-')) == 3:
            # Model with tuning params
            model_id = '-'.join(_model_name.split('-')[:2])
            model_tuning_file = os.path.join(
                result_dir,
                _model_name,
                'model-tuning.csv',
            )
            df = pd.read_csv(model_tuning_file)
            for n in range(len(df)):
                tuning_param[df.iloc[n]['Entry']] = df.iloc[n]['Value']
        model_ids.append(model_id)
        tuning_params.append(tuning_param)

    return model_ids, tuning_params


def available_tests(condition: str) -> typing.List[str]:
    r"""List all available tests.

    Args:
        condition: ``'arousal'``, ``'dominance'``,
            ``'valence'``, or ``'emotion'``
    Returns:
        list of test names

    """
    test_dir = os.path.join(CURRENT_DIR, 'test', condition)
    tests = audeer.list_file_names(test_dir, filetype='yaml')
    return [audeer.basename_wo_ext(t) for t in tests]


def get_model_interface(model_id, tuning_param, device, condition):
    r"""
    Return an interface for the model on the desired device and condition
    """
    if model_id == 'random-gaussian':
        model = RandomGaussian()
        model.condition = condition
        return model
    if model_id == 'random-categorical':
        model = RandomUniformCategorical()
        model.condition = condition
        return model

    model_root = audmodel.load(model_id)
    onnx_model = audonnx.load(model_root, device=device)

    if condition in ['arousal', 'dominance', 'valence']:

        labels = ['arousal', 'dominance', 'valence']
        default_output = 'logits'

        arousal_output_name, arousal_index = onnx_output_regression(
            onnx_model, default_output, 'arousal'
        )
        dominance_output_name, dominance_index = onnx_output_regression(
            onnx_model, default_output, 'dominance'
        )
        valence_output_name, valence_index = onnx_output_regression(
            onnx_model, default_output, 'valence'
        )
        if arousal_output_name is None:
            def process_func(x, sr, tuning_param):
                if 'max_signal_length' in tuning_param:
                    max_length = int(
                        tuning_param['max_signal_length'] * sr
                    )
                    x = x[:, -max_length:]
                y = onnx_model(x, sr)
                return (
                    y[0][arousal_index],
                    y[0][dominance_index],
                    y[0][valence_index],
                )
        else:
            def process_func(x, sr, tuning_param):
                if 'max_signal_length' in tuning_param:
                    max_length = int(
                        tuning_param['max_signal_length'] * sr
                    )
                    x = x[:, -max_length:]
                y = onnx_model(x, sr)
                y = onnx_model(x, sr)
                return (
                    y[arousal_output_name][0][arousal_index],
                    y[dominance_output_name][0][dominance_index],
                    y[valence_output_name][0][valence_index],
                )

    elif condition in ['emotion']:

        labels = ['emotion']
        default_output = 'logits'
        default_labels = CATEGORY_LABELS[condition]
        output_name, output_labels = onnx_output_category(
            onnx_model, model_root, default_output, default_labels
        )
        if output_name is None:
            def process_func(x, sr, tuning_param):
                if 'max_signal_length' in tuning_param:
                    max_length = int(
                        tuning_param['max_signal_length'] * sr
                    )
                    x = x[:, -max_length:]
                y = onnx_model(x, sr)
                idx = np.argmax(y[0])
                return output_labels[idx]
        else:
            def process_func(x, sr, tuning_param):
                if 'max_signal_length' in tuning_param:
                    max_length = int(
                        tuning_param['max_signal_length'] * sr
                    )
                    x = x[:, -max_length:]
                y = onnx_model(x, sr)
                idx = np.argmax(y[output_name][0])
                return output_labels[idx]

    if device == 'cpu':
        num_workers = 8
    else:
        num_workers = 1

    model = audinterface.Feature(
        labels,
        process_func=process_func,
        process_func_args={
            'tuning_param': tuning_param
        },
        num_workers=num_workers,
        verbose=False,
    )
    model.uid = model_id
    model.condition = condition
    model.tuning_params = tuning_param

    if condition in ['arousal', 'dominance', 'valence']:
        model.mode = 'regression'
    elif condition in ['emotion']:
        model.mode = 'classification'
    return model


def onnx_output_category(
    onnx_model: audonnx.Model, model_root: str, default_name: str,
    required_labels: typing.List[str]
) -> typing.Tuple[str, typing.List[str]]:
    r"""Get onnx output name and onnx output labels for category

    Args:
        onnx_model: the onnx model to pull output information from
        model_root: the path root of the onnx model
        default_name: the default output name to use if no other
            information is available
        required_labels: the output labels that are required
    Returns:
        the onnx output name and the list of labels in the order the model
            returns them
    """
    # onnx model only outputs dict when the number of outputs is >1
    # -> handle case of there being only one output by setting the
    #    label output name to None
    if len(onnx_model.outputs) == 1:
        onnx_output_name, onnx_output = list(onnx_model.outputs.items())[0]
        if all([label in onnx_output.labels for label in required_labels]):
            output_labels = onnx_output.labels
        else:
            # Try to get label order information from encoder.yaml file in
            #  model root
            # (e.g. for model 51b86127-ee8d-a66b-2fbe-9df87eb3ecf1)
            encoder_path = os.path.join(model_root, 'encoder.yaml')
            if os.path.exists(encoder_path):
                encoder = audobject.from_yaml(encoder_path)
                output_labels = encoder.labels
            else:
                output_labels = required_labels
        return None, output_labels

    if default_name in onnx_model.outputs:
        onnx_output = onnx_model.outputs[default_name]
        if all([label in onnx_output.labels for label in required_labels]):
            output_labels = onnx_output.labels
            return default_name, output_labels

    for onnx_output_name, onnx_output in onnx_model.outputs.items():
        # either all required labels correspond to labels of one onnx output
        # or each required label corresponds to a separate onnx output of
        # dimension 1
        if all([label in onnx_output.labels for label in required_labels]):
            return onnx_output_name, onnx_output.labels
    raise ValueError(
        "The selected model lacks the required output label")


def onnx_output_regression(
    onnx_model: audonnx.Model, default_name: str,
    required_label: typing.List[str]
) -> typing.Tuple[str, int]:
    r"""Get onnx output name and index for the regression value

    Args:
        onnx_model: the onnx model to pull output information from
        default_name: the default output name to use if no other
            information is available
        required_label: the output label that is required
    Returns:
        the onnx output name and the index of the required label
    """
    # onnx model only outputs dict when the number of outputs is >1
    # -> handle case of there being only one output by setting the
    #    label output name to None
    if len(onnx_model.outputs) == 1:
        onnx_output_name, onnx_output = list(onnx_model.outputs.items())[0]
        if required_label in onnx_output.labels:
            return None, onnx_output.labels.index(required_label)
        elif (required_label == onnx_output_name):
            return None, 0
        else:
            raise ValueError(
                "The selected model lacks the required output label")

    if default_name in onnx_model.outputs:
        onnx_output = onnx_model.outputs[default_name]
        if required_label in onnx_output.labels:
            return default_name, onnx_output.labels.index(required_label)
        elif required_label == onnx_output_name:
            return default_name, 0

    for onnx_output_name, onnx_output in onnx_model.outputs.items():
        if required_label in onnx_output.labels:
            return onnx_output_name, onnx_output.labels.index(required_label)
        elif required_label == onnx_output_name:
            return onnx_output_name, 0

    raise ValueError(
        "The selected model lacks the required output label")


def parse_arguments():
    parser = argparse.ArgumentParser(description='Model tests')
    parser.add_argument(
        'condition',
        choices=['arousal', 'dominance', 'emotion', 'valence'],
        help='Emotion dimension or categories.',
    )
    parser.add_argument(
        '--model',
        metavar='MODEL_ID',
        type=str,
        default=None,
        help=(
            'ONNX model ID. '
            'If none is provided, '
            'will run the test on all previous models. '
        ),
    )
    parser.add_argument(
        '--device',
        metavar='DEVICE',
        type=str,
        default='cpu',
        help=(
            'Device to run the tests on. '
            'E.g. "cpu", "cuda:2". '
            'If "cpu" is slected, it will use 8 workers.'
        ),
    )
    parser.add_argument(
        '--max-signal-length',
        metavar='DURATION_IN_SEC',
        type=int,
        default=None,
        help=(
            'Maximum length of input signal in seconds. '
            'If the input signal is longer, '
            'only its last samples '
            'matching the provided duration will be used. '
            'This will add a `-{uid}` corresponding to the '
            'maximum signal length tuning parameter '
            'to the model ID in the tests.'
        ),
    )
    parser.add_argument(
        '--test',
        metavar='TEST',
        type=str,
        default=None,
        help=(
            "Select a test. "
            "Use 'list' to see all available tests. "
            "If not provided, "
            "it executes all tests."
        ),
    )
    args = parser.parse_args()
    return args


def run_tests(model, condition, tests):
    # Set passed/failed/skipped test counter to 0
    Tests.reset()

    print_title(model_name(model), condition)

    # Create result dir
    result_dir = os.path.join(
        CURRENT_DIR,
        'docs',
        'results',
        'test',
        condition,
        model_name(model),
    )
    audeer.mkdir(result_dir)

    # Store model info
    model_info_file = os.path.join(result_dir, 'model-info.csv')
    model_params_file = os.path.join(result_dir, 'model-params.csv')
    model_tuning_file = os.path.join(result_dir, 'model-tuning.csv')
    df_info, df_params, df_tuning = model_information(model)
    df_info.to_csv(model_info_file)
    df_params.to_csv(model_params_file)
    if len(df_tuning) > 0:
        df_tuning.to_csv(model_tuning_file)

    # Execute all requested tests
    for test in tests:
        module = importlib.import_module(f'common.{test}')
        run = getattr(module, 'run')
        run(model, condition)

    print_summary(Tests)


if __name__ == '__main__':
    main()
