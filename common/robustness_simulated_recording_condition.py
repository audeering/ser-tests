r"""Test robustness simulated recording condition."""
import operator
import os
import typing
import yaml

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import audb
import audeer
import audonnx
import auglib

from . import (
    CATEGORY_LABELS,
    figure_filename,
    model_name,
    print_header,
    print_result,
    run_model,
)
from common.metrics import percentage_of_identity
from common.plots import plot_shift


auglib.seed(0)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CURRENT_TEST = audeer.basename_wo_ext(os.path.abspath(__file__))
TEST_NAME = audeer.basename_wo_ext(__file__)


def run(
        model: audonnx.Model,
        condition: str,
):
    r"""Execute tests.

    Args:
        model: model to run
        condition: test condition, e.g. ``'arousal'``

    """
    if model.uid == 'random-rater':
        return 1

    print_header(TEST_NAME)

    # Load metrics and data to test from config file
    databases, metrics = load_config(condition)

    result_dir = os.path.join(
        CURRENT_DIR,
        '..',
        'docs',
        'results',
        'test',
        condition,
        model_name(model),
        CURRENT_TEST,
    )
    audeer.rmdir(result_dir)
    audeer.mkdir(result_dir)

    results = []
    for database in databases:
        # The database may contain very many samples
        # so we restrict it to a lower number
        num_samples = 5000

        name = database['name']
        version = database['version']
        table = database['table']

        db = audb.load(
            name,
            version=version,
            tables=[table],
            sampling_rate=16000,
            mixdown=True,
            format='wav',
            verbose=False,
            num_workers=4,
            only_metadata=True,
            full_path=False,
        )
        # set df to contain two file columns, with the microphone
        # name as the suffix
        df = db[table].get()
        df_sampled = df.sample(min(num_samples, len(df)), random_state=1)

        task = f'{name}-{version}-{table}'

        db = audb.load(
            name,
            version=version,
            tables=[table],
            media=list(df_sampled.index),
            sampling_rate=16000,
            mixdown=True,
            format='wav',
            verbose=False,
            num_workers=4,
        )
        df_sampled = db[table].get()

        result = test_metrics(
            metrics,
            condition,
            task,
            df_sampled.index,
            model,
            result_dir,
        )

        results.append(result)

    # Store results to docs/ folder
    for metric in metrics:
        name = metric['name']
        # Skip if metric has not added any results
        if name not in results[0]:
            continue

        # Store robustness values
        metric_result = pd.concat([result[name] for result in results])
        filename = name.lower().replace(' ', '-')
        result_file = os.path.join(result_dir, f'{filename}.csv')
        metric_result.to_csv(result_file)


def load_config(condition):
    r"""Load configuration from YAML file.

    This loads the settings from a YAML file
    with the same basename as this script.

    The settings include the databases
    to use with the following dictionary entries:
    * 'name'
    * 'version'
    * 'split'

    In addition, test metrics are loaded
    and corresponding functions
    and test criteria are added.
    The returned metrics list will contain
    dictionaries for every metric
    with some of the following content:
    * 'name' - name of the metric displayed in the test output
    * 'short' - short name of metric used in result table
    * 'function' - metric assuming the two input args truth and prediction
    * 'threshold' - threshold of calculated metric to pass the test
    * 'comparison' - comparison assuming score and threshold as input args
    * 'display' - function to convert score value to string
    * 'plot' - if the metric is returning an image instead of a value

    Args:
        condition: test condition, e.g.``'arousal'``

    Returns:
        * databases
        * metrics

    """
    yaml_file = os.path.join(
        CURRENT_DIR,
        '..',
        'test',
        condition,
        f'{TEST_NAME}.yaml',
    )
    with open(yaml_file, 'r') as fp:
        d = yaml.safe_load(fp)
        metrics = d['Metrics']
        databases = d['Data']
    for metric in metrics:
        metric['plot'] = False
        metric['transform'] = None
        if metric['name'] in [
            'Percentage Unchanged Predictions Simulated Position',
            'Visualization Simulated Position'
        ]:
            ir_type = 'position'
        elif metric['name'] in [
            'Percentage Unchanged Predictions Simulated Room',
            'Visualization Simulated Room'
        ]:
            ir_type = 'location'
        else:
            raise ValueError(f'Test {metric["name"]} not known.')
        if ir_type is not None:
            metric['transform'] = auglib.transform.Function(
                function=ir_transform,
                function_args={
                    'ir_type': ir_type,
                },
            )
            metric['transform_reference'] = auglib.transform.Function(
                function=ir_transform,
                function_args={
                    'ir_type': ir_type,
                    'reference': True
                },
            )
        if 'Percentage Unchanged Predictions' in metric['name']:
            metric['function'] = percentage_of_identity
            metric['comparison'] = operator.gt
        elif 'Visualization' in metric['name']:
            metric['function'] = plot_shift
            metric['plot'] = True

        metric['display'] = lambda x: f'{x:.2f}'

    return databases, metrics


def test_metrics(
        metrics: typing.Sequence,
        condition: str,
        task: str,
        index: pd.Index,
        model: audonnx.Model,
        result_dir: str,
) -> typing.Dict:
    r"""Execute calculation of metrics.

    If the metric is a plot,
    it will store the plot file
    in ``result_dir``.

    Otherwise it will return the metric calculation
    as a dataframe
    using the condition as index entry
    within a dictionary,
    named after the metric.

    """
    results = {}
    for metric in metrics:

        df = pd.DataFrame(index=[task])
        df.index.name = 'Data'
        augment_reference = auglib.Augment(metric['transform_reference'])
        index_ref_augmented = augment_reference.augment(index)
        prediction_a = run_model(model, index_ref_augmented)
        augment = auglib.Augment(metric['transform'])
        index_augmented = augment.augment(index)
        prediction_b = run_model(model, index_augmented)

        if metric['plot']:
            if 'Position' in metric['name']:
                truth_label = 'Pred Base Position'
                prediction_label = 'Pred Diff. Position'
            elif 'Room' in metric['name']:
                truth_label = 'Pred Base Room'
                prediction_label = 'Pred Diff. Room'
            if model.mode == 'classification':
                class_labels = CATEGORY_LABELS[model.condition]
                score = metric['function'](
                    prediction_a[condition],
                    prediction_b[condition].dropna(),
                    class_labels,
                    truth_label=truth_label,
                    prediction_label=prediction_label,
                )
            else:
                score = metric['function'](
                    prediction_a[condition],
                    prediction_b[condition],
                    None,
                    truth_label=truth_label,
                    prediction_label=prediction_label,
                    y_range=(-0.35, 0.35),
                    allowed_range=(-0.05, 0.05),
                    # Present databases with different number of samples
                    # similarly
                    bins=(
                        np.arange(0, 1., 0.03),
                        np.arange(-0.35, 0.35, 0.02)
                    ),
                )
            name = metric['name'].lower().replace(' ', '-')
            result_file = os.path.join(
                result_dir,
                figure_filename(name, task),
            )
            plt.suptitle(task)
            plt.tight_layout()
            plt.savefig(result_file)
            plt.close()
            continue
        else:
            score = metric['function'](
                prediction_a[condition], prediction_b[condition])

        column_name = metric['name']
        display_name = metric['name']
        threshold = metric['threshold']

        # Round to ensure reproducibility of results
        score = np.round(score, 2)

        # Store results to dataframe to save later as CSV
        df[column_name] = score
        passed = metric['comparison'](score, threshold)
        display_score = metric['display'](score)
        print_result(task, display_name, display_score, passed)
        results[metric['name']] = df

    return results


def ir_transform(
        speech: np.array,
        sampling_rate: int,
        ir_type: str,
        *,
        demo: bool = False,
        reference: bool = False,
) -> auglib.transform.Base:
    r"""Return the impulse response transform."""

    if ir_type == 'position':
        # Load irs from mardy
        if reference:
            media = ['data/ir_1_C.wav']
        elif demo:
            media = ['data/ir_2_R.wav']
        else:
            media = [
                'data/ir_1_L.wav',
                'data/ir_1_R.wav',
                'data/ir_2_C.wav',
                'data/ir_2_L.wav',
                'data/ir_2_R.wav',
                'data/ir_3_C.wav',
                'data/ir_3_L.wav',
                'data/ir_3_R.wav',
            ]
        db = audb.load(
            "mardy",
            version="1.0.0",
            tables="files",
            media=media,
            channels=[0],
            sampling_rate=16000,
            verbose=False,
        )
        transform = auglib.transform.Compose(
            [
                auglib.transform.FFTConvolve(
                    auglib.observe.List(db.files, draw=True),
                    keep_tail=False,
                ),
                auglib.transform.NormalizeByPeak(),
            ]
        )

    elif ir_type == 'location':
        if reference:
            media = ['data/air_binaural_booth_0_3.wav']
        elif demo:
            media = ['data/air_binaural_lecture_0_1.wav']
        else:
            media = [
                'data/air_binaural_lecture_0_1.wav',
                'data/air_binaural_meeting_0_3.wav',
                'data/air_binaural_office_0_2.wav',
            ]

        db = audb.load(
            'air',
            tables='rir',
            media=media,
            version='1.4.2',
            format='wav',
            channels=[0],
            sampling_rate=16000,
            verbose=False,
        )
        transform = auglib.transform.Compose(
            [
                auglib.transform.FFTConvolve(
                    auglib.observe.List(db.files, draw=True),
                    keep_tail=False,
                ),
                auglib.transform.NormalizeByPeak(),
            ]
        )
    augment = auglib.Augment(transform)
    return augment(speech, sampling_rate)
