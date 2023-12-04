r"""Test robustness background noise."""
import operator
import os
import typing
import yaml

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import audb
import audeer
import audmetric
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
from common.comparisons import abs_lt
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

        name = database['name']
        version = database['version']
        table = database['table']
        column = database['column']

        db = audb.load(
            name,
            version=version,
            tables=[table],
            sampling_rate=16000,
            mixdown=True,
            format='wav',
            verbose=False,
            num_workers=4,
        )
        truth = db[table][column].get()
        truth = truth.dropna()
        if condition in CATEGORY_LABELS:
            class_labels = CATEGORY_LABELS[condition]
            truth = truth[truth.isin(class_labels)]

        task = f'{name}-{version}-{table}'
        result = test_metrics(
            metrics,
            condition,
            task,
            truth,
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

        if 'White Noise' in metric['name']:
            noise_type = 'white'
        elif 'Babble Noise' in metric['name']:
            noise_type = 'babble'
        elif 'Environmental Noise' in metric['name']:
            noise_type = 'environmental'
        elif 'Music' in metric['name']:
            noise_type = 'music'
        elif 'Coughing' in metric['name']:
            noise_type = 'coughing'
        elif 'Sneezing' in metric['name']:
            noise_type = 'sneezing'
        else:
            raise ValueError(f'Test {metric["name"]} not known.')

        if 'CCC' in metric['name']:
            metric['function'] = audmetric.concordance_cc
            metric['comparison'] = operator.gt
            metric['difference_metric'] = True
        elif 'UAR' in metric['name']:
            metric['function'] = audmetric.unweighted_average_recall
            metric['comparison'] = operator.gt
            metric['difference_metric'] = True
        elif 'UAP' in metric['name']:
            metric['function'] = audmetric.unweighted_average_precision
            metric['comparison'] = operator.gt
            metric['difference_metric'] = True
        elif 'Change Average Value' in metric['name']:
            metric['function'] = (
                lambda truth, prediction:
                prediction.mean()
            )
            metric['comparison'] = abs_lt
            metric['difference_metric'] = True
        elif 'Percentage Unchanged Predictions' in metric['name']:
            metric['function'] = percentage_of_identity
            metric['comparison'] = operator.gt
            metric['difference_metric'] = False

        elif 'Visualization' in metric['name']:
            metric['function'] = plot_shift
            metric['plot'] = True

        metric['transform'] = auglib.transform.Function(
            function=noise_transform,
            function_args={
                'noise_type': noise_type,
            },
        )
        metric['display'] = lambda x: f'{x:.2f}'

    return databases, metrics


def test_metrics(
        metrics: typing.Sequence,
        condition: str,
        task: str,
        truth: typing.Union[typing.Sequence, pd.Series],
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

        # Apply augmentation and get metric scores
        augment = auglib.Augment(metric['transform'])
        truth_augmented = augment.augment(truth)
        prediction_truth = run_model(model, truth.index)
        prediction_truth = prediction_truth[condition]
        prediction_augmented = run_model(model, truth_augmented.index)
        prediction_augmented = prediction_augmented[condition]

        if metric['plot']:
            truth_label = 'Prediction clean'
            if 'White Noise' in metric['name']:
                prediction_label = 'Prediction white noise'
            elif 'Babble Noise' in metric['name']:
                prediction_label = 'Prediction babble noise'
            elif 'Environmental Noise' in metric['name']:
                prediction_label = 'Prediction environmental noise'
            elif 'Music' in metric['name']:
                prediction_label = 'Prediction music'
            elif 'Coughing' in metric['name']:
                prediction_label = 'Prediction coughing'
            elif 'Sneezing' in metric['name']:
                prediction_label = 'Prediction sneezing'

            # Plot regression and classification results differently
            if model.mode == 'classification':
                class_labels = CATEGORY_LABELS[model.condition]
                score = metric['function'](
                    prediction_truth,
                    prediction_augmented.dropna(),
                    class_labels,
                    truth_label=truth_label,
                    prediction_label=prediction_label,
                )
            else:
                score = metric['function'](
                    prediction_truth,
                    prediction_augmented,
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
            if metric['difference_metric']:
                # Change in metric
                score = (
                    metric['function'](truth, prediction_augmented)
                    - metric['function'](truth, prediction_truth)
                )
            else:
                score = metric['function'](prediction_truth, prediction_augmented)

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


def noise_transform(
        speech: np.array,
        sampling_rate: int,
        noise_type: str,
        *,
        demo: bool = False,
) -> auglib.transform.Base:
    r"""Return the background noise transform."""
    # Use a fixed SNR of 20 dB
    snr = 20

    # Transformatiopns for different noise types
    if noise_type == 'white':
        transform = auglib.transform.WhiteNoiseGaussian()

    elif noise_type == 'babble':
        # Load speech from musan and create babble noise from it
        if demo:
            # Load only 10 files from musan
            media = '.*speech-librivox-000\d'
        else:
            media = None
        db = audb.load(
            'musan',
            tables='speech',
            media=media,
            version='1.0.0',
            format='wav',
            sampling_rate=16000,
            mixdown=True,
            verbose=False,
        )
        transform = auglib.transform.Mix(
            auglib.observe.List(db.files, draw=True),
            gain_aux_db=auglib.observe.IntUni(-20, -13),
            num_repeat=auglib.observe.IntUni(4, 7),
            read_pos_aux=auglib.observe.FloatUni(0, 1),
            unit='relative',
            loop_aux=True,
        )

    elif noise_type == 'environmental':

        if demo:
            media = ['noise/free-sound/noise-free-sound-0003.wav']
        else:
            media = None

        db = audb.load(
            'musan',
            tables='noise',
            media=media,
            version='1.0.0',
            format='wav',
            sampling_rate=16000,
            mixdown=True,
            verbose=False,
        )
        transform = auglib.transform.Mix(
            auglib.observe.List(db.files, draw=True),
            read_pos_aux=auglib.observe.FloatUni(0, 1),
            unit='relative',
            loop_aux=True,
        )

    elif noise_type == 'music':

        if demo:
            media = ['music/fma/music-fma-0000.wav']
        else:
            media = None

        db = audb.load(
            'musan',
            tables='music',
            media=media,
            version='1.0.0',
            format='wav',
            sampling_rate=16000,
            mixdown=True,
            verbose=False,
        )
        transform = auglib.transform.Mix(
            auglib.observe.List(db.files, draw=True),
            read_pos_aux=auglib.observe.FloatUni(0, 1),
            unit='relative',
            loop_aux=True,
        )

    elif noise_type == 'coughing':

        snr = 10

        if demo:
            media = ['coughing/kopzxumj430_40.94-41.8.wav']
        else:
            media = None

        db = audb.load(
            'cough-speech-sneeze',
            tables='files',
            media=media,
            version='2.0.1',
            format='wav',
            sampling_rate=16000,
            mixdown=True,
            verbose=False,
        )
        df = db['files'].get()
        df = df[df.category == 'coughing']
        transform = auglib.transform.Mix(
            auglib.observe.List(list(df.index), draw=True),
            write_pos_base=auglib.observe.FloatUni(0, 1),
            unit='relative',
            loop_aux=False,
        )

    elif noise_type == 'sneezing':

        snr = 10

        if demo:
            media = ['sneezing/mnmsksalfui_286.02-289.61.wav']
        else:
            media = None

        db = audb.load(
            'cough-speech-sneeze',
            tables='files',
            media=media,
            version='2.0.1',
            format='wav',
            sampling_rate=16000,
            mixdown=True,
            verbose=False,
        )
        df = db['files'].get()
        df = df[df.category == 'sneezing']
        transform = auglib.transform.Mix(
            auglib.observe.List(list(df.index)),
            write_pos_base=auglib.observe.FloatUni(0, 1),
            unit='relative',
            loop_aux=False,
        )

    augment = auglib.Augment(transform)
    # Start from empty signal to adjust SNR
    noise = np.zeros(speech.shape)
    noise = augment(noise, sampling_rate)
    noise = np.atleast_2d(noise)
    # Adjust level of augmented signal to match original level
    rms_speech = np.sqrt(np.mean(speech[0, :] ** 2))
    rms_noise = np.sqrt(np.mean(noise[0, :] ** 2))
    scale = 10 ** (snr / 20) * rms_noise / rms_speech
    signal_augmented = (scale * speech + noise) / 2
    # Make sure we have the same level as the input signal
    rms_augmented = np.sqrt(np.mean(signal_augmented[0, :] ** 2))
    scaling_factor = rms_speech / rms_augmented
    signal_augmented = scaling_factor * signal_augmented
    # Ensure we don't clip
    if signal_augmented.max() > 1:
        signal_augmented /= (signal_augmented.max() + np.finfo(np.float32).eps)
    return signal_augmented
