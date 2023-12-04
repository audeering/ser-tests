import os
import typing
import yaml

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import parselmouth

import audb
import audeer
import audinterface
import audmetric
import audonnx
import audplot

from . import (
    balance_data,
    CATEGORY_LABELS,
    figure_filename,
    model_name,
    print_header,
    print_result,
    run_model,
    split_figure_title,
)

from common.comparisons import abs_lt_or_nan
from common.metrics import (
    mean_directional_error,
    precision_per_bin,
    recall_per_bin,
)
from common.plots import plot_confusion_matrix


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CURRENT_TEST = audeer.basename_wo_ext(os.path.abspath(__file__))
TEST_NAME = audeer.basename_wo_ext(__file__)


def run(
        model: audonnx.Model,
        condition: str,
):
    r"""Execute actual fairness tests.

    Args:
        model: model to run
        condition: test condition, e.g. ``'arousal'``

    """
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
        speaker_table = database['speaker_table']
        speaker_column = database['speaker_column']

        db = audb.load(
            name,
            version=version,
            tables=[table, speaker_table],
            sampling_rate=16000,
            mixdown=True,
            format='wav',
            verbose=False,
            num_workers=4,
        )
        df = db[table].get()
        # Filter for available class labels
        if condition in CATEGORY_LABELS:
            class_labels = CATEGORY_LABELS[condition]
            df = df[df[column].isin(class_labels)]
            
        # Calculate pitch as truth and cache pitch results
        pitch_model = audinterface.Process(process_func=pitch)
        pitch_model.uid = 'pitch-praat'
        pitch_model.mode = 'regression'
        truth_pitch = run_model(pitch_model, df.index)
        # restore index back to filewise in case df.index is filewise and the
        # truth index has changed
        truth_pitch.index = df.index
        df['pitch'] = truth_pitch

        prediction = run_model(model, df.index, database)
        prediction.index = df.index
        prediction = prediction[condition]

        # Filter for relevant available labels and speaker labels, and pitch
        df['speaker'] = db[speaker_table][speaker_column].get(index=df.index)
        df = df.dropna(subset=['speaker', column, 'pitch'])

        # Filter out speakers with few samples
        speaker_samples = df.groupby('speaker')[column].count()
        n_sample_threshold = 25
        speakers_too_few_samples = speaker_samples[
            speaker_samples < n_sample_threshold
        ].index.to_list()
        df = df[~df['speaker'].isin(speakers_too_few_samples)]
        df['average_pitch'] = df.groupby('speaker')['pitch'].transform('mean')
        # Group each speaker into one of the pitch groups
        pitch_tuples = [(0, 145), (145,190), (190, 350)]
        pitch_groups = ['low', 'medium', 'high']
        pitch_bins = pd.IntervalIndex.from_tuples(pitch_tuples)
        pitch_label_mapping = {
            interval: pitch_groups[i] for i, interval in enumerate(pitch_bins)
        }
        df['pitch_interval'] = pd.cut(
            df['average_pitch'],
            pitch_bins,
        )
        df['pitch_group'] = df['pitch_interval'].map(pitch_label_mapping)

        # Ensure that the test set ground truth
        # has an equal distribution
        balance_cache_file = audeer.path(
            CURRENT_DIR,
            '..',
            'cache',
            'data-balancing',
            CURRENT_TEST,
            f'{name}-{version}-{condition}.pkl',
        )
        original_cache_file = audeer.path(
            CURRENT_DIR,
            '..',
            'cache',
            'data-balancing',
            CURRENT_TEST,
            f'{name}-{version}-{condition}-original.pkl',
        )
        if os.path.exists(balance_cache_file):
            df = pd.read_pickle(balance_cache_file)
        else:
            audeer.mkdir(os.path.dirname(balance_cache_file))
            df = df.dropna()
            df.to_pickle(original_cache_file)
            df = balance_data(
                df,
                column=column,
                group_column='pitch_group',
                samples=1000,
            )
            df.to_pickle(balance_cache_file)
        prediction = prediction.loc[df.index]
        truth = df[column]

        task = f'{name}-{version}-{table}'
        result = test_metrics(
            metrics,
            task,
            df['pitch_group'],
            prediction,
            truth,
            result_dir,
            model,
        )

        results.append(result)

    # Store results to docs/ folder
    for metric in metrics:
        name = metric['name']
        # Skip if metric has not added any results
        if name not in results[0]:
            continue

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
        condition: test condition, e.g. ``'arousal'``

    Returns:
        * databases
        * metrics

    """


    # Minimum number of samples required per bin for 1000 samples per group
    # calculated as CDF[NormalDistribution[0.5, 1/6], 0.25] * 1000
    # (Expected number of samples in the range of [-inf, 0.25] for a gaussian
    # distribution with mean=0.5 and std=1/6, when sampling 1000 times)
    min_samples = 67

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

        # Match pitch group to be examined by metric
        if 'Medium Pitch' in metric['name']:
            metric['pitch_group'] = 'medium'
        elif 'Low Pitch' in metric['name']:
            metric['pitch_group'] = 'low'
        elif 'High Pitch' in metric['name']:
            metric['pitch_group'] = 'high'


        if 'Mean Absolute Error' in metric['name']:
            metric['function'] = audmetric.mean_absolute_error
            metric['comparison'] = abs_lt_or_nan

        elif 'Mean Directional Error' in metric['name']:
            metric['function'] = mean_directional_error
            metric['comparison'] = abs_lt_or_nan

        elif 'Concordance Correlation Coeff' in metric['name']:
            metric['function'] = audmetric.concordance_cc
            metric['comparison'] = abs_lt_or_nan

        elif 'Recall Per Bin' in metric['name']:
            metric['function'] = (
                lambda truth, prediction:
                recall_per_bin(truth, prediction, bins=4,
                               min_samples=min_samples)
            )
            metric['comparison'] = abs_lt_or_nan

        elif 'Precision Per Bin' in metric['name']:
            metric['function'] = (
                lambda truth, prediction:
                precision_per_bin(truth, prediction, bins=4,
                                  min_samples=min_samples)
            )
            metric['comparison'] = abs_lt_or_nan

        elif 'Recall Per Class' in metric['name'] :
            metric['function'] = audmetric.recall_per_class
            metric['comparison'] = abs_lt_or_nan

        elif 'Precision Per Class' in metric['name']:
            metric['function'] = audmetric.precision_per_class
            metric['comparison'] = abs_lt_or_nan

        elif 'Unweighted Average Recall' in metric['name']:
            metric['function'] = audmetric.unweighted_average_recall
            metric['comparison'] = abs_lt_or_nan

        elif 'Unweighted Average Precision' in metric['name']:
            metric['function'] = audmetric.unweighted_average_precision
            metric['comparison'] = abs_lt_or_nan

        elif metric['name'] == 'Visualization':
            if condition == 'emotion':
                metric['function'] = plot_confusion_matrix
            else:
                metric['function'] = (
                    lambda truth, prediction:
                    audplot.scatter(
                        truth,
                        prediction,
                        fit=True,
                    )
                )
            metric['plot'] = True

        metric['display'] = lambda x: f'{x:.2f}'

    return databases, metrics


def pitch(signal, sampling_rate):
    r"""Pitch of signal.

    It assumes we are interested in pitch of speech,
    and sets everything below 50 Hz,
    and everything above 350 Hz
    to NaN.

    """
    sound = parselmouth.Sound(signal[0, :], sampling_rate)
    pitch = sound.to_pitch()
    pitch = pitch.selected_array['frequency']
    pitch[pitch == 0] = np.NaN
    pitch = np.nanmean(pitch)
    if pitch < 50 or pitch > 350:
        pitch = np.NaN
    return pitch


def test_metrics(
        metrics: typing.Sequence,
        task: str,
        pitch: typing.Union[typing.Sequence, pd.Series],
        prediction: typing.Union[typing.Sequence, pd.Series],
        truth: typing.Union[typing.Sequence, pd.Series],
        result_dir: str,
        model: audonnx.Model,
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
    if model.mode != 'classification':
        truth = truth.astype('float32')

    results = {}
    for metric in metrics:

        df = pd.DataFrame(index=[task])
        df.index.name = 'Data'

        pitch_groups = list(pitch.unique())

        # Store plot if metric has one
        if metric['plot']:
            for pitch_group in pitch_groups:
                index = pitch[pitch==pitch_group].index
                metric['function'](
                    truth.loc[index],
                    prediction.loc[index],
                )
                name = metric['name'].lower().replace(' ', '-')
                result_file = os.path.join(
                    result_dir,
                    figure_filename(name, task, group=pitch_group.replace(' ', '_')),
                )
                title = split_figure_title(task, max_length=50)
                plt.title(f'{title}\n{pitch_group} pitch')
                fig = plt.gcf()
                fig.set_size_inches(4.8, 3.6)
                plt.tight_layout()
                plt.savefig(result_file)
                plt.close()
            
            continue
        else:
            # calculate score of groups combined first

            combined_result = metric['function'](
                truth, prediction
            )

            index = pitch[pitch==metric['pitch_group']].index
            pitch_group_result = metric['function'](
                truth.loc[index],
                prediction.loc[index],
            )

        if not isinstance(combined_result, dict):
            combined_result= {'': combined_result}
            pitch_group_result = {'': pitch_group_result}

        for class_label, combined_score in combined_result.items():
            pitch_group_score = pitch_group_result[class_label]
            if class_label != '':
                column_name = class_label
                display_name = f'{metric["short"]} {column_name}'
            else:
                column_name = metric['name']
                display_name = metric['name']

            threshold = metric['threshold']
            if pitch_group_score is None or combined_score is None:
                score = None
                passed = None
            else:
                score = pitch_group_score - combined_score
                # Round to ensure reproducibility of results
                score = np.round(score, 3)
                passed = metric['comparison'](score, threshold)
                score = metric['display'](score)

            # Store results to dataframe to save later as CSV
            df[column_name] = score
            print_result(task, display_name, score, passed)
            
        results[metric['name']] = df

    return results
