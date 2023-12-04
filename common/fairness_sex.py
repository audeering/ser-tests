import os
import typing
import yaml

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import audb
import audeer
import audformat
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
    r"""Execute actual correctness tests.

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

    dfs = []
    predictions = []
    results = []
    for database in databases:

        name = database['name']
        version = database['version']
        if 'tables' in database:
            tables = database['tables']
        else:
            tables = [database['table']]
        column = database['column']
        sex_table = database['sex_table']
        sex_column = database['sex_column']
        if 'sex_mapping' in database:
            sex_mapping = database['sex_mapping']
        else:
            sex_mapping = None

        db = audb.load(
            name,
            version=version,
            tables=tables,
            sampling_rate=16000,
            mixdown=True,
            format='wav',
            verbose=False,
            num_workers=4,
        )
        df = sum([db[table] for table in tables], start=audformat.Table()).get()
        # Filter for available class labels
        if condition in CATEGORY_LABELS:
            class_labels = CATEGORY_LABELS[condition]
            df = df[df[column].isin(class_labels)]

        # Add gender
        db_sex = audb.load(
            name,
            version=version,
            tables=[sex_table],
            only_metadata=True,
            sampling_rate=16000,
            mixdown=True,
            format='wav',
            verbose=False,
            num_workers=4,
        )
        df['sex'] = db_sex[sex_table][sex_column].get(
            index=df.index,
            map=sex_mapping,
        )
        prediction = run_model(model, df.index, database)
        prediction = prediction[condition]
        prediction.index = df.index

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
                group_column='sex',
                samples=1000,
            )
            df.to_pickle(balance_cache_file)

        # Filter for relevant available labels or sex annotation
        df = df.dropna(subset=[column, 'sex'])
        prediction = prediction.loc[df.index]
        truth = df[column]
        table_name = database.get('table_alias', '-'.join(tables))
        task = f'{name}-{version}-{table_name}'
        result = test_metrics(
            metrics,
            task,
            df['sex'],
            truth,
            prediction,
            result_dir,
            model,
        )

        results.append(result)
        predictions.append(prediction)
        dfs.append(df)

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

        if 'Female' in metric['name']:
            metric['sex_group'] = 'female'
        elif 'Male' in metric['name']:
            metric['sex_group'] = 'male'

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

        elif 'Recall Per Class' in metric['name']:
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
                        truth.astype('float32'),
                        prediction.astype('float32'),
                        fit=True,
                    )
                )
            metric['plot'] = True
        metric['display'] = lambda x: f'{x:.2f}'

    return databases, metrics


def test_metrics(
        metrics: typing.Sequence,
        task: str,
        sex: typing.Union[typing.Sequence, pd.Series],
        truth: typing.Union[typing.Sequence, pd.Series],
        prediction: typing.Union[typing.Sequence, pd.Series],
        result_dir: str,
        model: audonnx.Model,
) -> typing.Dict:
    r"""Execute calculation of metrics.

    If the metric is a plot,
    it will store the plot file
    in ``result_dir``.

    Otherwise it will return the metric calculation
    as a dataframe
    using the task as index entry
    within a dictionary,
    named after the metric.

    """
    if model.mode != 'classification':
        truth = truth.astype('float32')

    results = {}
    for metric in metrics:

        df = pd.DataFrame(index=[task])
        df.index.name = 'Data'

        sex_groups = list(sex.unique())


        # Store plot if metric has one
        if metric['plot']:
            for sex_group in sex_groups:
                index = sex[sex == sex_group].index
                metric['function'](
                    truth.loc[index],
                    prediction.loc[index],
                )
                name = metric['name'].lower().replace(' ', '-')
                result_file = os.path.join(
                    result_dir,
                    figure_filename(name, task, group=sex_group),
                )
                title = split_figure_title(task, max_length=50)
                plt.title(f'{title}\n{sex_group}')
                fig = plt.gcf()
                fig.set_size_inches(4.8, 3.6)
                plt.tight_layout()
                plt.savefig(result_file)
                plt.close()
            # Continue if we have only the plot
            continue

        else:

            combined_result = metric['function'](
                truth, prediction
            )
            index = sex[sex==metric['sex_group']].index
            sex_group_result = metric['function'](
                truth.loc[index],
                prediction.loc[index],
            )

        if not isinstance(combined_result, dict):
            combined_result = {'': combined_result}
            sex_group_result = {'': sex_group_result}

        for class_label, combined_score in combined_result.items():
            sex_group_score = sex_group_result[class_label]
            if class_label != '':
                column_name = class_label
                display_name = f'{metric["short"]} {column_name}'
            else:
                column_name = metric['name']
                display_name = metric['name']

            threshold = metric['threshold']
            if sex_group_score is None or combined_score is None:
                score = None
                passed = None
            else:
                score = sex_group_score - combined_score
                # Round to ensure reproducibility of results
                score = np.round(score, 3)
                passed = metric['comparison'](score, threshold)
                score = metric['display'](score)

            # Store results to dataframe to save later as CSV
            df[column_name] = score
            print_result(task, display_name, score, passed)
        
        results[metric['name']] = df

    return results
