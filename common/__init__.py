from bisect import bisect_left
import os
import random
import typing

import numpy as np
import pandas as pd

import audb
import audeer
import audformat
import audinterface
import audmodel
import auglib


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

CATEGORY_LABELS = {
    'emotion': ['anger', 'happiness', 'neutral', 'sadness']
}
SEED = 0
random.seed(SEED)

# Ensure cuda addresses the GPU devices
# with the BUS IDs
# as shown by nvidia-smi
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

# To ensure the same hash inside the model cache,
# we need to force the same auglib and audb cache
# for all users
if os.path.exists('/cache/audb'):
    audb.config.CACHE_ROOT = '/cache/audb'
if os.path.exists('/cache/auglib'):
    auglib.config.CACHE_ROOT = '/cache/auglib'


class Color():
    green = '\033[92m'
    red = '\033[91m'
    reset = '\033[97m'


class Impacts():
    r"""Counter of positive/neutral/negative/skipped impacts.

    This works like a global variable
    and you can increase the counters
    in every script
    just by importing :class:`Impacts`
    and changing the counter value, e.g.

    .. code-block:: python

        Impacts.positive += 1

    """
    positive = 0
    negative = 0
    neutral = 0
    skipped = 0

    def reset():
        r"""Reset counters to 0."""
        Impacts.positive = 0
        Impacts.negative = 0
        Impacts.neutral = 0
        Impacts.skipped = 0


class Tests():
    r"""Counter of passed/failed/skipped tests.

    This works like a global variable
    and you can increase the counters
    in every script
    just by importing :class:`Tests`
    and changing the counter value, e.g.

    .. code-block:: python

        Tests.passed += 1

    """
    passed = 0
    failed = 0
    skipped = 0

    def reset():
        r"""Reset counters to 0."""
        Tests.passed = 0
        Tests.failed = 0
        Tests.skipped = 0


def anonymize_column(
        df: pd.DataFrame,
        column: str,
) -> pd.DataFrame:
    r"""Replace column entries by numbers."""
    entries = list(df[column].unique())
    mapping = {c: n for n, c in enumerate(entries)}
    df[column] = df[column].map(mapping).astype(int)
    return df


def average_score(
        df: pd.DataFrame,
        column: str,
        entry: str,
        condition: str,
) -> float:
    r"""Calculates the average value for a given column entry.

    This could be used to calculate the average value
    for a call or speaker.

    Args:
        df: data frame
        column: column holding the selection entries,
            e.g. ``'call'``
        entry: selection entry,
            e.g. particular call ID
        condition: test condition, e.g. ``'arousal'``

    Returns:
        average value

    """
    y = df.loc[df[column] == entry, condition]
    return np.mean(y.values)


def balance_data(df, *, column, group_column, samples):
    r"""Balance ground truth data for protected fairness group.

    Args:
        df: data frame with prediction values at ``column``
            and group values at ``group_column``
        column: name of column in data frame
            holding the model predictions
        group_column: name of column in data frame
            holding the group values
        samples: number of samples to use for each group

    Returns:
        balanced dataframe

    Raises:
        RuntimeError: if any of the groups contains less
            than the requested ``samples``

    """
    groups = df[group_column].unique()

    # Find the group with lowest number of samples
    smallest_group = groups[0]
    for group in groups[1:]:
        if (
                len(df[df[group_column] == group])
                < len(df[df[group_column] == smallest_group])
        ):
            smallest_group = group

    if len(df[df[group_column] == smallest_group]) < samples:
        raise RuntimeError(
            f"You asked for {samples} samples per group, "
            "but the database has only "
            f"{len(df[df[group_column] == smallest_group])} samples "
            f"in group '{smallest_group}'."
        )

    other_groups = [
        group for group in groups
        if group != smallest_group
    ]

    # Select samples randomly from smallest group
    df_0 = df[df[group_column] == smallest_group].sample(
        samples,
        random_state=SEED,
    )
    data_0 = list(df_0[column])

    # Select matching samples for other groups
    dfs = [df_0]
    for group in other_groups:

        df_group = df[df[group_column] == group]
        data_group = sorted(list(df[df[group_column] == group][column]))

        values = []
        files = []
        starts = []
        ends = []
        for data in data_0:

            if isinstance(data, str):  # classes
                value = data
            else:  # regression
                value = take_closest(data_group, data)

            df_selected = df_group[df_group[column] == value]

            if len(df_selected) == 0:
                df_selected = df_group

            n = random.choice(range(len(df_selected)))
            values.append(df_selected.iloc[n, :].values.tolist())
            if audformat.is_segmented_index(df):
                file = df_selected.index.get_level_values('file')[0]
                start = df_selected.index.get_level_values('start')[0]
                end = df_selected.index.get_level_values('end')[0]
                starts.append(start)
                ends.append(end)
                # Remove selected values from dataframe
                df_group = df_group.drop((file, start, end))
            else:
                file = df_selected.index[0]
                # Remove selected values from dataframe
                df_group = df_group.drop((file))
            files.append(file)

            # Remove selected values from list
            data_group.remove(value)
        
        if audformat.is_segmented_index(df):
            index = audformat.segmented_index(files, starts, ends)
        else:
            index = audformat.filewise_index(files)
        dfs.append(
            pd.DataFrame(
                values,
                index=index,
                columns=df_0.columns,
            )
        )

    return pd.concat(dfs)


def bin_values(
    values: pd.Series,
    bins: int,
    value_range: typing.Tuple[float, float] = (0,1),
)-> pd.Series:
    r"""Return binned version of regression values
    
    Args:
        values: regression values to be binned
        bins: number of bins the truth and prediction values should be binned to
        value_range: the target value range. truth and prediction may still contain
            higher or lower values, and they will still be mapped to the closest bin
    Returns:
        values binned into one of the `bins` bins
    """
    # Assume that all ranges are target to be in `range`
    # but cover up to (-ninf, inf) in case the model output exceeds the range
    bin_edges = np.linspace(value_range[0],value_range[1], num=bins+1, endpoint=True)
    bin_edges = np.round(bin_edges, 2)
    bin_tuples = [(bin_edges[i], bin_edges[i+1]) for i in range(bins)]
    bin_tuples[0] = (-np.Inf, bin_tuples[0][1])
    bin_tuples[-1] = (bin_tuples[-1][0], np.Inf)
    bins = pd.IntervalIndex.from_tuples(bin_tuples)
    binned_values = pd.cut(values, bins, precision=6)
    return binned_values


def class_proportions(
    df: pd.DataFrame,
    column: str,
    entry: str,
    condition: str,
) -> pd.Series:
    r"""Calculates the class-wise proportions for a given
    column entry.

    This could be used to calculate the class label distributions
    for a call or speaker.

    Args:
        df: data frame
        column: column holding the selection entries,
            e.g. ``'call'``
        entry: selection entry,
            e.g. particular call ID
        condition: test condition, e.g. ``'emotion'``

    Returns:
        series with proportions for each class label

    """
    class_labels = list(df[condition].unique())
    n_total = len(df[df[column]==entry])
    result = {
        class_label: len(df[(df[column]==entry) & (df[condition]==class_label)])
        for class_label in class_labels
    }
    result = {
        k: v/n_total for k,v in result.items()
    }
    return pd.Series(result)

def comparison_name(
    model_baseline: str,
    model_candidate: str
):
    r"""Create name of a model comparison"""
    return f'{model_name(model_baseline)}_{model_name(model_candidate)}'


def figure_filename(
        test: str,
        condition: str,
        *,
        group: str = None,
) -> str:
    r"""Return filename for generated figures.

    All input strings should not contain empty spaces.

    Args:
        test: name of test
        condition: name of condition
        group: additional string to add at the end of the filename

    Returns:
        figure filename

    """
    if group is None:
        return f'{test}_{condition}.png'
    else:
        return f'{test}_{condition}_{group}.png'


def limit_string_length(text, limit):
    r"""Limit text length and replace too long text with ..."""
    if len(text) >= limit:
        text = text[:(limit - 3)] + '...'
    return text


def model_information(model):

    # Extract model info and params with audmodel
    if model.uid in ['random-gaussian', 'random-categorical']:
        header = model.header
        meta = {}
    else:
        header = audmodel.header(model.uid)
        meta = audmodel.meta(model.uid)
    info = {**header, **meta}
    info = {k.capitalize(): v for k, v in info.items()}
    params = info.pop('Parameters')

    # parameters may contain mixed types,
    # including sequences and non-sequences
    # convert to str first
    params = {k: str(v) for k, v in params.items()}

    df_info = pd.DataFrame.from_dict(
        info,
        orient='index',
        columns=['Value'],
    )
    df_info.index.name = 'Entry'

    df_params = pd.DataFrame.from_dict(
        params,
        orient='index',
        columns=['Value'],
    )
    df_params.index.name = 'Entry'

    df_tuning = pd.DataFrame.from_dict(
        model.tuning_params,
        orient='index',
        columns=['Value'],
    )
    df_tuning.index.name = 'Entry'

    return df_info, df_params, df_tuning


def model_name(model):
    tuning_params = getattr(model, 'tuning_params', {})

    if len(tuning_params) > 0:
        tuning_id = audeer.uid(from_string=str(tuning_params))[:3]
        return f'{model.uid}-{tuning_id}'
    else:
        return model.uid


def print_header(test_name):
    r"""Prints test header to command line.

    Args:
        test_name: name of test,
            e.g. 'fairness_sex'

    """
    test_name = test_name.replace("_", " ")
    test_name = test_name.replace("-", " ")
    test_name = test_name.capitalize()
    print()
    print(test_name)
    print('-' * len(test_name))


def print_impact_result(
        condition: str,
        test_name: str,
        score: str,
        impact_type: str
):
    r"""Print current impact test and result to command line.

    Args:
        condition: test condition,
            usually 'database-version-table'
        test_name: name of test,
            e.g. 'impact_classification'
        score: result of test
        impact_type: Type of impact the test result has
    """
    if impact_type is None:
        status = '[ skip ]'
        Impacts.skipped += 1
    elif impact_type == 'positive':
        status = f'[ {Color.green}positive{Color.reset} ]'
        Impacts.positive += 1
    elif impact_type == 'negative':
        status = f'[ {Color.red}negative{Color.reset} ]'
        Impacts.negative += 1
    else:
        status = '[ neutral  ]'
        Impacts.neutral += 1
    print(f'{condition: <55} {test_name: <30} {score: <8} {status}')


def print_impact_summary(
        impacts: Impacts,
):
    r"""Print impact test summary including positive, negative, neutral.

    Args:
        impacts: impacts object
            containing the attributes
            ``positive``,
            ``negative``,
            ``neutral``,
            ``skipped``

    """
    total_tests = impacts.positive + impacts.neutral + impacts.negative \
        + impacts.skipped
    if total_tests > 0:
        pos_percentage = f'{100 * impacts.positive / total_tests: 3.1f}%'
        neg_percentage = f'{100 * impacts.negative / total_tests: 3.1f}%'
        neu_percentage = f'{100 * impacts.neutral / total_tests: 3.1f}%'
    else:
        pos_percentage = '  0.0%'
        neg_percentage = '  0.0%'
        neu_percentage = '  0.0%'

    print()
    print('Summary')
    print('-------')

    pos_message = f'{impacts.positive} positive impacts'
    neg_message = f'{impacts.negative} negative impacts'
    neu_message = f'{impacts.neutral} neutral impacts'
    skipped_message = f'{impacts.skipped} impact tests skipped'
    print(f'{pos_message: <85} {pos_percentage: <6}')
    print(f'{neg_message: <85} {neg_percentage: <6}')
    print(f'{neu_message: <85} {neu_percentage: <6}')

    if impacts.skipped > 0:
        print(f'{skipped_message: <85}')


def print_impact_title(baseline, candidate, condition):
    r"""Print an impact analysis title including model IDs and condition.

    Args:
        baseline: model ID of the baseline
        candidate: model ID of the candidate
        condition: ``'arousal'``, ``'dominance'``,
            ``'valence'``, or ``'emotion'``

    """
    message = f'{baseline} vs. {candidate} ({condition})'
    print()
    print(message)
    print('=' * len(message))


def print_result(
        condition: str,
        test: str,
        score: str,
        passed: bool,
):
    r"""Print current test and result to command line.

    Args:
        conditon: test condition,
            ususally 'database-version-table'
        test: name of test,
            e.g. 'fairness_sex'
        score: result of test
        passed: ``True`` if test suceeded

    """
    # Limit text to fixed spaces in console output
    condition = f'{limit_string_length(condition, 55): <55}'
    test = f'{limit_string_length(test, 30): <30}'
    # Create status
    if passed is None:
        score = ' ' * 8
        status = '[ skip ]'
        Tests.skipped += 1
    else:
        score = f'{score: <8}'
        if passed:
            status = f'[ {Color.green}pass{Color.reset} ]'
            Tests.passed += 1
        else:
            status = f'[ {Color.red}fail{Color.reset} ]'
            Tests.failed += 1

    print(f'{condition} {test} {score} {status}')


def print_summary(
        tests: Tests,
):
    r"""Print test summary including passed and failed.

    Args:
        tests: test object
            containing the attributes
            ``passed``,
            ``failed``,
            `skipped``

    """
    total_tests = tests.passed + tests.failed
    if total_tests > 0:
        percentage = f'{100 * tests.passed / total_tests: 3.1f}%'
    else:
        percentage = '  0.0%'

    message = f'{tests.passed} tests passed / {tests.failed} tests failed'
    if tests.skipped > 0:
        message += f' / {tests.skipped} tests skipped'
    print()
    print('Summary')
    print('-------')
    print(f'{message: <85} {percentage: <6}')


def print_title(
        model_id: str,
        condition: str,
):
    r"""Print a test title inclduing model ID and condition.

    Args:
        model_id: model ID
        condition: ``'arousal'``, ``'dominance'``,
            ``'valence'``, or ``'emotion'``

    """
    message = f'{model_id} ({condition})'
    print()
    print(message)
    print('=' * len(message))


def quantile(
        y: pd.Series,
) -> int:
    r"""Return the length of a quantile.

    Args:
        y: input series

    Returns:
        1/4 length of the input

    """
    return int(np.round(len(y) / 4))


def run_model(
        model: audinterface.Process,
        index: pd.Index,
        database: typing.Dict[str, str] = None,
        *,
        call_based_rater: bool = False,
) -> pd.Series:
    r"""Run the model on provided index and return predictions.

    The predictions are also cached
    under a unique path
    based on the model ID and the truth index.

    Args:
        model: model with interface and ``uid`` attribute
        index: database index
        database: dictinonary with name, version, split(s) of database

    Returns:
        model predictions for index

    """
    hash = audformat.utils.hash(index)
    cache = audeer.safe_path(
        os.path.join(CURRENT_DIR, '..', 'cache', model_name(model), hash)
    )
    if model.mode is None:
        raise ValueError('Model mode has to be set for caching.')

    if model.uid == 'random-rater':
        if call_based_rater:
            random_type = 'call'
        else:
            random_type = 'segment'
        filepath = f'{random_type}-{model.mode}-prediction.pkl'
    else:
        filepath = f'{model.mode}-prediction.pkl'
    prediction_cache = os.path.join(cache, filepath)

    if os.path.exists(prediction_cache):
        prediction = pd.read_pickle(prediction_cache)
    else:
        audeer.mkdir(cache)
        if model.uid == 'random-rater':
            prediction = model.process_index(index, database, call_based_rater)
        else:
            prediction = model.process_index(index)
        # Only convert to float32 if output is actually numeric (and not a class labels)
        if pd.api.types.is_numeric_dtype(prediction):
            prediction = prediction.astype('float32')
        prediction.to_pickle(prediction_cache)

    # Limit to two decimal places to ensure reproducibility, see
    # https://github.com/audeering/audonnx/issues/61
    prediction = prediction.round(2)

    return prediction


def split_figure_title(title, *, max_length=50):
    if len(title) <= max_length:
        return title

    if '-' in title:
        dash_separated_parts = title.split('-')
        # Try to split once at second dash
        # If only one dash is available, split there
        newline_idx = min(2, len(dash_separated_parts) - 1)
        dash_separated_parts[newline_idx] = '\n' + dash_separated_parts[newline_idx]
        title = '-'.join(dash_separated_parts)
    return title


def take_closest(sequence, value):
    """Returns closest value from sorted sequence.

    If two numbers are equally close,
    return the smallest number.

    """
    pos = bisect_left(sequence, value)
    if pos == 0:
        return sequence[0]
    if pos == len(sequence):
        return sequence[-1]
    before = sequence[pos - 1]
    after = sequence[pos]
    if after - value < value - before:
        return after
    else:
        return before
