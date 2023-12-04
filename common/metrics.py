import typing

import numpy as np
import pandas as pd
import scipy.stats
import scipy.spatial

import audformat
import audmetric

from . import (
    bin_values,
    quantile,
)


def bin_proportion_shift_difference(
        predictions: pd.Series,
        group_values: pd.Series,
        group: str,
        context_values: pd.Series,
        context: str,
        bins: int,
        value_range: typing.Tuple[float, float] = (0,1),
        min_samples: int = 0,
)-> typing.Tuple[typing.Dict, typing.Dict, typing.Dict]:
    r"""Difference in bin proportion shift for a certain context and group.

    This is used to compute if predictions for a sepecific group
    change in a similar way compared to the average shift in
    predictions under a certain context (or background condition).
    The average shift in model predictions is computed by averaging
    the bin proportion shift across all groups.
    The first two returned values hold the group's bin proportion shift and the
    average bin proportion shift as reference, and the third value contains
    the actual metric (the difference between the two first values).

    Args:
        predictions: regression value predictions to be binned
        group_values: group values for each of the predictions
        group: group entry to filter by to calculate the shift in class proportions
        context_values: context labels for each of the predictions
        context: context entry to investigate
        bins: number of bins the prediction values should be binned to
        value_range: the target value range. Prediction values may still contain
            higher or lower values, and they will still be mapped to the closest bin
        min_samples: minimum number of required samples per bin.
            If bin has lower number of samples
            ``None`` is the result for that bin

    Returns:
        three dictionaries containing a float value between -1 and 1 for each bin,
        corresponding to the selected group's bin proportion shift
        when applying the given context,
        the average bin proportion shift in mean,
        and the difference between the first two values.
    """
    predictions = bin_values(predictions, bins, value_range=value_range)
    group_shift, avg_shift, shift_diff = class_proportion_shift_difference(
        predictions, group_values, group, context_values, context,
        class_labels=predictions.cat.categories.to_list()
    )

    shift_diff = skip_low_samples(
        predictions[context_values==context],
        results=shift_diff, samples=min_samples
    )
    return group_shift, avg_shift, shift_diff


def class_proportion_shift_difference(
        predictions: pd.Series,
        group_values: pd.Series,
        group: str,
        context_values: pd.Series,
        context: str,
        class_labels: typing.Optional[typing.Sequence[str]]=None,
)-> typing.Tuple[typing.Dict, typing.Dict, typing.Dict]:
    r"""Difference in class proportion shift for a certain context and group.

    This is used to compute if predictions for a sepecific group
    change in a similar way compared to the average shift in
    predictions under a certain context (or background condition).
    The average shift in model predictions is computed by averaging
    the class proportion shift across all groups.
    The first two returned values hold the group's class proportion shift and the
    average class proportion shift as reference, and the third value contains
    the actual metric (the difference between the two first values).

    Args:
        predictions: class label predictions
        group_values: group values for each of the predictions
        group: group entry to filter by to calculate the shift in class proportions
        context_values: context labels for each of the predictions
        context: context entry to investigate
        class_labels: the class labels to include

    Returns:
        three dictionaries containing a float value between -1 and 1 for each class,
        corresponding to the selected group's class proportion shift
        when applying the given context,
        the average class proportion shift in mean,
        and the difference between the first two values.
    """
    df = pd.DataFrame(
        data={'value': predictions, 'group': group_values, 'context': context_values}
    )

    # Compute the each individual group's shift in class proportion
    group_shifts = df.groupby('group').apply(
        lambda x: pd.Series(
            relative_difference_per_class(
                x['value'],
                x[x['context']==context]['value'],
                class_labels=class_labels
            )
        )
    )
    group_shift = group_shifts.loc[group, :]
    average_shift = group_shifts.mean()

    shift_diff = group_shift - average_shift
    return group_shift.to_dict(), average_shift.to_dict(), shift_diff.to_dict()

def difference_in_mean(
        truth: pd.Series,
        prediction: pd.Series,
) -> float:
    r"""Return the difference in the mean over all classes.

    Args:
        truth: truth values
        prediciton: prediciton values

    Returns:
        average prediciton - average truth

    """
    return np.mean(prediction) - np.mean(truth)


def distribution_3_classes(
        prediction: pd.Series,
        bins: typing.Sequence,
) -> typing.Dict:
    r"""Percentage of tone for 3-classes.

    This does not need any ground truth,
    but just calculates the distribution in percentage
    for the three classes negative, neutral, positive.

    Args:
        prediciton: prediciton values in the range 0..1
        bins: bins for converting regression values to three classes

    Returns:
        - percentage of negative samples
        - percentage of neutral samples
        - percentage of positive samples

    """
    prediction = np.digitize(prediction, bins)
    samples = len(prediction)
    negative_samples = len(prediction[prediction == 0])
    neutral_samples = len(prediction[prediction == 1])
    positive_samples = len(prediction[prediction == 2])

    return {
        'negative': negative_samples / samples,
        'neutral': neutral_samples / samples,
        'positive': positive_samples / samples,
    }


def jensen_shannon_distance(
        truth: pd.Series,
        prediction: pd.Series,
        *,
        bins=None,
) -> typing.Tuple[float, float]:
    r"""Jensen-Shannon distance of the two distributions.

    See
    https://en.wikipedia.org/wiki/Jensen–Shannon_divergence
    and
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.jensenshannon.html

    Args:
        truth: truth values
        prediciton: prediction values
        bins: if not ``None``
            it uses the provided bins
            for creating the distribution

    Returns:
        distance between 0 and 1

    """
    if bins is not None:
        # Create distribution,
        # and make sure also to include non-present classes
        truth_distribution = truth.value_counts(
            bins=bins,
            sort=False,
        ).values
        prediction_distribution = prediction.value_counts(
            bins=bins,
            sort=False,
        ).values
    else:
        # Concat ensures that classes without occurence are counted as 0
        truth.name = None
        prediction.name = None
        distributions = pd.concat(
            [
                truth.value_counts().sort_index(kind='stable'),
                prediction.value_counts().sort_index(kind='stable'),
            ],
            axis=1,
        ).fillna(0).astype(int)
        truth_distribution = distributions[0].values
        prediction_distribution = distributions[1].values
    return scipy.spatial.distance.jensenshannon(
        prediction_distribution,
        truth_distribution,
        base=2,
    )


def mean_directional_error(
        truth: pd.Series,
        prediction: pd.Series,
) -> float:
    r"""Mean directional error between truth and prediction.

    This indicates if you have a bias towards
    positive or negative tone
    over all predictions.

    Args:
        truth: truth values
        prediction: prediction values

    Returns:
        mean directional error,
            greater 0 means your prediction is higher

    """
    return np.mean(prediction - truth)


def mean_shift_difference(
        predictions: pd.Series,
        group_values: pd.Series,
        group: str,
        context_values: pd.Series,
        context: str,
)-> typing.Tuple[float, float, float]:
    r"""Difference in mean shift for a certain context and group.

    This is used to compute if predictions for a sepecific group
    change in a similar way compared to the average shift in
    predictions under a certain context (or background condition).
    The average shift in model predictions is computed by averaging
    the mean value shifts across all groups.
    The first two returned values hold the group mean shift and the
    average mean shift as reference, and the third value contains
    the actual metric (the difference between the two first values).

    Args:
        predictions: regression value predictions to be binned
        group_values: group values for each of the predictions
        group: group entry to filter by to calculate the shift in class proportions
        context_values: context labels for each of the predictions
        context: context entry to investigate

    Returns:
        three float values between -1 and 1 corresponding to
        the selected group's shift in mean when applying the given context,
        the average shift in mean, and the difference between the first two values.
    """
    df = pd.DataFrame(
        data={'value': predictions, 'group': group_values, 'context': context_values}
    )

    # Compute the each individual group's shift in class proportion
    group_shifts = df.groupby('group').apply(
        lambda x:
            difference_in_mean(
                x['value'],
                x[x['context']==context]['value'],
            )
    )
    group_shift = group_shifts[group]
    average_shift = group_shifts.mean()

    shift_diff = group_shift - average_shift
    return group_shift, average_shift, shift_diff


def percentage_of_added_errors(
    truth, prediction_baseline, prediction_new, threshold=None
):
    r"""Percentage of samples that are correctly predicted by the baseline
    prediction but not by the new prediction.

    If a threshold value is passed, instead of checking for true equality,
    labels are considered to be equal if their absolute difference lies
    below the given threshold. In addition, only samples whose difference
    in prediction lies above the given threshold are considered for the
    resulting percentage.

    Args:
        truth: truth values
        prediction_baseline: prediction values of the baseline
        prediction_new: new prediction values to compare the baseline to
        threshold: threshold that determines if two labels should be
            considered equal or not

    Returns:
        percentage of added errors

    """
    if threshold is None:
        n_added_errors = sum(
            (prediction_baseline == truth) & (prediction_new != truth)
        )
    else:
        correct_baseline = (prediction_baseline - truth).abs() < threshold
        incorrect_new = (prediction_new - truth).abs() >= threshold
        changed = (prediction_new - prediction_baseline).abs() >= threshold
        n_added_errors = sum(correct_baseline & incorrect_new & changed)
    return n_added_errors / len(prediction_new)


def percentage_of_changed_errors(
    truth, prediction_baseline, prediction_new, threshold=None
):
    r"""Percentage of samples that are incorrectly predicted by both predictions,
    and also have different predictions for the baseline and new.

    If a threshold value is passed, instead of checking for true equality,
    labels are considered to be equal if their absolute difference lies
    below the given threshold.

    Args:
        truth: truth values
        prediction_baseline: prediction values of the baseline
        prediction_new: new prediction values to compare the baseline to
        threshold: threshold that determines if two labels should be
            considered equal or not

    Returns:
        percentage of changed errors

    """
    if threshold is None:
        n_changed_errors = sum(
            (prediction_baseline != truth)
            & (prediction_new != truth)
            & (prediction_new != prediction_baseline)
        )
    else:
        incorrect_baseline = (prediction_baseline - truth).abs() >= threshold
        incorrect_new = (prediction_new - truth).abs() >= threshold
        changed_baseline_new = (
                prediction_baseline - prediction_new
            ).abs() >= threshold
        n_changed_errors = sum(
            incorrect_baseline & incorrect_new & changed_baseline_new
        )
    return n_changed_errors / len(prediction_new)


def percentage_of_error_corrections(
    truth, prediction_baseline, prediction_new, threshold=None
):
    r"""Percentage of samples that are incorrectly predicted by the baseline
    but correctly predicted in the new predictions.

    If a threshold value is passed, instead of checking for true equality,
    labels are considered to be equal if their absolute difference lies
    below the given threshold. In addition, only samples whose difference
    in prediction lies above the given threshold are considered for the
    resulting percentage.

    Args:
        truth: truth values
        prediction_baseline: prediction values of the baseline
        prediction_new: new prediction values to compare the baseline to
        threshold: threshold that determines if two labels should be
            considered equal or not

    Returns:
        percentage of error corrections

    """
    if threshold is None:
        n_corrected_errors = sum(
            (prediction_baseline != truth)
            & (prediction_new == truth)
        )
    else:
        incorrect_baseline = (prediction_baseline - truth).abs() >= threshold
        correct_new = (prediction_new - truth).abs() < threshold
        changed = (prediction_new - prediction_baseline).abs() >= threshold

        n_corrected_errors = sum(incorrect_baseline & correct_new & changed)
    return n_corrected_errors / len(prediction_new)


def proportion_in_range(
    prediction: pd.Series,
    allowed_range: typing.Tuple[
        typing.Optional[float], typing.Optional[float]
        ],
):
    r"""Proportion of samples that are in the given range.

    Args:
        prediction: prediction values
        allowed_range: tuple with the minimum and maximum allowed value
            if a value is set to None, there is no limit to the minimum/maximum
    
    Returns:
        proportion between 0 and 1
    """
    if allowed_range[0] is not None:
        above_minimum = prediction >= allowed_range[0]
        if allowed_range[1] is not None:
            samples_in_range = above_minimum & (prediction <= allowed_range[1])
        else:
            samples_in_range = above_minimum
    elif allowed_range[1] is not None:
        samples_in_range = prediction <= allowed_range[1]
    else:
        raise ValueError('Given range is empty')
    n_in_range = samples_in_range.sum()
    return n_in_range / len(prediction)


def percentage_of_identity(
        y1: pd.Series,
        y2: pd.Series,
) -> float:
    r"""Percentage of identy between two sequences of labels.

    Args:
        y1: first sequence of labels
        y2: second sequence of labels

    Returns:
        percentage of identity in the range 0 to 1

    """
    y2.index = y1.index
    if y1.dtype == np.object:
        identical = (
            y1 == y2
        ).sum()
    else:
        identical = (
            (y1 - y2).abs() < 0.05
        ).sum()

    return identical / len(y1)


def precision_per_bin(
    truth: pd.Series,
    prediction: pd.Series,
    bins: int,
    value_range: typing.Tuple[float, float] = (0,1),
    min_samples: int = 0,
) -> typing.Dict[typing.Any, float]:
    r"""Return the precision of the binned prediction values per bin.

    Args:
        truth: truth values
        prediction: prediction values
        bins: number of bins the truth and prediction values should be binned to
        value_range: the target value range. truth and prediction may still contain
            higher or lower values, and they will still be mapped to the closest bin
        min_samples: minimum number of required samples per bin.
            If bin has lower number of samples
            ``None`` is returned for as result for that bin

    Returns:
        dictionary containing the precision between 0 and 1
        for each bin.

    """
    truth = bin_values(truth, bins, value_range=value_range)
    prediction = bin_values(prediction, bins, value_range=value_range)

    labels = truth.cat.categories.to_list()
    result = audmetric.precision_per_class(truth, prediction, labels=labels)
    result = skip_low_samples(truth, result, min_samples)

    return result


def reactivity(
        truth: pd.Series,
        prediction: pd.Series,
        max_value: float = 0.3,
) -> float:
    r"""Estimate how well the prediction follows value changes.

    It considers only the 25% of calls from ``truth``
    with the highest average value change.

    Args:
        truth: true average value changes per recording
        prediction: predicted average value changes per recording
        max_value: maximum average value change

    Returns:
        reactivity

    """
    truth = truth.nlargest(int(len(truth) * 0.25))
    prediction = prediction.loc[truth.index]
    # Scale assuming max_value for maximum average value change
    truth = truth * (1 / max_value)
    prediction = prediction * (1 / max_value)
    reactivity = min(((prediction + 1) / (truth + 1)).mean(), 1.0)
    # Scale value to be in range 0..1
    return (reactivity - 0.5) * 2


def recall_per_bin(
    truth: pd.Series,
    prediction: pd.Series,
    bins: int,
    value_range: typing.Tuple[float, float] = (0,1),
    min_samples: int = 0,
) -> typing.Dict[typing.Any, float]:
    r"""Return the recall of the binned prediction values per bin.

    Args:
        truth: truth values
        prediction: prediction values
        bins: number of bins the truth and prediction values should be binned to
        value_range: the target value range. truth and prediction may still contain
            higher or lower values, and they will still be mapped to the closest bin
        min_samples: minimum number of required samples per bin.
            If bin has lower number of samples
            ``None`` is returned for as result for that bin

    Returns:
        dictionary containing the recall between 0 and 1
        for each bin.

    """
    truth = bin_values(truth, bins, value_range=value_range)
    prediction = bin_values(prediction, bins, value_range=value_range)

    labels = truth.cat.categories.to_list()
    result = audmetric.recall_per_class(truth, prediction, labels=labels)
    result = skip_low_samples(truth, result, min_samples)

    return result


def relative_difference_per_bin(
    truth: pd.Series,
    prediction: pd.Series,
    bins: int,
    value_range: typing.Tuple[float, float] = (0,1),
    min_samples: int = 0,
) -> typing.Dict[typing.Any, float]:
    r"""Return the percentage of change in samples for each class.

    Args:
        truth: truth values
        prediction: prediction values
        bins: number of bins the truth and prediction values should be binned to
        value_range: the target value range. truth and prediction may still contain
            higher or lower values, and they will still be mapped to the closest bin
        min_samples: minimum number of required samples per bin.
            If bin has lower number of samples
            ``None`` is returned for as result for that bin

    Returns:
        dictionary containing a float value between -1 and 1
        for each bin.
        A value of 0.2 would indicate
        that the prediction contains 20% more samples in this bin.

    """
    truth = bin_values(truth, bins, value_range=value_range)
    prediction = bin_values(prediction, bins, value_range=value_range)
    
    labels = truth.cat.categories.to_list()
    result = relative_difference_per_class(
        truth, prediction, class_labels=labels,
    )
    result = skip_low_samples(truth, result, min_samples)

    return result


def relative_difference_per_class(
        truth: pd.Series,
        prediction: pd.Series,
        class_labels: typing.Optional[typing.Sequence]=None,
) -> typing.Dict[typing.Any, float]:
    r"""Return the percentage of change in samples for each class.

    Args:
        truth: truth values
        prediciton: prediction values
        class_labels: the class labels to include

    Returns:
        dictionary containing a float value between -1 and 1
        for each class.
        A value of 0.2 would indicate
        that the prediction contains 20% more samples in this class.

    """

    # only use class labels that actually occur when class_labels is None
    # to fix issue with categorical datatype in pd.Series
    if class_labels is None:
        class_labels = sorted(set(truth.unique()).union(set(prediction.unique())))
    # filter results by class labels
    truth = truth[truth.isin(class_labels)]
    prediction = prediction[prediction.isin(class_labels)]
    n_truth_samples = len(truth)
    n_pred_samples = len(prediction)
    truth = truth.value_counts().sort_index(kind='stable').to_dict()
    prediction = prediction.value_counts().sort_index(kind='stable').to_dict()
    changes = {}
    for class_label in class_labels:
        if class_label not in prediction:
            prediction[class_label] = 0
        if class_label not in truth:
            truth[class_label] = 0
        changes[class_label] = (
            (prediction[class_label] / n_pred_samples) -
            (truth[class_label] / n_truth_samples)
        )
    return changes


def skip_low_samples(
        truth: pd.Series,
        results: typing.Dict,
        samples: int,
) -> typing.Dict:
    r"""Set results for low number of samples to None.

    This is helpful
    to skip tests
    for bins with too few samples.

    """
    counts = truth.value_counts()
    for label in results:
        if counts[label] < samples:
            results[label] = None
    return results


def spearmanr(
        truth: pd.Series,
        prediction: pd.Series,
) -> float:
    r"""Calculate Spearmans Rho.

    Args:
        truth: truth values
        prediction: prediction values

    Returns:
        Spearman's Rho

    """
    rho, _ = scipy.stats.spearmanr(truth, prediction)
    return rho


def stability(
        truth: pd.Series,
        prediction: pd.Series,
        max_value: float = 0.3,
) -> float:
    r"""Estimate how well the prediction follows value changes.

    It considers only the 25% of recordings from ``truth``
    with the lowest average value changes.

    Args:
        truth: true average value changes per recording
        prediction: predicted average value changes per recording
        max_value: maximum average value change

    Returns:
        stability as a value between 0 and 1

    """
    truth = truth.nsmallest(int(len(truth) * 0.25))
    prediction = prediction.loc[truth.index]
    # Scale assuming max_value for maximum average value change
    truth = truth * (1 / max_value)
    prediction = prediction * (1 / max_value)
    stability = min(((truth + 1) / (prediction + 1)).mean(), 1.0)
    # Scale value to be in range 0..1
    return (stability - 0.5) * 2


def value_changes_per_file(
        y: pd.Series,
) -> pd.Series:
    r"""Count the number of value changes per file.

    The input series is expected
    to contain a segmented index,
    and the file names correspond to single recordings.

    It will then return a filewise table
    listing the number of value changes per file.

    The values are expected
    to be regression values in the range 0..1, or categorical
    values.
    For regression we count the mean change of value from one segment
    to the next segment, and for classification we count the proportion of
    segments that change from one to the next.
    """
    files = list(set(y.index.get_level_values('file')))
    changes = []
    selected_files = []
    for file in files:
        # count the number of value changes
        # see https://stackoverflow.com/a/56509920
        y_file = y.loc[file]
        if len(y_file) < 4:
            continue
        if pd.api.types.is_float_dtype(y_file):
            # Mean absolute value difference between two consecutive segments
            value_diffs = np.diff(y_file)
            changes_per_segment = np.mean(np.abs(value_diffs))
        else:
            # number of segments with a different label between two consecutive
            # segments
            value_changed = (y_file.shift(1)!= y_file).iloc[1:]
            changes_per_segment = value_changed.mean()

        changes.append(changes_per_segment)
        selected_files.append(file)
    index = audformat.filewise_index(selected_files)
    return pd.Series(changes, index=index).sort_index(kind='stable')


def top_bottom_confusion(
        truth: pd.Series,
        prediction: pd.Series,
) -> float:
    r"""Percentage of confusions between top and bottom quantile.

    Might be used for judging call/speaker rankings.

    Args:
        truth: truth values
        prediction: prediction values

    Returns:
        precision

    """
    # Sort results after call average tone
    truth = truth.sort_values(kind='stable')
    prediction = prediction.sort_values(kind='stable')
    # Get length for 25%
    length = quantile(truth)
    truth_top = set(truth.iloc[-length:].index)
    truth_bottom = set(truth.iloc[:length].index)
    prediction_top = set(prediction.iloc[-length:].index)
    prediction_bottom = set(prediction.iloc[:length].index)
    common = (
        len(truth_top & prediction_bottom)
        + len(truth_bottom & prediction_top)
    )
    number = len(truth_top) + len(truth_bottom)
    return common / number


def top_bottom_precision(
        truth: pd.Series,
        prediction: pd.Series,
        region: str,
) -> float:
    r"""Precision of including the right calls in the given quantile.

    Args:
        truth: truth values
        prediciton: prediciton values
        region: ``'top'`` or ``'bottom'``

    Returns:
        precision

    """
    # Sort results after call average tone
    truth = truth.sort_values(kind='stable')
    prediction = prediction.sort_values(kind='stable')
    # Get length for 25%
    length = quantile(truth)
    if region == 'top':
        truth = truth.iloc[-length:]
        prediction = prediction.iloc[-length:]
    elif region == 'bottom':
        truth = truth.iloc[:length]
        prediction = prediction.iloc[:length]
    truth = set(truth.index)
    prediction = set(prediction.index)
    common = len(truth & prediction)
    return common / len(truth)
