import os
import random
import typing

import numpy as np
import pandas as pd
from scipy.stats import truncnorm

import audmetric


def bin_values(
    values: pd.Series,
    bins: int,
    value_range: typing.Tuple[float, float] = (0, 1),
) -> pd.Series:
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
    bin_edges = np.linspace(value_range[0], value_range[1], num=bins + 1, endpoint=True)
    bin_edges = np.round(bin_edges, 2)
    bin_tuples = [(bin_edges[i], bin_edges[i + 1]) for i in range(bins)]
    bin_tuples[0] = (-np.Inf, bin_tuples[0][1])
    bin_tuples[-1] = (bin_tuples[-1][0], np.Inf)
    bins = pd.IntervalIndex.from_tuples(bin_tuples)
    binned_values = pd.cut(values, bins, precision=6)
    return binned_values


def bin_proportion_shift_difference(
    predictions: pd.Series,
    group_values: pd.Series,
    group: str,
    context_values: pd.Series,
    context: str,
    bins: int,
    value_range: typing.Tuple[float, float] = (0, 1),
) -> typing.Tuple[typing.Dict, typing.Dict, typing.Dict]:
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

    Returns:
        three dictionaries containing a float value between -1 and 1 for each bin,
        corresponding to the selected group's bin proportion shift
        when applying the given context,
        the average bin proportion shift in mean,
        and the difference between the first two values.
    """
    predictions = bin_values(predictions, bins, value_range=value_range)
    result = class_proportion_shift_difference(
        predictions,
        group_values,
        group,
        context_values,
        context,
        class_labels=predictions.cat.categories.to_list(),
    )
    return result


def class_proportion_shift_difference(
    predictions: pd.Series,
    group_values: pd.Series,
    group: str,
    context_values: pd.Series,
    context: str,
    class_labels: typing.Optional[typing.Sequence[str]] = None,
) -> typing.Tuple[typing.Dict, typing.Dict, typing.Dict]:
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
        data={"value": predictions, "group": group_values, "context": context_values}
    )

    # Compute the each individual group's shift in class proportion
    group_shifts = df.groupby("group").apply(
        lambda x: pd.Series(
            relative_difference_per_class(
                x["value"],
                x[x["context"] == context]["value"],
                class_labels=class_labels,
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


def mean_shift_difference(
    predictions: pd.Series,
    group_values: pd.Series,
    group: str,
    context_values: pd.Series,
    context: str,
) -> typing.Tuple[float, float, float]:
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
        data={"value": predictions, "group": group_values, "context": context_values}
    )

    # Compute the each individual group's shift in class proportion
    group_shifts = df.groupby("group").apply(
        lambda x: difference_in_mean(
            x["value"],
            x[x["context"] == context]["value"],
        )
    )
    group_shift = group_shifts[group]
    average_shift = group_shifts.mean()

    shift_diff = group_shift - average_shift
    return group_shift, average_shift, shift_diff


def precision_per_bin(
    truth: pd.Series,
    prediction: pd.Series,
    bins: int,
    value_range: typing.Tuple[float, float] = (0, 1),
) -> typing.Dict[typing.Any, float]:
    r"""Return the precision of the binned prediction values per bin.

    Args:
        truth: truth values
        prediction: prediction values
        bins: number of bins the truth and prediction values should be binned to
        value_range: the target value range. truth and prediction may still contain
            higher or lower values, and they will still be mapped to the closest bin

    Returns:
        dictionary containing the precision between 0 and 1
        for each bin.

    """
    truth = bin_values(truth, bins, value_range=value_range)
    prediction = bin_values(prediction, bins, value_range=value_range)
    return audmetric.precision_per_class(
        truth, prediction, labels=truth.cat.categories.to_list()
    )


def recall_per_bin(
    truth: pd.Series,
    prediction: pd.Series,
    bins: int,
    value_range: typing.Tuple[float, float] = (0, 1),
) -> typing.Dict[typing.Any, float]:
    r"""Return the recall of the binned prediction values per bin.

    Args:
        truth: truth values
        prediction: prediction values
        bins: number of bins the truth and prediction values should be binned to
        value_range: the target value range. truth and prediction may still contain
            higher or lower values, and they will still be mapped to the closest bin

    Returns:
        dictionary containing the recall between 0 and 1
        for each bin.

    """
    truth = bin_values(truth, bins, value_range=value_range)
    prediction = bin_values(prediction, bins, value_range=value_range)
    return audmetric.recall_per_class(
        truth, prediction, labels=truth.cat.categories.to_list()
    )


def relative_difference_per_bin(
    truth: pd.Series,
    prediction: pd.Series,
    bins: int,
    value_range: typing.Tuple[float, float] = (0, 1),
) -> typing.Dict[typing.Any, float]:
    r"""Return the percentage of change in samples for each class.

    Args:
        truth: truth values
        prediction: prediction values
        bins: number of bins the truth and prediction values should be binned to
        value_range: the target value range. truth and prediction may still contain
            higher or lower values, and they will still be mapped to the closest bin

    Returns:
        dictionary containing a float value between -1 and 1
        for each bin.
        A value of 0.2 would indicate
        that the prediction contains 20% more samples in this bin.

    """
    truth = bin_values(truth, bins, value_range=value_range)
    prediction = bin_values(prediction, bins, value_range=value_range)
    result = relative_difference_per_class(
        truth, prediction, class_labels=truth.cat.categories.to_list()
    )
    return result


def relative_difference_per_class(
    truth: pd.Series,
    prediction: pd.Series,
    class_labels: typing.Optional[typing.Sequence] = None,
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
    truth = truth.value_counts().sort_index(kind="stable").to_dict()
    prediction = prediction.value_counts().sort_index(kind="stable").to_dict()
    changes = {}
    for class_label in class_labels:
        if class_label not in prediction:
            prediction[class_label] = 0
        if class_label not in truth:
            truth[class_label] = 0
        changes[class_label] = (prediction[class_label] / n_pred_samples) - (
            truth[class_label] / n_truth_samples
        )
    return changes


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


def generate_categorical(choices, n_samples, probabilities=None):
    s = np.random.choice(a=choices, p=probabilities, size=n_samples)
    y = pd.Series(data=s)
    return y


def generate_gaussian(minimum, maximum, mu, sigma, n_samples):
    s = truncnorm.rvs(
        (minimum - mu) / sigma,
        (maximum - mu) / sigma,
        loc=mu,
        scale=sigma,
        size=n_samples,
    )
    y = pd.Series(data=s)
    return y


def simulate_rel_diff_class(n_samples, sample_probability, n_groups, seed):
    labels = ["anger", "happiness", "neutral", "sadness"]
    random.seed(seed)
    np.random.seed(seed)
    full_s = generate_categorical(
        choices=labels, n_samples=n_samples * n_groups, probabilities=sample_probability
    )
    # first n_samples belong to tested group
    group_s = full_s.head(n_samples)

    # now compare in relative difference per class
    res = relative_difference_per_class(full_s, group_s, class_labels=labels)
    # get the maximum difference per class
    max_diff = max([np.abs(v) for v in res.values()])
    return max_diff


def simulate_rel_diff_bin(n_samples, n_groups, seed):
    random.seed(seed)
    np.random.seed(seed)
    full_s = generate_gaussian(
        minimum=0, maximum=1, mu=0.5, sigma=1 / 6, n_samples=n_samples * n_groups
    )
    # first n_samples belong to tested group
    group_s = full_s.head(n_samples)

    # now compare in relative difference per BIN
    res = relative_difference_per_bin(full_s, group_s, bins=4, value_range=(0, 1))
    # get the maximum difference per class
    max_diff = max([np.abs(v) for v in res.values()])
    return max_diff


def simulate_mean_diff(n_samples, n_groups, seed):
    random.seed(seed)
    np.random.seed(seed)
    full_s = generate_gaussian(
        minimum=0, maximum=1, mu=0.5, sigma=1 / 6, n_samples=n_samples * n_groups
    )
    # first n_samples belong to tested group
    group_s = full_s.head(n_samples)

    # now compare in mean
    res = group_s.mean() - full_s.mean()
    return np.abs(res)


def simulate_mean_shift_diff(n_samples, n_groups, n_shift_groups, seed):
    random.seed(seed)
    np.random.seed(seed)
    groups = [str(i) for i in range(n_groups)]
    shifts = [str(i) for i in range(n_shift_groups)]

    full_s = generate_gaussian(
        minimum=0,
        maximum=1,
        mu=0.5,
        sigma=1 / 6,
        n_samples=n_samples * n_groups * n_shift_groups,
    )
    group_list = []
    shift_list = []
    for group in groups:
        group_list += [group] * n_samples * n_shift_groups
        for shift in shifts:
            shift_list += [shift] * n_samples
    group_values = pd.Series(data=group_list)
    context_values = pd.Series(data=shift_list)
    _, _, diff = mean_shift_difference(
        full_s,
        group_values=group_values,
        group="0",
        context_values=context_values,
        context="0",
    )
    return np.abs(diff)


def simulate_bin_proportion_shift_diff(n_samples, n_groups, n_shift_groups, seed):
    random.seed(seed)
    np.random.seed(seed)
    groups = [str(i) for i in range(n_groups)]
    shifts = [str(i) for i in range(n_shift_groups)]

    full_s = generate_gaussian(
        minimum=0,
        maximum=1,
        mu=0.5,
        sigma=1 / 6,
        n_samples=n_samples * n_groups * n_shift_groups,
    )
    group_list = []
    shift_list = []
    for group in groups:
        group_list += [group] * n_samples * n_shift_groups
        for shift in shifts:
            shift_list += [shift] * n_samples
    group_values = pd.Series(data=group_list)
    context_values = pd.Series(data=shift_list)
    _, _, diff_dict = bin_proportion_shift_difference(
        full_s,
        group_values=group_values,
        group="0",
        context_values=context_values,
        context="0",
        bins=4,
    )
    max_diff = max([np.abs(v) for v in diff_dict.values()])

    return np.abs(max_diff)


def simulate_class_proportion_shift_diff(
    n_samples, sample_probability, n_groups, n_shift_groups, seed
):
    labels = ["anger", "happiness", "neutral", "sadness"]

    random.seed(seed)
    np.random.seed(seed)
    groups = [str(i) for i in range(n_groups)]
    shifts = [str(i) for i in range(n_shift_groups)]

    full_s = generate_categorical(
        choices=labels,
        n_samples=n_samples * n_groups * n_shift_groups,
        probabilities=random.sample(sample_probability, len(labels)),
    )
    group_list = []
    shift_list = []
    for group in groups:
        group_list += [group] * n_samples * n_shift_groups
        for shift in shifts:
            shift_list += [shift] * n_samples
    group_values = pd.Series(data=group_list)
    context_values = pd.Series(data=shift_list)
    _, _, diff_dict = class_proportion_shift_difference(
        full_s,
        group_values=group_values,
        group="0",
        context_values=context_values,
        context="0",
        class_labels=labels,
    )

    max_diff = max([np.abs(v) for v in diff_dict.values()])
    return np.abs(max_diff)


def simulate_recall_per_class(
    n_samples, sample_probability_truth, sample_probability_pred, n_groups, seed, varied
):
    labels = ["anger", "happiness", "neutral", "sadness"]
    random.seed(seed)
    np.random.seed(seed)
    if varied:
        results = []
        for _ in range(n_groups):
            s_truth = generate_categorical(
                choices=labels,
                n_samples=n_samples,
                probabilities=random.sample(sample_probability_truth, len(labels)),
            )
            results.append(s_truth)
        full_s_truth = pd.concat(results)
    else:
        full_s_truth = generate_categorical(
            choices=labels,
            n_samples=n_samples * n_groups,
            probabilities=sample_probability_truth,
        )

    full_s_pred = generate_categorical(
        choices=labels,
        n_samples=n_samples * n_groups,
        probabilities=sample_probability_pred,
    )
    # first n_samples belong to tested group
    group_s_truth = full_s_truth.head(n_samples)

    group_s_pred = full_s_pred.head(n_samples)

    group_recall = audmetric.recall_per_class(
        group_s_truth, group_s_pred, labels=labels
    )
    full_recall = audmetric.recall_per_class(full_s_truth, full_s_pred, labels=labels)
    # get the maximum difference per class
    max_diff = max(
        [np.abs(group_recall[label] - full_recall[label]) for label in labels]
    )
    return max_diff


def simulate_recall(
    n_samples, sample_probability_truth, sample_probability_pred, n_groups, seed, varied
):
    labels = ["anger", "happiness", "neutral", "sadness"]
    random.seed(seed)
    np.random.seed(seed)

    if varied:
        results = []
        for _ in range(n_groups):
            s_truth = generate_categorical(
                choices=labels,
                n_samples=n_samples,
                probabilities=random.sample(sample_probability_truth, len(labels)),
            )
            results.append(s_truth)
        full_s_truth = pd.concat(results)
    else:
        full_s_truth = generate_categorical(
            choices=labels,
            n_samples=n_samples * n_groups,
            probabilities=sample_probability_truth,
        )

    full_s_pred = generate_categorical(
        choices=labels,
        n_samples=n_samples * n_groups,
        probabilities=sample_probability_pred,
    )
    # first n_samples belong to tested group
    group_s_truth = full_s_truth.head(n_samples)

    group_s_pred = full_s_pred.head(n_samples)

    group_prec = audmetric.unweighted_average_recall(
        group_s_truth, group_s_pred, labels=labels
    )
    full_prec = audmetric.unweighted_average_recall(
        full_s_truth, full_s_pred, labels=labels
    )

    return np.abs(group_prec - full_prec)


def simulate_precision_per_class(
    n_samples, sample_probability_truth, sample_probability_pred, n_groups, seed, varied
):
    labels = ["anger", "happiness", "neutral", "sadness"]
    random.seed(seed)
    np.random.seed(seed)
    if varied:
        results = []
        for _ in range(n_groups):
            s_truth = generate_categorical(
                choices=labels,
                n_samples=n_samples,
                probabilities=random.sample(sample_probability_truth, len(labels)),
            )
            results.append(s_truth)
        full_s_truth = pd.concat(results)
    else:
        full_s_truth = generate_categorical(
            choices=labels,
            n_samples=n_samples * n_groups,
            probabilities=sample_probability_truth,
        )

    full_s_pred = generate_categorical(
        choices=labels,
        n_samples=n_samples * n_groups,
        probabilities=sample_probability_pred,
    )
    # first n_samples belong to tested group
    group_s_truth = full_s_truth.head(n_samples)

    group_s_pred = full_s_pred.head(n_samples)

    group_prec = audmetric.precision_per_class(
        group_s_truth, group_s_pred, labels=labels
    )
    full_prec = audmetric.precision_per_class(full_s_truth, full_s_pred, labels=labels)
    # get the maximum difference per class
    max_diff = max([np.abs(group_prec[label] - full_prec[label]) for label in labels])
    return max_diff


def simulate_precision(
    n_samples, sample_probability_truth, sample_probability_pred, n_groups, seed, varied
):
    labels = ["anger", "happiness", "neutral", "sadness"]
    random.seed(seed)
    np.random.seed(seed)

    if varied:
        results = []
        for _ in range(n_groups):
            s_truth = generate_categorical(
                choices=labels,
                n_samples=n_samples,
                probabilities=random.sample(sample_probability_truth, len(labels)),
            )
            results.append(s_truth)
        full_s_truth = pd.concat(results)
    else:
        full_s_truth = generate_categorical(
            choices=labels,
            n_samples=n_samples * n_groups,
            probabilities=sample_probability_truth,
        )

    full_s_pred = generate_categorical(
        choices=labels,
        n_samples=n_samples * n_groups,
        probabilities=sample_probability_pred,
    )
    # first n_samples belong to tested group
    group_s_truth = full_s_truth.head(n_samples)

    group_s_pred = full_s_pred.head(n_samples)

    group_prec = audmetric.unweighted_average_precision(
        group_s_truth, group_s_pred, labels=labels
    )
    full_prec = audmetric.unweighted_average_precision(
        full_s_truth, full_s_pred, labels=labels
    )

    return np.abs(group_prec - full_prec)


def simulate_recall_per_bin(n_samples, n_groups, seed, varied):
    random.seed(seed)
    np.random.seed(seed)
    if varied:
        mu_opts = [0.5, 0.55, 0.6]
        results = []
        for _ in range(n_groups):
            s_truth = generate_gaussian(
                minimum=0,
                maximum=1,
                mu=random.sample(mu_opts, 1)[0],
                sigma=1 / 6,
                n_samples=n_samples,
            )
            results.append(s_truth)
        full_s_truth = pd.concat(results)
    else:
        full_s_truth = generate_gaussian(
            minimum=0, maximum=1, mu=0.5, sigma=1 / 6, n_samples=n_samples * n_groups
        )

    full_s_pred = generate_gaussian(
        minimum=0, maximum=1, mu=0.5, sigma=1 / 6, n_samples=n_samples * n_groups
    )
    # first n_samples belong to tested group
    group_s_truth = full_s_truth.head(n_samples)

    group_s_pred = full_s_pred.head(n_samples)

    group_recall = recall_per_bin(group_s_truth, group_s_pred, bins=4)
    full_recall = recall_per_bin(full_s_truth, full_s_pred, bins=4)
    # get the maximum difference per class
    max_diff = max(
        [np.abs(group_recall[bin] - full_recall[bin]) for bin in group_recall.keys()]
    )
    return max_diff


def simulate_precision_per_bin(n_samples, n_groups, seed, varied):
    random.seed(seed)
    np.random.seed(seed)
    if varied:
        mu_opts = [0.5, 0.55, 0.6]
        results = []
        for _ in range(n_groups):
            s_truth = generate_gaussian(
                minimum=0,
                maximum=1,
                mu=random.sample(mu_opts, 1)[0],
                sigma=1 / 6,
                n_samples=n_samples,
            )
            results.append(s_truth)
        full_s_truth = pd.concat(results)
    else:
        full_s_truth = generate_gaussian(
            minimum=0, maximum=1, mu=0.5, sigma=1 / 6, n_samples=n_samples * n_groups
        )
    full_s_pred = generate_gaussian(
        minimum=0, maximum=1, mu=0.5, sigma=1 / 6, n_samples=n_samples * n_groups
    )
    # first n_samples belong to tested group
    group_s_truth = full_s_truth.head(n_samples)

    group_s_pred = full_s_pred.head(n_samples)

    group_prec = precision_per_bin(group_s_truth, group_s_pred, bins=4)
    full_prec = precision_per_bin(full_s_truth, full_s_pred, bins=4)
    # get the maximum difference per class
    max_diff = max(
        [np.abs(group_prec[bin] - full_prec[bin]) for bin in group_prec.keys()]
    )
    return max_diff


def simulate_ccc_diff(n_samples, n_groups, seed, varied):
    random.seed(seed)
    np.random.seed(seed)
    if varied:
        mu_opts = [0.5, 0.55, 0.6]
        results = []
        for _ in range(n_groups):
            s_truth = generate_gaussian(
                minimum=0,
                maximum=1,
                mu=random.sample(mu_opts, 1)[0],
                sigma=1 / 6,
                n_samples=n_samples,
            )
            results.append(s_truth)
        full_s_truth = pd.concat(results)
    else:
        full_s_truth = generate_gaussian(
            minimum=0, maximum=1, mu=0.5, sigma=1 / 6, n_samples=n_samples * n_groups
        )
    full_s_pred = generate_gaussian(
        minimum=0, maximum=1, mu=0.5, sigma=1 / 6, n_samples=n_samples * n_groups
    )
    # first n_samples belong to tested group
    group_s_truth = full_s_truth.head(n_samples)

    group_s_pred = full_s_pred.head(n_samples)

    group_ccc = audmetric.concordance_cc(group_s_truth, group_s_pred)
    full_ccc = audmetric.concordance_cc(full_s_truth, full_s_pred)

    return np.abs(group_ccc - full_ccc)


def simulate_mae_diff(n_samples, n_groups, seed, varied):
    random.seed(seed)
    np.random.seed(seed)
    if varied:
        mu_opts = [0.5, 0.55, 0.6]
        results = []
        for _ in range(n_groups):
            s_truth = generate_gaussian(
                minimum=0,
                maximum=1,
                mu=random.sample(mu_opts, 1)[0],
                sigma=1 / 6,
                n_samples=n_samples,
            )
            results.append(s_truth)
        full_s_truth = pd.concat(results)
    else:
        full_s_truth = generate_gaussian(
            minimum=0, maximum=1, mu=0.5, sigma=1 / 6, n_samples=n_samples * n_groups
        )
    full_s_pred = generate_gaussian(
        minimum=0, maximum=1, mu=0.5, sigma=1 / 6, n_samples=n_samples * n_groups
    )
    # first n_samples belong to tested group
    group_s_truth = full_s_truth.head(n_samples)

    group_s_pred = full_s_pred.head(n_samples)

    group_mae = audmetric.mean_absolute_error(group_s_truth, group_s_pred)
    full_mae = audmetric.mean_absolute_error(full_s_truth, full_s_pred)

    return np.abs(group_mae - full_mae)


def simulate_mde_diff(n_samples, n_groups, seed, varied):
    random.seed(seed)
    np.random.seed(seed)
    if varied:
        mu_opts = [0.5, 0.55, 0.6]
        results = []
        for _ in range(n_groups):
            s_truth = generate_gaussian(
                minimum=0,
                maximum=1,
                mu=random.sample(mu_opts, 1)[0],
                sigma=1 / 6,
                n_samples=n_samples,
            )
            results.append(s_truth)
        full_s_truth = pd.concat(results)
    else:
        full_s_truth = generate_gaussian(
            minimum=0, maximum=1, mu=0.5, sigma=1 / 6, n_samples=n_samples * n_groups
        )
    full_s_pred = generate_gaussian(
        minimum=0, maximum=1, mu=0.5, sigma=1 / 6, n_samples=n_samples * n_groups
    )
    # first n_samples belong to tested group
    group_s_truth = full_s_truth.head(n_samples)

    group_s_pred = full_s_pred.head(n_samples)

    group_mde = mean_directional_error(group_s_truth, group_s_pred)
    full_mde = mean_directional_error(full_s_truth, full_s_pred)

    return np.abs(group_mde - full_mde)


if __name__ == "__main__":
    N = 1000
    sample_probabilities = {
        "uniform": [0.25, 0.25, 0.25, 0.25],
        "sparse": [0.05, 0.05, 0.3, 0.6],
    }
    n_samples = [
        20,
        40,
        60,
        80,
        100,
        120,
        140,
        160,
        180,
        200,
        220,
        240,
        260,
        280,
        300,
        400,
        600,
        800,
        1000,
    ]

    simulations = [
        "relative_difference_per_class",
        "class_shift_diff",
        "relative_difference_per_bin",
        "bin_shift_diff",
        "mean",
        "mean_shift_diff",
        "recall_per_class",
        "precision_per_class",
        "recall_per_bin",
        "precision_per_bin",
        "recall",
        "precision",
        "ccc",
        "mae",
        "mde",
    ]

    n_groups = [2, 3, 6, 8, 20, 30]
    res_path = "simulation.csv"
    required_cols = [
        "n_samples",
        "n_groups",
        "max",
        "distribution",
        "simulation",
        "distribution_truth",
        "distribution_pred",
        "varied",
    ]
    if os.path.exists(res_path):
        res_df = pd.read_csv(res_path)
        res_df = res_df[res_df["n_groups"].isin(n_groups)]
        res_df = res_df[res_df["n_samples"].isin(n_samples)]
        for col in required_cols:
            if col not in res_df:
                if col == "varied":
                    res_df[col] = False
                else:
                    res_df[col] = None

    else:
        res_df = pd.DataFrame(columns=required_cols)
    result_list = []
    for simulation in simulations:
        print(simulation)
        res = []

        if simulation in ["relative_difference_per_class", "class_shift_diff"]:
            for (
                sample_probability_name,
                sample_probability,
            ) in sample_probabilities.items():
                for n_group in n_groups:
                    for n in n_samples:
                        rows = res_df[
                            (res_df["simulation"] == simulation)
                            & (res_df["distribution"] == sample_probability_name)
                            & (res_df["n_groups"] == n_group)
                            & (res_df["n_samples"] == n)
                        ]
                        if len(rows) > 0:
                            continue
                        max_diffs = []
                        for i in range(1, N + 1):
                            if simulation == "relative_difference_per_class":
                                max_diff = simulate_rel_diff_class(
                                    n,
                                    sample_probability=sample_probability,
                                    n_groups=n_group,
                                    seed=i,
                                )
                            elif simulation == "class_shift_diff":
                                # hard code n_shift_groups/n_context to 3
                                max_diff = simulate_class_proportion_shift_diff(
                                    n,
                                    sample_probability=sample_probability,
                                    n_groups=n_group,
                                    n_shift_groups=3,
                                    seed=i,
                                )
                            max_diffs.append(max_diff)
                        max_diff_s = pd.Series(data=max_diffs)
                        res.append(
                            {
                                "n_samples": n,
                                "n_groups": n_group,
                                "max": max(max_diffs),
                                "distribution": sample_probability_name,
                                "simulation": simulation,
                                "99_perc": np.percentile(max_diffs, 99),
                            }
                        )
        elif simulation in [
            "relative_difference_per_bin",
            "bin_shift_diff",
            "mean",
            "mean_shift_diff",
        ]:
            for n_group in n_groups:
                for n in n_samples:
                    rows = res_df[
                        (res_df["simulation"] == simulation)
                        & (res_df["n_groups"] == n_group)
                        & (res_df["n_samples"] == n)
                    ]
                    if len(rows) > 0:
                        continue
                    max_diffs = []
                    for i in range(1, N + 1):
                        if simulation == "relative_difference_per_bin":
                            max_diff = simulate_rel_diff_bin(
                                n, n_groups=n_group, seed=i
                            )
                        elif simulation == "bin_shift_diff":
                            max_diff = simulate_bin_proportion_shift_diff(
                                n, n_groups=n_group, n_shift_groups=3, seed=i
                            )
                        elif simulation == "mean":
                            max_diff = simulate_mean_diff(n, n_groups=n_group, seed=i)
                        elif simulation == "mean_shift_diff":
                            max_diff = simulate_mean_shift_diff(
                                n, n_groups=n_group, n_shift_groups=3, seed=i
                            )

                        max_diffs.append(max_diff)
                    max_diff_s = pd.Series(data=max_diffs)
                    res.append(
                        {
                            "n_samples": n,
                            "n_groups": n_group,
                            "max": max(max_diffs),
                            "simulation": simulation,
                            "99_perc": np.percentile(max_diffs, 99),
                        }
                    )
        elif simulation in [
            "recall_per_class",
            "precision_per_class",
            "recall",
            "precision",
        ]:
            for (
                sample_probability_name_truth,
                sample_probability_truth,
            ) in sample_probabilities.items():
                for (
                    sample_probability_name_pred,
                    sample_probability_pred,
                ) in sample_probabilities.items():
                    for varied in [False, True]:
                        # Only vary the sample probability for the truth if not uniform
                        if varied and sample_probability_name_truth == "uniform":
                            continue
                        for n_group in n_groups:
                            for n in n_samples:
                                rows = res_df[
                                    (res_df["simulation"] == simulation)
                                    & (
                                        res_df["distribution_truth"]
                                        == sample_probability_name_truth
                                    )
                                    & (
                                        res_df["distribution_pred"]
                                        == sample_probability_name_pred
                                    )
                                    & (res_df["n_groups"] == n_group)
                                    & (res_df["n_samples"] == n)
                                    & (res_df["varied"] == varied)
                                ]
                                if len(rows) > 0:
                                    continue
                                max_diffs = []
                                for i in range(1, N + 1):
                                    if simulation == "recall_per_class":
                                        max_diff = simulate_recall_per_class(
                                            n,
                                            sample_probability_truth=sample_probability_truth,
                                            sample_probability_pred=sample_probability_pred,
                                            n_groups=n_group,
                                            seed=i,
                                            varied=varied,
                                        )
                                    elif simulation == "precision_per_class":
                                        max_diff = simulate_precision_per_class(
                                            n,
                                            sample_probability_truth=sample_probability_truth,
                                            sample_probability_pred=sample_probability_pred,
                                            n_groups=n_group,
                                            seed=i,
                                            varied=varied,
                                        )
                                    elif simulation == "recall":
                                        max_diff = simulate_recall(
                                            n,
                                            sample_probability_truth=sample_probability_truth,
                                            sample_probability_pred=sample_probability_pred,
                                            n_groups=n_group,
                                            seed=i,
                                            varied=varied,
                                        )
                                    elif simulation == "precision":
                                        max_diff = simulate_precision(
                                            n,
                                            sample_probability_truth=sample_probability_truth,
                                            sample_probability_pred=sample_probability_pred,
                                            n_groups=n_group,
                                            seed=i,
                                            varied=varied,
                                        )
                                    max_diffs.append(max_diff)
                                max_diff_s = pd.Series(data=max_diffs)

                                res.append(
                                    {
                                        "n_samples": n,
                                        "n_groups": n_group,
                                        "max": max(max_diffs),
                                        "distribution_truth": sample_probability_name_truth,
                                        "distribution_pred": sample_probability_name_pred,
                                        "simulation": simulation,
                                        "99_perc": np.percentile(max_diffs, 99),
                                        "varied": varied,
                                    }
                                )
        elif simulation in ["recall_per_bin", "precision_per_bin", "ccc", "mde", "mae"]:
            for n_group in n_groups:
                for n in n_samples:
                    for varied in [False, True]:
                        rows = res_df[
                            (res_df["simulation"] == simulation)
                            & (res_df["n_groups"] == n_group)
                            & (res_df["n_samples"] == n)
                            & (res_df["varied"] == varied)
                        ]
                        if len(rows) > 0:
                            continue
                        max_diffs = []
                        for i in range(1, N + 1):
                            if simulation == "recall_per_bin":
                                max_diff = simulate_recall_per_bin(
                                    n, n_groups=n_group, seed=i, varied=varied
                                )
                            elif simulation == "precision_per_bin":
                                max_diff = simulate_precision_per_bin(
                                    n, n_groups=n_group, seed=i, varied=varied
                                )
                            elif simulation == "ccc":
                                max_diff = simulate_ccc_diff(
                                    n, n_groups=n_group, seed=i, varied=varied
                                )
                            elif simulation == "mae":
                                max_diff = simulate_mae_diff(
                                    n, n_groups=n_group, seed=i, varied=varied
                                )
                            elif simulation == "mde":
                                max_diff = simulate_mde_diff(
                                    n, n_groups=n_group, seed=i, varied=varied
                                )
                            max_diffs.append(max_diff)
                        max_diff_s = pd.Series(data=max_diffs)
                        res.append(
                            {
                                "n_samples": n,
                                "n_groups": n_group,
                                "max": max(max_diffs),
                                "simulation": simulation,
                                "99_perc": np.percentile(max_diffs, 99),
                                "varied": varied,
                            }
                        )
        result_list.append(res)
    res = sum(result_list, start=[])
    new_res_df = pd.DataFrame(data=res)
    res_df = pd.concat((res_df, new_res_df))
    res_df.to_csv(res_path, index=False)
