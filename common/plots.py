import typing
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import audmetric
import audplot

from . import (
    quantile,
)

SHORT_CLASSES = {
    'anger': 'anger',
    'happiness': 'happiness',
    'neutral': 'neutral',
    'sadness': 'sadness'
}


def plot_average_value(
        truth: typing.Union[pd.Series, pd.DataFrame],
        prediction: typing.Union[pd.Series,pd.DataFrame],
        xlabel: str,
        threshold: float,
):
    r"""Plot average truth and prediction value to same figure.

    The values are ordered after the values
    given in ``truth`` for regression. For classification,
    the values stay in the same order as they are passed, 
    because there are multiple subplots, one per class.
    Args:
        truth: truth values of average values, or average values per class
        prediction: prediction values of average values, or average values per class

    """
    # Categorical case, where we have class-wise results
    if isinstance(truth, pd.DataFrame):
        class_labels = truth.columns
        dfs = {}
        for class_label in class_labels:
            class_truth = truth[class_label].astype('float32')
            class_prediction = prediction[class_label].astype('float32')

            df_truth = pd.DataFrame(
                data=class_truth.values,
                columns=['Truth'],
                index=class_truth.index,
                dtype='float32',
            )
            df_prediction = pd.DataFrame(
                data=class_prediction.values,
                columns=['Prediction'],
                index=class_prediction.index,
                dtype='float32',
            )
            dfs[f'Proportion of {class_label} samples'] = (df_truth, df_prediction)
    # Regression case, where we have one average value
    else:
        truth = truth.astype('float32')
        prediction = prediction.astype('float32')

        # Sort results after average value
        # NOTE: 'stable' ensures that equal entries are always returned
        # in the same order
        truth = truth.sort_values(kind='stable')
        prediction = prediction.sort_values(kind='stable')

        df_truth = pd.DataFrame(
            data=truth.values,
            columns=['Truth'],
            index=truth.index,
            dtype='float32',
        )
        df_prediction = pd.DataFrame(
            data=prediction.values,
            columns=['Prediction'],
            index=prediction.index,
            dtype='float32',
        )
        dfs = {'Average Value': (df_truth, df_prediction)}
    colors = {
        True: '#318b57',
        False: '#dd223c',
    }

    _, axs = plt.subplots(len(dfs), 1, figsize=[6.4, 3.2*len(dfs)])
    # Make iterating over multiple plots easier by enforcing
    # axs to be a list
    if len(dfs) == 1:
        axs = [axs]

    for i, (y_label, (df_truth, df_prediction)) in enumerate(dfs.items()):
        df_prediction = df_prediction.loc[df_truth.index]
        df_truth = df_truth.reset_index()
        df_truth = df_truth.drop('index', axis=1)
        df_truth = df_truth.reset_index()
        df_truth = df_truth.rename(columns={'index': xlabel})
        df_prediction = df_prediction.reset_index()
        df_prediction = df_prediction.drop('index', axis=1)
        df_prediction = df_prediction.reset_index()
        df_prediction = df_prediction.rename(columns={'index': xlabel})
        df_prediction['Passed'] = (
            df_prediction['Prediction'] - df_truth['Truth']
        ).abs() < threshold
        sns.scatterplot(
            data=df_truth,
            x=xlabel,
            y='Truth',
            color='#828282',
            marker='o',
            label='Truth',
            legend=False,
            ax=axs[i],
        )
        g = sns.scatterplot(
            data=df_prediction,
            x=xlabel,
            y='Prediction',
            hue='Passed',
            palette=colors,
            marker='X',
            legend=False,
            ax=axs[i],
        )
        axs[i].set_ylim([0, 1])
        # Adjust legend position and entries
        markersize = 6
        legend_elements = [
            matplotlib.lines.Line2D(
                [0],
                [0],
                marker='o',
                color='#828282',
                label='Truth',
                markersize=markersize,
                linewidth=0,
            ),
            matplotlib.lines.Line2D(
                [0],
                [0],
                marker='X',
                color=colors[True],
                label=f'Prediction (|Δ|<{threshold})',
                markersize=markersize,
                linewidth=0,
            ),
            matplotlib.lines.Line2D(
                [0],
                [0],
                marker='X',
                color=colors[False],
                label=f'Prediction (|Δ|>={threshold})',
                markersize=markersize,
                linewidth=0,
            ),
        ]
        # In the case of classification, values are not sorted
        # and the legend in the lower right might cover up
        # some points. Try to improve this by putting the
        # legend in the upper right when the last 3 values are
        # below 0.3
        last_min_truth = df_truth['Truth'].tail(3).min()
        last_min_pred = df_prediction['Prediction'].tail(3).min()
        last_min = min(last_min_truth, last_min_pred)
        if last_min < 0.3:
            legend_position = 'upper right'
        else:
            legend_position = 'lower right'
        axs[i].legend(handles=legend_elements, loc=legend_position)
        length = len(truth)
        axs[i].set_xlim([-0.5, length - 0.5])
        axs[i].set_xlabel(xlabel)
        axs[i].set_ylabel(y_label)
        # Add Grid lines behind all other elements
        axs[i].grid(alpha=0.4)
        g.set_axisbelow(True)
        sns.despine()


def plot_confusion_matrix(
    truth: pd.Series,
    prediction: pd.Series,
    *,
    labels: typing.Sequence = None,
    short_labels: bool = True,
):
    r"""Plot confusion matrix."""
    if labels is not None and short_labels:
        label_aliases = SHORT_CLASSES
    else:
        label_aliases = None
    audplot.confusion_matrix(
        truth,
        prediction,
        percentage=True,
        show_both=True,
        labels=labels,
        label_aliases=label_aliases,
    )


def plot_shift(
        truth: pd.Series,
        prediction: pd.Series,
        order: typing.List[str] = None,
        *,
        truth_label: str = 'Truth',
        prediction_label: str = 'Prediction',
        y_range: typing.Optional[typing.Tuple[float, float]] = None,
        allowed_range: typing.Optional[typing.Tuple[float, float]] = None,
        bins: typing.Union[str, float, typing.Sequence] = 'auto',
):
    r"""Plot shift from one distribution to another.

    It plots the residuals (prediction - truth) for regression
    via a 2d histplot, highlighting the allowed error range
    if ``allowed_range`` is provided, as well as the distribution plots
    for both prediction and truth.
    If the ``order`` argument is provided
    it used :func:`audplot.confusion_matrix` instead
    to show the change in class predictions.

    Args:
        truth: truth values
        prediction: prediction values
        order: order of labels on x-axis.
            If ``None`` the distribution is plotted
            without class labels on the x-axis
        truth_label: label to describe truth
        prediction_label: label to describe prediciton
        allowed_range: allowed regression error range
        bins: bins for creating the distribution
    """

    if order is None:
        _, axs = plt.subplots(2, 1, figsize=[6.4, 8.8])
        residuals = prediction.values - truth.values
        y_label = f'{prediction_label} - {truth_label}'
        data = pd.DataFrame(
            {
                truth_label: truth.values,
                y_label: residuals}
        )
        sns.histplot(
            data=data,
            x=truth_label,
            y=y_label,
            ax=axs[0],
            bins=bins,
        )
        axs[0].axhline(y=0, color='g', linestyle='--')
        if allowed_range is not None:
            axs[0].fill(
                [0, 0, 1, 1],
                [allowed_range[0], allowed_range[1],
                    allowed_range[1], allowed_range[0]],
                color='green', alpha=0.15, zorder=-1
            )
        axs[0].grid(alpha=0.4)
        sns.despine(ax=axs[0])
        # Force y ticks at integer locations
        axs[0].yaxis.set_major_locator(
            matplotlib.ticker.MaxNLocator(integer=True))
        axs[0].set_xlim([0, 1])
        if y_range is not None:
            axs[0].set_ylim(y_range)
        data = pd.DataFrame(
            data=np.array([truth, prediction]).T,
            columns=[truth_label, prediction_label],
        )
        sns.histplot(
            data,
            common_bins=False,
            stat='frequency',
            kde=True,
            edgecolor=None,
            kde_kws={'cut': 3},  # hard code like in distplot()
            ax=axs[1],
            bins=bins[0],
        )
        axs[1].set_xlim([0, 1])
        axs[1].set_xlabel('Prediction')
        axs[1].grid(alpha=0.4)
        sns.despine(ax=axs[1])
    else:
        palette = sns.color_palette()
        if len(truth) == 0:
            # Skip first color if no truth given
            palette = palette[1:]
        _, ax = plt.subplots(1, 1)
        audplot.confusion_matrix(
            truth, prediction, labels=order,
            percentage=True, show_both=True, ax=ax)
        ax.set_xlabel(prediction_label)
        ax.set_ylabel(truth_label)


def plot_distribution(
        truth: pd.Series,
        prediction: pd.Series,
        order: typing.List[str] = None,
        *,
        truth_label: str = 'Truth',
        prediction_label: str = 'Prediction',
        bins: typing.Union[str, float, typing.Sequence] = 'auto',
):
    r"""Plot truth and prediction distributions in same figure.

    It uses :func:`audplot.distribution`
    to show regression output.
    If the ``order`` argument is provided
    it used :func:`seaborn.countplot` instead
    to show single classes.

    Args:
        truth: truth values
        prediction: prediction values
        order: order of labels on x-axis.
            If ``None`` a distribution is plotted
            without class labels on the x-axis
        truth_label: the label to be shown in the legend for truth
        prediction_label: the label to be shown in the legend for prediction
        bins: bin parameter passed to histplot for regression
    """
    if truth_label is None:
        truth_label = '_hidden'
    if prediction_label is None:
        prediction_label = '_hidden'

    if order is None:
        _, ax = plt.subplots(1, 1)
        data = pd.DataFrame(
            data=np.array([truth, prediction]).T,
            columns=[truth_label, prediction_label],
        )
        # Ignore warnings regarding _hidden legend label
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sns.histplot(
                data,
                common_bins=False,
                stat='frequency',
                kde=True,
                edgecolor=None,
                kde_kws={'cut': 3},  # hard code like in distplot()
                ax=ax,
                bins=bins,
            )
        ax.grid(alpha=0.4)
        sns.despine(ax=ax)
        # Force y ticks at integer locations
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        plt.xlim([0, 1])
    else:
        palette = sns.color_palette()
        if len(truth) == 0:
            # Skip first color if no truth given
            palette = palette[1:]
        columns = ['Result', 'Type']
        truth = np.array(
            [truth, [truth_label] * len(truth)]
        ).T
        prediction = np.array(
            [prediction, [prediction_label] * len(prediction)]
        ).T
        df = pd.concat(
            [
                pd.DataFrame(data=truth, columns=columns),
                pd.DataFrame(data=prediction, columns=columns),
            ]
        )
        sns.countplot(
            data=df,
            x='Result',
            hue='Type',
            order=order,
            palette=palette,
        )
        # Remove 'Type' title from legend
        plt.gca().legend().set_title('')
    plt.xlabel('Value')


def plot_distribution_by_category(
        truth: pd.Series,
        prediction: pd.Series,
        order: typing.List[str] = None,
        expected_ranges: typing.Optional[
            typing.Dict[str, typing.Tuple[float, float]]
        ] = None,
):
    r"""Plot multiple distributions as boxplots separated by category.

    If the ``expected_ranges`` parameter is given, also highlight the
    area of the expected value range for each category

    Args:
        truth: ground truth category labels
        prediction: predicted dimensional values
        order: order of category labels
        expected_ranges: dict mapping from category to the expecte range

    """
    truth_label = 'Truth'
    prediction_label = 'Prediction'
    if order is None:
        order = sorted(truth.unique())
    ax = plt.gca()
    data = pd.DataFrame(
        {truth_label: truth,
         prediction_label: prediction}
    )
    # Always use the same colors for categories
    palette = {
        'anger': '#d62728',
        'boredom': '#7f7f7f',
        'disgust': '#2ca02c',
        'fear': '#ff7f0e',
        'frustration': '#17becf',
        'happiness': '#e377c2',
        'neutral': '#8c564b',
        'sadness': '#1f77b4',
        'surprise': '#9467bd'
    }

    # Ignore warnings regarding _hidden legend label
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sns.boxplot(
            data=data,
            x=prediction_label,
            y=truth_label,
            ax=ax,
            order=order,
            width=0.5,
            palette=palette
        )
    # draw green boxes in the background to highlight
    # the expected ranges
    if expected_ranges is not None:
        half_box_width = 0.4
        for y_ind, truth_val in enumerate(order):
            expected_range = expected_ranges[truth_val]
            x_min = expected_range[0] if expected_range[0] is not None else 0
            x_max = expected_range[1] if expected_range[1] is not None else 1
            ax.fill(
                [x_min, x_min, x_max, x_max],
                [y_ind-half_box_width, y_ind+half_box_width,
                    y_ind+half_box_width, y_ind-half_box_width],
                color='green', alpha=0.3, zorder=-1
            )
    ax.grid(alpha=0.4)
    sns.despine(ax=ax)
    plt.xlim([0, 1])
    plt.xlabel('Prediction dimensional value')


def plot_distribution_normalized(
    truth: pd.Series,
    prediction: pd.Series,
    order: typing.List[str] = None,
    truth_groups: typing.Optional[pd.Series] = None,
    prediction_groups: typing.Optional[pd.Series] = None,
    group_order: typing.Optional[typing.List[str]] = None,
    *,
    truth_label: str = 'Truth',
    prediction_label: str = 'Prediction',
    bins: typing.Union[str, float, typing.Sequence] = 'auto',
):
    r"""Plot normalized truth and prediction distributions for each group.

    It splits the truth and prediction distributions by a group variable
    and creates one plot for each group
    containing corresponding truth and prediction, which are normalized
    individually.

    Args:
        truth: truth values
        prediction: prediction values
        truth_groups: optional group values for the truth
        prediction_groups: optional group values for the prediction
        group_order: the order in which the groups should be shown, if provided
        order: order of labels on x-axis.
            If ``None`` a distribution is plotted
            without class labels on the x-axis
        truth_label: the label to be shown in the legend for truth
        prediction_label: the label to be shown in the legend for prediction
        bins: bin parameter passed to histplot for regression

    """
    df_truth = pd.DataFrame(data={'Value': truth}, index=truth.index)
    row = None
    row_order = None
    if truth_groups is not None:
        df_truth = pd.concat((df_truth, truth_groups), axis=1)
        row=truth_groups.name
        row_order = group_order
    df_truth['Type'] = truth_label

    df_pred = pd.DataFrame(data={'Value': prediction}, index=prediction.index)
    if truth_groups is not None:
        df_pred = pd.concat((df_pred, prediction_groups), axis=1)
    df_pred['Type'] = prediction_label
    df = pd.concat((df_truth, df_pred))
    df.reset_index(inplace=True, drop=True)

    palette = sns.color_palette()
    palette = palette[:2]
    if len(truth) == 0:
        # Skip first color if no truth given
        palette = palette[1:]
    if order is None:
        g = sns.displot(
            data=df,
            x='Value',
            row=row,
            row_order=row_order,
            hue='Type',
            common_bins=True,
            stat='percent',
            common_norm=False,
            kde=True,
            edgecolor=None,
            kde_kws={'cut': 3},
            bins=bins,
            facet_kws={'legend_out': False},
            height=3.2,
            aspect=2,
        )
        g.set(xlim=(0, 1))
    else:
        # Use Categorical datatype to ensure the order of the class labels
        df['Value'] = pd.Categorical(df['Value'], order)
        g = sns.displot(
            data=df,
            x='Value',
            row=row,
            row_order=row_order,
            hue='Type',
            palette=palette,
            multiple='dodge',
            shrink=.8,
            common_norm=False,
            stat='percent',
            edgecolor=None,
            facet_kws={'legend_out': False},
            height=3.2,
            aspect=2,
        )
    g.legend.set_title('')


def plot_distribution_value_changes(
        truth: pd.Series,
        prediction: pd.Series,
        bins: typing.Sequence,
):
    r"""Plot truth and prediction distributions.

    Args:
        truth: truth values
        prediction: prediction values
        bins: bins for creating the distribution

    """
    ax = plt.gca()
    data = pd.DataFrame(
        data=np.array([truth, prediction]).T,
        columns=['Truth', 'Prediction'],
        dtype='float32',
    )
    sns.histplot(
        data,
        common_bins=False,
        stat='frequency',
        kde=False,
        edgecolor=None,
        bins=bins,
        ax=ax,
    )
    ax.grid(alpha=0.4)
    sns.despine(ax=ax)
    # Force y ticks at integer locations
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_xlim([0, max(bins)])
    plt.xlabel('Change per Segment')


def plot_error_change(
    truth: pd.Series,
    prediction_baseline: pd.Series,
    prediction_new: pd.Series,
    order: typing.List[str] = None,
    threshold: typing.Optional[float] = None,
    y_range: typing.Optional[typing.Tuple[float, float]] = None,
    allowed_range: typing.Optional[typing.Tuple[float, float]] = None,
    bins: typing.Union[str, float, typing.Sequence] = 'auto',
):
    truth_label = 'Truth'
    new_label = 'Prediction New'
    baseline_label = 'Prediction Baseline'
    error_types = [
        'New Errors',
        'Changed Errors',
        'Corrections'
    ]
    palette = {
        'New Errors': '#d62728',
        'Changed Errors': '#ff7f0e',
        'Corrections': '#2ca02c',
    }
    cmap = {
        'New Errors': 'Reds',
        'Changed Errors': 'Oranges',
        'Corrections': 'Greens',
    }
    data = pd.DataFrame(
        {
            truth_label: truth,
            'Prediction Baseline': prediction_baseline,
            new_label: prediction_new,
        }
    )

    if threshold is None:
        if order is None:
            order = sorted(truth.unique())

        def get_error_change(row):
            if row[truth_label] == row[new_label]:
                if row[truth_label] != row[baseline_label]:
                    return 'Corrections'
            else:
                if row[baseline_label] == row[truth_label]:
                    return 'New Errors'
                elif row[baseline_label] != row[new_label]:
                    return 'Changed Errors'
            return 'Unchanged'

        data['Error Type'] = data.apply(get_error_change, axis=1)

        _, axs = plt.subplots(4, 1, figsize=[6.4, 12.8])
        for i, error_type in enumerate(error_types):
            plot_df = data[data['Error Type'] == error_type]
            cm = audmetric.confusion_matrix(
                plot_df[baseline_label],
                plot_df[new_label],
                labels=order,
                normalize=False,
            )
            cm = pd.DataFrame(cm, index=order)
            annot = cm.applymap(lambda x: audplot.human_format(x))
            annot2 = cm.applymap(
                lambda x: f'{100 *x:.0f}%' if x == 0
                else f'{100 *x/len(data):.0f}%'
            )

            def combine_string(x, y):
                return f'{x}\n({y})'

            combine_string = np.vectorize(combine_string)
            annot = pd.DataFrame(combine_string(annot, annot2), index=order)
            sns.heatmap(
                cm,
                annot=annot,
                xticklabels=order,
                yticklabels=order,
                cbar=False,
                fmt='',
                cmap=cmap[error_type],
                ax=axs[i],
                vmin=0,
                vmax=len(data)/len(order),
            )
            axs[i].set_xlabel(new_label)
            axs[i].set_ylabel(baseline_label)
            axs[i].tick_params(axis='y', rotation=0)
            axs[i].set_title(error_type)

        sns.countplot(
            data=data,
            x=truth_label,
            hue='Error Type',
            hue_order=error_types,
            order=order,
            palette=palette,
            ax=axs[3]
        )
    else:
        def get_error_change(row):
            if abs(row[new_label] - row[baseline_label]) >= threshold:
                if abs(row[truth_label] - row[new_label]) < threshold:
                    if abs(row[truth_label] - row[baseline_label]) \
                            >= threshold:
                        return 'Corrections'
                else:
                    if abs(row[baseline_label] - row[truth_label]) < threshold:
                        return 'New Errors'
                    else:
                        return 'Changed Errors'
            return 'Unchanged'

        data['Error Type'] = data.apply(get_error_change, axis=1)
        _, axs = plt.subplots(4, 1, figsize=[6.4, 12.8])
        y_label = f'Prediction New - {baseline_label}'
        data[y_label] = (
            data[new_label].values - data[baseline_label].values
        )
        # Plot changes in prediction compared to the baseline
        # Split by Error Type
        for i, error_type in enumerate(error_types):
            axs[i].axhline(y=0, color='k', linestyle='--')
            if allowed_range is not None:
                axs[i].fill(
                    [0, 0, 1, 1],
                    [allowed_range[0], allowed_range[1],
                        allowed_range[1], allowed_range[0]],
                    color='gray', alpha=0.15, zorder=-1
                )
            axs[i].grid(alpha=0.4)
            sns.despine(ax=axs[i])
            # Force y ticks at integer locations
            axs[i].yaxis.set_major_locator(
                matplotlib.ticker.MaxNLocator(integer=True))
            axs[i].set_xlim([0, 1])
            if y_range is not None:
                axs[i].set_ylim(y_range)

            # Set maximum color scale to a proportion of the dataset
            # that depends on the number of bins.
            # For len(bins[0])+len(bins[1]) == 84, this corresponds to
            # ~1.19 percent of the dataset.
            vmax = len(data)/(len(bins[0])+len(bins[1]))
            sns.histplot(
                data=data[data['Error Type'] == error_type],
                x=baseline_label,
                y=y_label,
                ax=axs[i],
                color=palette[error_type],
                bins=bins,
                vmin=0,
                vmax=len(data)/(len(bins[0])+len(bins[1])),
                cbar=True,
            )

            # Change colorbar ticks labels to represent
            # the percentage of the dataset
            cbar = axs[i].collections[0].colorbar
            n_ticks = 6
            tick_locs = np.linspace(0, vmax, num=n_ticks)
            cbar.ax.yaxis.set_major_locator(
                matplotlib.ticker.FixedLocator(
                    locs=tick_locs
                )
            )
            cbar.ax.yaxis.set_ticklabels([
                f'{100 *x/len(data):.2f}%' for x in tick_locs
            ])
            axs[i].set_title(error_type)

        # Plot distribution of new prediction on truth
        sns.histplot(
            data,
            x=truth_label,
            hue='Error Type',
            hue_order=error_types,
            palette=palette,
            common_bins=False,
            stat='frequency',
            kde=True,
            edgecolor=None,
            kde_kws={'cut': 3},  # hard code like in distplot()
            ax=axs[3],
            bins=bins[0],
        )
        axs[3].set_xlim([0, 1])
        axs[3].grid(alpha=0.4)
        sns.despine(ax=axs[3])


def plot_ranking(
        truth_average: typing.Union[pd.Series, pd.DataFrame],
        prediction_average: typing.Union[pd.Series,pd.DataFrame],
        task: str,
):
    r"""Plot truth and prediction rankings in same figure.

    Useful for judging call/speaker rankings.

    Args:
        truth: truth values of average values, or average values per class
        prediction: prediction values of average values, or average values per class
        task: to be presented on the axis labels,
         e.g. ``'call'``

    """
    # Categorical case, where we have class-wise results
    if isinstance(truth_average, pd.DataFrame):
        values_dict = {
            f'{class_label.capitalize()} ': (
                truth_average[class_label], prediction_average[class_label]
            ) for class_label in truth_average.columns
        }
    else:
        values_dict ={'': (truth_average, prediction_average)}
    _, axs = plt.subplots(len(values_dict), 1, figsize=[6.4, 3.2*len(values_dict)])
    # Make iterating over multiple plots easier by enforcing
    # axs to be a list
    if len(values_dict) == 1:
        axs = [axs]
    for i, (prefix, (truth, prediction)) in enumerate(values_dict.items()):
        # Sort results after average value
        truth = truth.sort_values()
        truth = truth.reset_index()
        id_mapping = {
            row[1]['index']: row[0] for row in truth.iterrows()
        }
        prediction = prediction.sort_values()
        prediction.index = prediction.index.map(id_mapping)

        # Get length for 25%
        length = quantile(truth)
        truth_top = set(truth.iloc[-length:].index)
        truth_bottom = set(truth.iloc[:length].index)
        prediction_top = set(prediction.iloc[-length:].index)
        prediction_bottom = set(prediction.iloc[:length].index)
        # Get false positives, and false negatives
        true_positives = list(
            prediction_top.intersection(truth_top).union(
                prediction_bottom.intersection(truth_bottom)
            )
        )
        false_positives = list(
            (prediction_top - truth_top).union(
                prediction_bottom - truth_bottom
            )
        )
        false_negatives = list(
            (truth_top - prediction_top).union(
                truth_bottom - prediction_bottom
            )
        )
        top_bottom_confusions = list(
            truth_top.intersection(prediction_bottom).union(
                truth_bottom.intersection(prediction_top)
            )
        )

        truth_list = list(truth.index)
        prediction = pd.Series(
            list(range(len(prediction))),
            index=prediction.index,
        )
        prediction_list = list(prediction.loc[truth.index].values)
        xlabel = f'{prefix}{task.capitalize()} Ranking Truth'
        ylabel = f'{prefix}{task.capitalize()} Ranking Prediction'
        df = pd.DataFrame(
            data=np.array([truth_list, prediction_list]).T,
            columns=[xlabel, ylabel],
        )
        df['Ranking'] = 'true negatives'

        df.loc[false_positives, 'Ranking'] = 'false positives'
        df.loc[false_negatives, 'Ranking'] = 'false negatives'
        df.loc[true_positives, 'Ranking'] = 'true positives'
        df.loc[top_bottom_confusions, 'Ranking'] = 'top-bottom confusion'

        colors = {
            'true positives': '#318b57',
            'true negatives': '#828282',
            'false negatives': '#828282',
            'false positives': '#dd223c',
            'top-bottom confusion': '#dd223c',
        }
        markers = {
            'true positives': 'o',
            'true negatives': 'o',
            'false negatives': '^',
            'false positives': '^',
            'top-bottom confusion': 's',
        }
        order = [
            'top-bottom confusion',
            'true positives',
            'false positives',
            'true negatives',
            'false negatives',
        ]
        g = sns.scatterplot(
            data=df,
            x=xlabel,
            y=ylabel,
            hue='Ranking',
            hue_order=order,
            style_order=order,
            palette=colors,
            style='Ranking',
            markers=markers,
            legend=True,
            ax=axs[i],
        )

        # Remove legend title
        sns.move_legend(
            axs[i],
            'center left',
            bbox_to_anchor=(1, 0.5),
            title=None,
            fontsize=9,
            markerscale=1,
            frameon=False,
        )

        length = len(truth)
        region = quantile(truth)

        axs[i].set_xlim([-0.5, length])
        axs[i].set_ylim([-0.5, length])
        axs[i].set_aspect('equal')

        # Add Grid lines to highlight different statistic regions
        minor_ticks = [-0.5, region - 0.5, length - region - 0.5, length - 0.5]
        axs[i].set_yticks(minor_ticks, minor=True)
        axs[i].set_xticks(minor_ticks, minor=True)
        axs[i].tick_params(axis='both', which='minor', length=0)
        axs[i].grid(alpha=0.4, which='minor')
        g.set_axisbelow(True)

        # Add rectangle to mark Bottom 25% and Top 25%
        rect_bottom = plt.Rectangle(
            (-0.5, -0.5),
            region,
            region,
            facecolor='black',
            alpha=0.05,
        )
        rect_top = plt.Rectangle(
            (length - region - 0.5, length - region - 0.5),
            region,
            region,
            facecolor='black',
            alpha=0.05,
        )
        axs[i].add_patch(rect_bottom)
        axs[i].add_patch(rect_top)
        axs[i].text(
            x=1.05 * length,
            y=region / 2,
            s='Bottom 25%',
            color='black',
            alpha=0.3,
            horizontalalignment='left',
            verticalalignment='center',
        )
        axs[i].text(
            x=1.05 * length,
            y=length - region / 2,
            s='Top 25%',
            color='black',
            alpha=0.3,
            horizontalalignment='left',
            verticalalignment='center',
        )

        sns.despine()


def plot_robustness(df, name, robustness_name='Robustness'):
    r"""Plot robustness overview against changes.

    The different changes/augmentations are supposed
    to be provided as columns.
    The index should be the database,
    hence we expect a dataframe with a single entry.
    If the ``robustness_name`` is a common prefix of all
    columns, it is removed from the columns in the plot.

    Args:
        df: data frame
        name: name of database
        robustness_name: name of the robustness metric
    """
    plt.figure(figsize=[6.4, 3.2])
    ax = plt.subplot()

    colors = {
        'passed': '#318b57',
        'failed': '#dd223c',
    }

    # In order to fit long robustness metric names on the plot
    # remove the robustness_name prefix if it is shared among
    # all index entries
    plot_df = df.copy()
    shared_prefix = plot_df['index'].str.startswith(robustness_name).all()
    if shared_prefix:
        plot_df['index'] = plot_df['index'].str[len(robustness_name):]
        plot_df['index'] = plot_df['index'].str.lstrip()

    # Connect all points with a line
    ax.plot(plot_df.robustness, color='#828282', zorder=0)

    # Add points with different colors based on test result
    g = sns.pointplot(
        x='index',
        y='robustness',
        hue='result',
        data=plot_df,
        ax=ax,
        join=False,
        palette=colors,
    )
    g.legend_.remove()

    ax.set_title(name)

    # x-axis
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right',
        rotation_mode='anchor',
    )
    ax.set_xlabel('')

    # y-axis
    ax.set_yscale(  # exponential scale
        'function',
        functions=(
            lambda a: np.exp(a) - 1,
            lambda b: np.log(b + 1),
        ),
    )
    ax.set_ylim(0.25, 1.02)
    ax.set_ylabel(robustness_name)
    ax.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))

    # grid
    ax.grid(alpha=0.4)
    g.set_axisbelow(True)

    sns.despine()


def plot_scatter_pitch(
        pitch: pd.Series,
        values: pd.Series,
        sex: pd.Series,
        *,
        excluded_value_range: typing.Sequence[float] = None,
):
    r"""Plot pitch vs value per sex.

    Args:
        pitch: pitch values
        values: comparison values
        sex: sex values
        excluded_tone_range: all samples inside this range
            will be plotted in grey
            instead of coloring them by sex

    """
    if excluded_value_range is not None:
        r1, r2 = excluded_value_range
        excluded_index = values[(values > r1) & (values < r2)].index
        index = values[(values <= r1) | (values >= r2)].index
        ax = sns.scatterplot(
            x=pitch.loc[excluded_index],
            y=values.loc[excluded_index],
            color='#e3e3e3',
            legend=False,
        )
    else:
        index = values.index
        _, ax = plt.subplots()

    sns.scatterplot(
        x=pitch.loc[index],
        y=values.loc[index],
        hue=sex.loc[index],
        hue_order=['female', 'male'],
        palette={
            'female': '#c51b7d',
            'male': '#4d9221',
        },
        ax=ax,
    )
    plt.xlim(50, 350)
    plt.ylim(-0.1, 1.1)
    plt.xlabel('F0 / Hz')
    plt.ylabel('Values')

    ax.grid(alpha=0.4)
    ax.set_axisbelow(True)

    sns.despine()
