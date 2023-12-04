from collections import defaultdict, Counter
import importlib
import os
import re
import shutil
import sys
import typing
import yaml

import pandas as pd

import audeer

from common.correctness_consistency import (
    CATEGORY_LOW,
    CATEGORY_NEUTRAL,
    CATEGORY_HIGH,
)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(CURRENT_DIR, 'results')
TEST_RESULT_DIR = os.path.join(RESULT_DIR, 'test')
COMPARISON_RESULT_DIR = os.path.join(CURRENT_DIR, '..', 'comparison')
CONDITIONS = ['arousal', 'dominance', 'valence', 'emotion']
MODEL_ALIAS_PATH = os.path.join(CURRENT_DIR, 'model_names.yml')
with open(MODEL_ALIAS_PATH, 'r') as fp:
    MODEL_ALIASES = yaml.safe_load(fp)
sys.path.append(os.path.join(CURRENT_DIR, '..'))


def main(html_theme_options):
    r"""Collect results and create RST files.

    This scans :file:`docs/results/` for available results
    and generates all needed RST files
    to compile the HTML pages from those.

    The test results are expected to be saved as::

        results/test/{condition}/{model-id}/{test-name}/*

    The comparisons that should be displayed are expected to be specified in::
        comparison/{condition}.yaml

    The results are then presented on the HTML page as::

        Condition
            Model ID
                Test ID

    And the comparisons between a baseline and candidates are presented
    on the HTML page as::

        Condition
            Baseline ID vs. Candidate-1 ID vs. Candidate-2 ID
                Test ID

    Args:
        html_theme_options: html theme options for sphinx to add wide pages to

    """
    # Remove left-over files and create directory structure
    for folder in ['test', 'comparison']:
        audeer.rmdir(audeer.path(CURRENT_DIR, folder))
        audeer.mkdir(audeer.path(CURRENT_DIR, folder))
    create_results(html_theme_options)
    index_file = os.path.join(CURRENT_DIR, 'index.rst')
    with open(index_file, 'w') as fp:

        # Create landing page by displaying README
        fp.write('.. include:: ../README.rst\n')
        fp.write('\n')

        # Include a hidden TOC tree
        # to define the structure of the HTML page
        fp.write('\n')
        fp.write('.. toctree::\n')
        fp.write('    :caption: Documentation\n')
        fp.write('    :hidden:\n')
        fp.write('\n')
        fp.write('    installation\n')
        fp.write('    usage\n')
        fp.write('    method-tests\n')
        fp.write('    bibliography\n')
        fp.write('\n')
        fp.write('.. toctree::\n')
        fp.write('    :caption: Results\n')
        fp.write('    :hidden:\n')
        fp.write('\n')
        for condition in CONDITIONS:
            fp.write(f'    test/{condition}\n')
        fp.write('\n')
        fp.write('.. toctree::\n')
        fp.write('    :caption: Comparisons\n')
        fp.write('    :hidden:\n')
        fp.write('\n')
        for condition in CONDITIONS:
            fp.write(f'    comparison/{condition}\n')


def convert_csvs_to_comparison_table(title, model_results):
    # Parse model csv results, detect how many columns are required,
    # and detect potential reference columns
    result_dfs = {}
    for model, model_csv in model_results.items():
        model_df = pd.read_csv(model_csv, index_col=0)
        model_df.fillna('', inplace=True)
        if len(result_dfs) == 0:
            columns = model_df.columns
            non_ref_columns = [
                col for col in columns if 'Reference' not in col
            ]
            ref_columns = [col for col in columns if 'Reference' in col]
            index_name = model_df.index.name
            # Assume indices and reference columns are the same for each csv
            index = model_df.index
            reference_df = model_df.reset_index()
        result_dfs[model] = model_df.reset_index()
    n_models = len(model_results)
    model_aliases = [MODEL_ALIASES.get(model, model) for model in model_results.keys()]
    # Create flat-table from linux-doc to support nested columns
    res = f'.. flat-table:: {title}\n'

    # Write nested columns only if there is more than one model result,
    # otherwise create a regular table
    if len(model_results) == 1:
        res += '    :header-rows: 1\n\n'
        res += f'    * - {index_name}\n'
        for non_ref_column in non_ref_columns:
            res += f'      - {non_ref_column}\n'
        for ref_column in ref_columns:
            res += f'      - {ref_column}\n'

    else:
        res += '    :header-rows: 2\n\n'
        # Header row 1
        res += f'    * - :rspan:`1` {index_name}\n'
        for non_ref_column in non_ref_columns:
            res += f'      - :cspan:`{n_models - 1}` {non_ref_column}\n'
        for ref_column in ref_columns:
            res += f'      - :rspan:`1` {ref_column}\n'

        # Header row 2
        res += '    * - '
        res += '\n      - '.join(
            model_aliases * len(non_ref_columns)) + '\n'

    # Use i for indexing results, because rst references in columns
    # may be different
    for i, row in enumerate(index):
        res += f'    * - {row}\n'

        for non_ref_column in non_ref_columns:
            for model in model_results:
                res += f'      - {result_dfs[model].at[i, non_ref_column]}\n'

        for ref_column in ref_columns:
            res += f'      - {reference_df.at[i, ref_column]}\n'
    return res


def convert_vis_to_comparison_table(title, model_results, task, test):
    res = f'.. flat-table:: {title}\n'
    res += '    :header-rows: 1\n\n'

    # Header row
    model_aliases = [
        MODEL_ALIASES.get(model, model) for model in model_results.keys()
    ]
    res += '    * - '
    res += '\n      - '.join(model_aliases) + '\n'

    # Assume all models have the same number of visualizations, in the same
    # order
    for i in range(len(list(model_results.values())[0])):
        res += '    * - '
        result_list = []
        for model in model_results:
            png_file = os.path.basename(model_results[model][i])
            result_list.append(
                f'.. figure:: ../../../test/{task}/{model}/{test}/{png_file}\n'
            )
        res += '\n      - '.join(result_list) + '\n'
    return res


def create_model_overview(test_dir, condition, tests, model_id, executed_tests,
                          passed_tests):
    r"""Collect model CSV files and present on model page."""
    test_result_dir = os.path.join(TEST_RESULT_DIR, condition, model_id)
    model_info_file = 'model-info.csv'
    model_params_file = 'model-params.csv'
    model_tuning_file = 'model-tuning.csv'
    # Copy CSV files to docs folder
    for file in [
            model_info_file,
            model_params_file,
            model_tuning_file,
    ]:
        src = os.path.join(test_result_dir, file)
        dst = os.path.join(test_dir, file)
        if os.path.exists(src):
            shutil.copyfile(src, dst)
    index_file = os.path.join(test_dir, 'index.rst')
    model_passed_tests = sum(passed_tests[model_id].values())
    model_executed_tests = sum(executed_tests[model_id].values())
    score = (
            model_passed_tests
            / model_executed_tests
    )
    with open(index_file, 'w') as fp:
        fp.write(f'.. _test-{condition}-{model_id}:\n')
        fp.write('\n')
        fp.write(f'{MODEL_ALIASES.get(model_id, model_id)}\n')
        fp.write(f'{"=" * len(MODEL_ALIASES.get(model_id, model_id))}\n')
        fp.write('\n')
        fp.write('.. role:: red\n')
        fp.write('.. role:: green\n')
        fp.write('\n')
        fp.write(f'{100 * score:.1f}% passed tests ')
        fp.write(f'({model_passed_tests} :green:`passed` / ')
        fp.write(
            f'{model_executed_tests - model_passed_tests} :red:`failed`).\n'
        )
        fp.write('\n')
        fp.write('.. csv-table:: Tests overview\n')
        fp.write('    :header-rows: 1\n')
        fp.write('    :file: tests_overview.csv\n')
        fp.write('\n')
        if os.path.exists(os.path.join(test_dir, model_tuning_file)):
            fp.write('.. csv-table:: Model tuning parameters\n')
            fp.write('    :header-rows: 1\n')
            fp.write(f'    :file: {model_tuning_file}\n')
            fp.write('\n')
        fp.write('.. csv-table:: Model parameters\n')
        fp.write('    :header-rows: 1\n')
        fp.write(f'    :file: {model_params_file}\n')
        fp.write('\n')
        fp.write('.. csv-table:: Model information\n')
        fp.write('    :header-rows: 1\n')
        fp.write(f'    :file: {model_info_file}\n')
        fp.write('\n')
        fp.write('.. toctree::\n')
        fp.write('    :hidden:\n')
        fp.write('\n')
        for test in tests:
            fp.write(f'    {test}\n')


def create_results(html_theme_options):
    r"""Create RST files for results pages.

    The detailed result pages will show
    tables and if available figures
    for each model
    and each test.

    In addition,
    a landing page is created for each condition
    that displayes a ranking of the models.

    Args:
        html_theme_options: html theme options for sphinx to add wide pages to

    """
    test_score = {}
    for condition in CONDITIONS:
        executed_tests = defaultdict(Counter)
        passed_tests = defaultdict(Counter)
        test_score[condition] = {}
        condition_result_dir = os.path.join(TEST_RESULT_DIR, condition)
        index_file = os.path.join(CURRENT_DIR, 'test', f'{condition}.rst')
        with open(index_file, 'w') as fp:
            fp.write(f'.. _test-{condition}:\n')
            fp.write('\n')
            fp.write(f'{display_name(condition)}\n')
            fp.write(f'{"=" * len(condition)}\n')
            fp.write('\n')
            fp.write('.. csv-table:: Ranking of tested models\n')
            fp.write('    :header-rows: 1\n')
            fp.write('    :widths: 10, 75, 15\n')
            fp.write(f'    :file: {condition}.csv\n')
            # Skip if we don't have results
            if not os.path.exists(condition_result_dir):
                continue
            model_result_dirs = audeer.list_dir_names(condition_result_dir)
            # Sort model results by alias name
            model_result_dirs = sorted(
                model_result_dirs,
                key=lambda model_result_dir: MODEL_ALIASES.get(
                    os.path.basename(model_result_dir), os.path.basename(model_result_dir))
            )
            if model_result_dirs:
                fp.write('\n')
                fp.write('.. toctree::\n')
                fp.write('    :hidden:\n')
                fp.write('\n')
                for model_result_dir in model_result_dirs:
                    model_id = os.path.basename(model_result_dir)
                    create_test_results(
                        model_result_dir,
                        condition,
                        model_id,
                        executed_tests,
                        passed_tests
                    )
                    fp.write(f'    {condition}/{model_id}/index\n')
                    total_passed_tests = sum(passed_tests[model_id].values())
                    total_executed_tests = sum(
                        executed_tests[model_id].values()
                    )
                    test_score[condition][model_id] = (
                        total_passed_tests / total_executed_tests
                    )

        # Comparison of results
        index_file = os.path.join(CURRENT_DIR, 'comparison',
                                  f'{condition}.rst')
        with open(index_file, 'w') as fp:
            fp.write(f'.. _comparison-{condition}:\n')
            fp.write('\n')
            fp.write(f'{display_name(condition)}\n')
            fp.write(f'{"=" * len(condition)}\n')
            fp.write('\n')
            # Load which comparisons to perform
            yaml_file = os.path.join(
                COMPARISON_RESULT_DIR,
                f'{condition}.yaml'
            )
            # Skip if there should be no comparisons for this condition
            if not os.path.exists(yaml_file):
                continue
            with open(yaml_file, 'r') as comparison_fp:
                condition_comparisons = yaml.safe_load(comparison_fp)

            fp.write('\n')
            fp.write('.. toctree::\n')
            fp.write('    :hidden:\n')
            fp.write('\n')
            for comparison in condition_comparisons:
                models = [comparison['baseline']] + \
                    comparison['candidates']
                comparison_id = '_'.join(models)
                create_comparison_results(
                    {model_id: os.path.join(
                        CURRENT_DIR, 'test', condition, model_id)
                        for model_id in models},
                    condition,
                    html_theme_options,
                    executed_tests,
                    passed_tests
                )
                fp.write(
                    f'    {condition}/{comparison_id}/index\n'
                )
    rank_models(test_score)


def create_test_results(
    src_dir: str,
    condition: str,
    model_id: str,
    executed_tests: defaultdict,
    passed_tests: defaultdict
):
    r"""Create test overview and result pages per model.

    Args:
        src_dir: folder where the results of the tests are stored
        condition: test condition, e.g. ``'arousal'``
        model_id: model ID
        executed_tests: Dict mapping each model to the test-wise executed test
            counter
        passed_tests: Dict mapping each model to the test-wise passed test
            counter
    """
    # Get all tests and corresponding CSV files
    tests = audeer.list_dir_names(src_dir, basenames=True)
    csv_files = [
        os.path.join(src_dir, test, f'{test.replace("test_", "")}.csv')
        for test in tests
    ]
    # First create an index file for each model ID
    # listing all the single test pages
    model_dir = audeer.mkdir(
        os.path.join(CURRENT_DIR, 'test', condition, model_id)
    )

    # Count test results per model,
    # besides global executed_tests and passed_tests
    per_model_tests = []

    for test in tests:
        # TODO: it might be that some metrics contain values
        # and images.
        # It would be nice to group them together during presentation.
        thresholds, descriptions = load_test_thresholds(condition, test)
        thresholds = {k: thresholds[k] for k in sorted(thresholds)}

        # Store CSV file with test and threshold overview.
        # This is used inside the Method Tests section
        path = os.path.join(CURRENT_DIR, 'method-tests', condition)
        audeer.mkdir(path)
        outfile = os.path.join(path, f'{test}.csv')
        with open(outfile, 'w') as fp:
            fp.write('Test,Threshold\n')
            for sub_test, (_, threshold) in thresholds.items():
                if isinstance(threshold, dict):
                    threshold = ', '.join(
                        [f'{k}: {v}' for k, v in threshold.items()]
                    )
                fp.write(f'{display_name(sub_test)},"{threshold}"\n')

        # Tables
        csv_files = audeer.list_file_names(
            os.path.join(src_dir, test),
            filetype='csv',
        )
        # Figures
        png_files = audeer.list_file_names(
            os.path.join(src_dir, test),
            filetype='png',
        )
        test_file = os.path.join(model_dir, f'{test}.rst')
        test_dir = audeer.mkdir(os.path.join(model_dir, test))
        heading = ' '.join(test.capitalize().split('_'))
        with open(test_file, 'w') as fp:
            fp.write(f'.. _test-{condition}-{model_id}-{test}:\n')
            fp.write('\n')
            fp.write(f'{heading}\n')
            fp.write(f'{"=" * len(heading)}\n')
            fp.write('\n')
            fp.write('.. role:: red\n')
            fp.write('.. role:: green\n')
            fp.write('\n')
            for csv_file in csv_files:
                title = display_name(csv_file)
                key = audeer.basename_wo_ext(csv_file)
                threshold = thresholds[key]
                description = descriptions[key]
                # Parse the CSV file and get
                # executed_tests and passed_tests counters
                n_executed, n_passed = parse_csv_file(
                    csv_file, test_dir, threshold
                )
                executed_tests[model_id][test] += n_executed
                passed_tests[model_id][test] += n_passed
                fp.write(
                    f'\n'
                    f'{title}\n'
                    f'{"-" * len(title)}\n'
                    f'\n'
                )
                if description:
                    fp.write(f'{description}\n\n')
                fp.write(
                    f'.. csv-table:: Threshold: {threshold[1]}\n'
                )
                fp.write('    :header-rows: 1\n')
                fp.write(f'    :file: {test}/{os.path.basename(csv_file)}\n')

            previous_title = ''
            for png_file in png_files:
                title = display_name(os.path.basename(png_file).split('_')[0])
                shutil.copyfile(
                    png_file,
                    os.path.join(test_dir, os.path.basename(png_file)),
                )
                # We have several PNG files per title
                if title != previous_title:
                    key = os.path.basename(png_file).split('_')[0]
                    description = descriptions[key]
                    fp.write(
                        '\n'
                        f'{title}\n'
                        f'{"-" * len(title)}\n'
                        f'\n'
                    )
                    previous_title = title
                    if description:
                        fp.write(f'{description}\n\n')
                fp.write(
                    f'\n'
                    f'.. figure:: {test}/{os.path.basename(png_file)}\n'
                    '    :align: center\n'
                )

        # Get combined results for the single test
        per_test_failed_tests = (executed_tests[model_id][test]
                                 - passed_tests[model_id][test])
        test_score = (passed_tests[model_id][test]
                      / executed_tests[model_id][test]
                      )

        # Add test overview at top of test page.
        # This adds text in the middle of an existing file,
        # see https://stackoverflow.com/a/10507291
        with open(test_file, 'r') as fp:
            contents = fp.readlines()
        text = (
            f'{100 * test_score:.1f}% passed tests '
            f'({passed_tests[model_id][test]} :green:`passed` / '
            f'{per_test_failed_tests} :red:`failed`).'
            '\n\n'
        )
        contents.insert(8, text)
        with open(test_file, 'w') as fp:
            contents = "".join(contents)
            fp.write(contents)

        test_link = f':ref:`{heading} <test-{condition}-{model_id}-{test}>`'
        test_result = f'{100 * test_score:.1f}%'
        if test_score == 1:
            test_result = f':green:`{test_result}`'
        elif test_score < 0.75:
            test_result = f':red:`{test_result}`'
        per_model_tests.append([test_link, test_result])

    # Create RST files for each test in a sub-folder
    create_model_overview(
        model_dir, condition, tests, model_id,
        executed_tests, passed_tests
    )

    df = pd.DataFrame(per_model_tests, columns=['Topic', 'Passed Tests'])
    outfile = os.path.join(model_dir, 'tests_overview.csv')
    df.to_csv(outfile, index=False)

    # Store additional files for certain tests to use inside the Method Tests
    # section
    # Emotion Ranges used in correctness_consistency
    dimensions = ['arousal', 'dominance', 'valence']
    emotions = [
        set(
            CATEGORY_LOW[dimension] +
            CATEGORY_NEUTRAL[dimension] +
            CATEGORY_HIGH[dimension]
        ) for dimension in dimensions
    ]
    emotions = set.union(*emotions)
    emo_ranges = {}
    for emotion in emotions:
        emo_ranges[emotion] = {}
        for dimension in dimensions:
            if emotion in CATEGORY_LOW[dimension]:
                emo_ranges[emotion][dimension] = 'low'
            elif emotion in CATEGORY_NEUTRAL[dimension]:
                emo_ranges[emotion][dimension] = 'neutral'
            elif emotion in CATEGORY_HIGH[dimension]:
                emo_ranges[emotion][dimension] = 'high'
    df = pd.DataFrame.from_dict(emo_ranges, orient='index')
    df.sort_index(inplace=True)
    df.to_csv(
        os.path.join(CURRENT_DIR, 'method-tests',
                     'correctness_consistency_ranges.csv')
    )


def create_comparison_results(
        model_result_dirs: typing.Dict[str, str],
        condition: str,
        html_theme_options: typing.Dict[str, typing.Any],
        executed_tests: defaultdict,
        passed_tests: defaultdict
):
    r"""Create comparison overview and result pages per model comparison

    Args:
        model_result_dirs: dict mapping from model id to
            result folder
        condition: test condition, e.g. ``'arousal'``
        html_theme_options: html theme options for sphinx to add wide pages to
        executed_tests: Dict mapping each model to the test-wise executed test
            counter
        passed_tests: Dict mapping each model to the test-wise passed test
            counter
    """

    baseline_model_id = list(model_result_dirs.keys())[0]
    candidates = list(model_result_dirs.keys())[1:]
    comparison_id = '_'.join(model_result_dirs.keys())
    alias_comparison_id = '_'.join(
        [MODEL_ALIASES.get(model, model) for model in model_result_dirs.keys()])
    comparison_dir = audeer.mkdir(os.path.join(
        CURRENT_DIR, 'comparison', condition, comparison_id))
    model_aliases = [MODEL_ALIASES.get(model, model) for model in model_result_dirs.keys()]

    # Comparison of Test Results
    baseline_result_dir = model_result_dirs[baseline_model_id]
    tests = audeer.list_dir_names(baseline_result_dir, basenames=True)
    for test in tests:
        comparison_test_dir = audeer.mkdir(os.path.join(
            comparison_dir, test
        ))
        thresholds, descriptions = load_test_thresholds(condition, test)
        thresholds = {k: thresholds[k] for k in sorted(thresholds)}
        baseline_csv_files = audeer.list_file_names(
            os.path.join(baseline_result_dir, test),
            filetype='csv',
        )
        baseline_png_files = audeer.list_file_names(
            os.path.join(baseline_result_dir, test),
            filetype='png',
        )
        test_file = os.path.join(comparison_dir, f'{test}.rst')
        # Add to wide pages to support long comparison tables
        html_theme_options['wide_pages'].append(
            f'comparison/{condition}/{comparison_id}/{test}'
        )

        heading = ' '.join(test.capitalize().split('_'))
        with open(test_file, 'w') as fp:
            fp.write(f'.. _comparison-{condition}-{comparison_id}-{test}:\n')
            fp.write('\n')
            fp.write(f'{heading}\n')
            fp.write(f'{"=" * len(heading)}\n')
            fp.write('\n')
            fp.write('.. role:: red\n')
            fp.write('.. role:: green\n')
            fp.write('\n')
            # Write overall test results to table
            overall_scores = []
            for model_id in model_result_dirs.keys():
                n_passed = passed_tests[model_id][test]
                n_executed = executed_tests[model_id][test]
                score = n_passed / n_executed
                score_text = f'{100 * score:.1f}% passed tests ({n_passed} ' \
                             f':green:`passed` /  {n_executed - n_passed} ' \
                             f':red:`failed`).\n'
                overall_scores.append(score_text)
            overall_table = pd.DataFrame.from_dict(
                {'Overall Score': overall_scores},
                orient='index',
                columns=model_aliases
            )
            overall_table.to_csv(
                os.path.join(comparison_test_dir, 'overall_scores.csv')
            )
            fp.write('.. csv-table:: Overall scores\n')
            fp.write('    :header-rows: 1\n')
            fp.write(f'    :file: {test}/overall_scores.csv\n')
            fp.write('\n')
            for csv_file in baseline_csv_files:
                title = display_name(csv_file)
                key = audeer.basename_wo_ext(csv_file)
                threshold = thresholds[key]
                description = descriptions[key]
                model_csv_files = {
                    model_id: os.path.join(
                        test_dir, test, os.path.basename(csv_file)
                    )
                    for model_id, test_dir in model_result_dirs.items()
                }
                test_table = convert_csvs_to_comparison_table(
                    f'Threshold: {threshold[1]}', model_csv_files
                )
                fp.write(
                    f'\n'
                    f'{title}\n'
                    f'{"-" * len(title)}\n'
                    f'\n'
                )
                if description:
                    fp.write(f'{description}\n\n')
                fp.write(test_table)

            title = display_name(
                os.path.basename(baseline_png_files[0]).split('_')[0]
            )
            fp.write(
                '\n'
                f'{title}\n'
                f'{"-" * len(title)}\n'
                f'\n'
            )
            model_png_files = {
                model_id: [
                    os.path.join(model_dir, test, os.path.basename(png_file))
                    for png_file in baseline_png_files
                ]
                for model_id, model_dir in model_result_dirs.items()
            }
            key = os.path.basename(baseline_png_files[0]).split('_')[0]
            description = descriptions[key]
            if description:
                fp.write(f'{description}\n\n')
            test_table = convert_vis_to_comparison_table(
                '', model_png_files, condition, test)
            fp.write(test_table)

    # Index page with test overview table

    passed_table = convert_csvs_to_comparison_table(
        'Tests overview',
        {model_id: os.path.join(result_dir, 'tests_overview.csv') for
         model_id, result_dir in model_result_dirs.items()}
    )
    # Change references in index of table, which currently link to baseline
    # model results, so that they link to comparison pages
    passed_table = passed_table.replace(
        f'<test-{condition}-{baseline_model_id}',
        f'<comparison-{condition}-{comparison_id}'
    )
    # Add overall scores to the top row of the table
    overall_scores = []
    for model_id in model_result_dirs.keys():
        n_passed = sum(passed_tests[model_id].values())
        n_executed = sum(executed_tests[model_id].values())
        score = n_passed / n_executed
        score_text = f'{100 * score:.1f}% ({n_passed} ' \
                     f':green:`passed` /  {n_executed - n_passed} ' \
                     f':red:`failed`)\n'
        overall_scores.append(score_text)
    passed_table_rows = passed_table.split('\n')
    overview_row = (
        '    * - Overall Score\n      - ' +
        '      - '.join(overall_scores)
    )
    # insert overall score row after the table directive, header row option,
    # a new line, the first header row's two columns,
    # and the second header row's columns (number equal to number of models
    insert_pos = 5 + len(overall_scores)
    passed_table = (
        '\n'.join(passed_table_rows[:insert_pos]) + '\n' + overview_row
        + '\n'.join(passed_table_rows[insert_pos:]) + '\n'
    )

    index_file = os.path.join(comparison_dir, 'index.rst')
    comparison_aliases = [MODEL_ALIASES.get(model_id, model_id) for model_id in model_result_dirs.keys()]
    comparison = ' vs. '.join(comparison_aliases)
    with open(index_file, 'w') as fp:
        fp.write(f'.. _comparison-{condition}-{comparison_id}:\n')
        fp.write('\n')
        fp.write(f'{comparison}\n')
        fp.write(f'{"=" * len(comparison)}\n')
        fp.write('\n')
        fp.write('.. role:: red\n')
        fp.write('.. role:: green\n')
        fp.write('\n')

        model_links = [
            f':ref:`{MODEL_ALIASES.get(model, model)} <test-{condition}-{model}>`'
            for model in [baseline_model_id]+candidates
        ]
        model_str = ' and '.join(model_links)

        fp.write('This compares the models ')
        fp.write(f'{model_str} to one another.\n')
        fp.write('\n')

        # Test overview
        fp.write(passed_table)
        fp.write('\n')
        fp.write('.. toctree::\n')
        fp.write('    :hidden:\n')
        fp.write('\n')
        for test in tests:
            fp.write(f'    {test}\n')
    # TODO: Add some plots for the model comparison


def display_name(name):
    r"""E.g. 'name1-mane2' => 'Name1 Name2'."""
    name = audeer.basename_wo_ext(name)
    return ' '.join(
        [part.capitalize() for part in name.split('-')]
    )


def display_result(x, passed, annotation=None):
    r"""Format the result presented in the tables.

    This selects the number of digits to show
    and marks the test as passed or failed by color.

    """
    if passed is None:
        # Skip annotation if None
        return ''
    elif passed:
        result = f':green:`{x:.2f}`'
    else:
        result = f':red:`{x:.2f}`'

    # Add annotation after result if provided
    if annotation is not None:
        # Round all floating point numbers
        # contained in annotation
        fp_pattern = (
            r'[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?'
        )
        rounded_annotation = re.sub(fp_pattern, lambda x: f'{float(x.group(0)):.2f}', annotation)
        result += f' {rounded_annotation}'
    return result


def load_test_thresholds(
        condition: str,
        test: str,
) -> typing.Tuple[typing.Dict, typing.Dict]:
    r"""Load test thresholds as defined in the tests.

    This loads the thresholds, comparisons, and descriptions
    from the configuration YAML files.

    It requires that every test
    implements a :func:`load_config`
    under :mod:`common.{test_name}`.

    Args:
        condition: test condition, e.g. ``'arousal'``
        test: name of test,
            e.g. 'fairness_sex'

    Returns:
        * dictionary with metric name as key and (comparison, threshold)
          as value
        * dictionary with metric as key and description as value

    """
    module = importlib.import_module(f'common.{test}')
    load_config = getattr(module, 'load_config')
    _, metrics = load_config(condition)
    thresholds = {}
    for metric in metrics:
        if 'threshold' not in metric or 'comparison' not in metric:
            continue
        if 'description' in metric:
            description = metric['description']
        else:
            description = ''
        key = metric['name'].lower().replace(' ', '-')
        thresholds[key] = (metric['comparison'], metric['threshold'])
    descriptions = {}
    for metric in metrics:
        if 'description' in metric:
            description = metric['description']
        else:
            description = ''
        key = metric['name'].lower().replace(' ', '-')
        descriptions[key] = description
    return thresholds, descriptions


def parse_csv_file(csv_file: str, result_dir: str, threshold: typing.Tuple) \
        -> typing.Tuple[int, int]:
    r"""Parse result tables and store as new file."""
    executed_test_counter = 0
    passed_test_counter = 0
    comparison, threshold = threshold
    df = pd.read_csv(csv_file, index_col='Data')
    if isinstance(threshold, dict):
        # Calculate mean over all tests before adding text to columns
        mean = df.mean()
        for key, value in threshold.items():
            passed = df[key].apply(lambda x: comparison(x, value))
            passed = passed.rename('passed')
            df[key] = pd.concat((df[key], passed), axis=1).apply(
                lambda x: display_result(x[key], x['passed']),
                axis=1
            )
            executed_test_counter += len(passed.dropna())
            passed_test_counter += passed.sum()
    else:
        test_columns = [col for col in df.columns if 'Reference' not in col and col[0]!= '_']
        reference_columns = [col for col in df.columns if 'Reference' in col]
        annotation_columns = [col for col in df.columns if col[0] == '_']
        # Calculate mean over all tests before adding text to columns
        # Don't include annotation columns in mean calculation
        mean = df.drop(columns=annotation_columns).mean()
        for test_column in test_columns:
            passed = df[test_column].apply(lambda x: comparison(x, threshold))
            passed = passed.rename('passed')
            df[test_column] = pd.concat(
                (df, passed), axis=1).apply(
                lambda x: display_result(
                    x[test_column],
                    x['passed'],
                    annotation=(
                        x[f'_{test_column}'] if f'_{test_column}' in annotation_columns else None
                    )
                ),
                axis=1
            )
            executed_test_counter += len(passed.dropna())
            passed_test_counter += passed.sum()
        df[reference_columns] = df[reference_columns].applymap(
            lambda x: f'{x:.2f}'
        )

        # Drop annotation columns, they are now displayed in the test_column
        df.drop(columns=annotation_columns, inplace=True)

    # Assign mean at the end to not count it as a test.
    # Using explicit string conversion as `mean.round(2)`
    # would return `0.2` instead of `0.20`
    df.loc['mean'] = [f'{m:.2f}' if not pd.isna(m) else '' for m in mean]
    out_file = os.path.join(result_dir, os.path.basename(csv_file))
    df.to_csv(out_file)
    return executed_test_counter, passed_test_counter


def rank_models(test_score):
    r"""Store CSV files with winning model overview."""
    for condition in test_score:
        index_column = 'Model ID'
        result_column = 'Passed Tests'
        df = pd.DataFrame.from_dict(
            test_score[condition],
            orient='index',
            columns=[result_column],
        )
        df.index.name = index_column
        if df.empty:
            # Ensure we at least one entry in CSV file
            df = pd.DataFrame('', index=[''], columns=[result_column])
        df = df.sort_values(result_column, axis=0, ascending=False)
        df = df.reset_index()
        df.index.name = 'Position'
        df[result_column] = df[result_column].map(
            lambda x: f'{100 * x:.1f}%'
        )
        df[index_column] = df[index_column].map(
            lambda x: f':ref:`{MODEL_ALIASES.get(x,x)} <test-{condition}-{x}>`'
        )
        df.to_csv(os.path.join(CURRENT_DIR, 'test', f'{condition}.csv'))


if __name__ == '__main__':
    main(html_theme_options={'wide_pages': []})
