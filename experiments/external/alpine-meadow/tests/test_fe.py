"""Tests for feature engineering. Not unit tests, but for integration testing."""
import os
import json

import pandas as pd

from alpine_meadow.common import PerformanceMetric
from alpine_meadow.facade import AMClassifier, AMRegressor
from alpine_meadow.common.metric import get_score


def fe_facade(dataset, fe_enabled=True, time_budget=60, fe_hyperparams=None,
              fe_budget_proportion=0.1, base_dir=None, problem_type=None):
    """Run facade test for feature engineering for a single dataset."""

    data_dir = "/home/fe_data/data/" if base_dir is None else base_dir
    df = pd.read_csv(data_dir + dataset + "/" + dataset + "_dataset/tables/learningData.csv")
    info_path = data_dir + dataset + "/" + dataset + "_dataset/datasetDoc.json"
    with open(info_path, encoding="utf-8", errors="replace") as f:
        problem_info = json.load(f)
        problem_columns = problem_info["dataResources"][0]["columns"]
    target_column = None
    for c in problem_columns:
        if c["role"][0] == "suggestedTarget":
            target_column = c["colName"]
            break
    assert target_column is not None

    splits_df = pd.read_csv(data_dir + dataset + "/" + dataset + "_problem/dataSplits.csv")
    train_x = df[splits_df["type"] == "TRAIN"].drop(target_column, axis=1)
    train_y = df[splits_df["type"] == "TRAIN"][target_column]
    test_x = df[splits_df["type"] == "TEST"].drop(target_column, axis=1)
    test_y = df[splits_df["type"] == "TEST"][target_column]

    if problem_type == "C":
        model = AMClassifier(fe_enabled=fe_enabled, fe_hyperparams=fe_hyperparams,
                             fe_budget_proportion=fe_budget_proportion, timeout_seconds=time_budget)
        metric = 'F1_MACRO'
    elif problem_type == "R":
        model = AMRegressor(fe_enabled=fe_enabled, fe_hyperparams=fe_hyperparams,
                            fe_budget_proportion=fe_budget_proportion, timeout_seconds=time_budget)
        metric = 'MEAN_SQUARED_ERROR'
    else:
        raise ValueError(f"Invalid problem type given: {problem_type}")

    model.fit(train_x, train_y)

    pred_y = model.predict(test_x)
    score = get_score(PerformanceMetric.Value(metric), test_y, pred_y)
    print(f'Score: {score}')
    return score


def run_tests(fe_enabled=False):
    """Run tests with fe either enabled or disabled on all datasets in the default directory."""

    import csv
    # data_split_dir = "/home/fe_data/dataSplits/"
    # test_data_dir = "/home/fe_data/data/"
    data_split_dir = "/Users/wesleyrunnels//MIT/MEng/fe_data/data/dataSplits/"
    test_data_dir = "/Users/wesleyrunnels/MIT/MEng/fe_data/data/data/"
    test_datasets = {}
    with open(os.path.join(data_split_dir, "test_datasets.csv")) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            test_datasets[row["Dataset"]] = row["TaskType"]

    present_datasets = [d for d in os.listdir(test_data_dir) if
                        (os.path.isdir(os.path.join(test_data_dir, d)) and d in test_datasets)]

    ids = {}
    for dataset in present_datasets:
        ids[dataset] = test_datasets[dataset][0]

    fe_hyperparams = {"verbose": True, "auto_fs": True, "algo_setting": "rf", "n_cores": 1}
    fe_budget_prop = 0.5
    total_budget = 60
    scores = {}
    for id_, problem_type in ids.items():
        scores[id_] = (fe_facade(dataset=id_, fe_hyperparams=fe_hyperparams, time_budget=total_budget,
                                 fe_budget_proportion=fe_budget_prop, fe_enabled=fe_enabled,
                                 base_dir="/Users/wesleyrunnels/MIT/MEng/fe_data/data/data/",
                                 problem_type=problem_type), problem_type)
        print(scores)


def combo_tests():
    run_tests(True)


def am_tests():
    run_tests(False)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "am":
        am_tests()
    else:
        combo_tests()
