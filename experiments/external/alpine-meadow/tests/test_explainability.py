# pylint: disable=missing-docstring
"""Test cases for explainability in Alpine Meadow."""

import time

import pytest

from alpine_meadow.common import Task, TaskKeyword, PerformanceMetric, Config
from alpine_meadow.core import Optimizer
from alpine_meadow.core.optimization.explainability import explain_by_shap


@pytest.mark.skip(reason="SHAP is not compatible with Xgboost")
def test_shap_tree_models(df_185):
    # make task
    df = df_185
    metric = PerformanceMetric.Value('F1_MACRO')
    target_column = "Hall_of_Fame"
    task = Task([TaskKeyword.Value('CLASSIFICATION')], [metric], [target_column], df)

    # run optimizer
    config = Config(explainable_feature_processing=True,
                    only_tree_models=True)
    optimizer = Optimizer(task, config=config)
    result = None
    for result in optimizer.optimize():
        print(f'Time: {result.elapsed_time}, Score: {result.score}')

        # explain
        model = result.pipeline.primitives[-1].primitive
        print(f'Explaining {model}')

        start = time.perf_counter()
        explain_by_shap(task, result.pipeline, df)
        print(f'Explaining time: {time.perf_counter() - start}')
