# pylint: disable=missing-docstring
"""Tests for search space and rules."""

from alpine_meadow.common import Config, PerformanceMetric, Task, TaskKeyword
from alpine_meadow.core.meta_learning import PipelineHistory
from alpine_meadow.core.rule import RuleExecutor
from alpine_meadow.core import Optimizer


def test_search_space_classification(df_185):
    # make task
    config = Config()
    df = df_185
    metric = PerformanceMetric.Value('F1_MACRO')
    target_column = "Hall_of_Fame"
    task = Task([TaskKeyword.Value('CLASSIFICATION')], [metric], [target_column], df)
    task.compute_meta_features()
    task.config = config

    # get pipeline arms
    rule_executor = RuleExecutor(config)
    pipeline_history = PipelineHistory()
    search_space = rule_executor.execute(task, pipeline_history)
    pipeline_arms = search_space.pipeline_arms
    assert len(pipeline_arms) == 550
    assert len(search_space.unique_primitives) == 32

    # check pipeline arms
    for rule in rule_executor._enforcement_rules:  # pylint: disable=protected-access
        if rule.predicate(task):
            for pipeline_arm in pipeline_arms:
                assert rule.enforce(task, pipeline_arm)


def test_search_space_regression(df_196):
    # make task
    config = Config()
    df = df_196
    metric = PerformanceMetric.Value('MEAN_SQUARED_ERROR')
    target_column = "class"
    task = Task([TaskKeyword.Value('REGRESSION')], [metric], [target_column], df)
    task.compute_meta_features()
    task.config = config

    # get pipeline arms
    rule_executor = RuleExecutor(config)
    pipeline_history = PipelineHistory()
    search_space = rule_executor.execute(task, pipeline_history)
    pipeline_arms = search_space.pipeline_arms
    assert len(pipeline_arms) == 300
    assert len(search_space.unique_primitives) == 30

    # check pipeline arms
    for rule in rule_executor._enforcement_rules:  # pylint: disable=protected-access
        if rule.predicate(task):
            for pipeline_arm in pipeline_arms:
                assert rule.enforce(task, pipeline_arm)


def test_pipeline_generation(df_185):
    # make task
    df = df_185
    metrics = [PerformanceMetric.Value('F1_MACRO'), PerformanceMetric.Value('ACCURACY')]
    target_column = "Hall_of_Fame"
    task = Task([TaskKeyword.Value('CLASSIFICATION')], metrics, [target_column], df)

    # run optimizer
    config = Config(timeout_seconds=5)
    optimizer = Optimizer(task, config=config)
    for result in optimizer.optimize():
        print(f'Time: {result.elapsed_time}, Score: {result.score}')

    # select pipeline arm
    selected_pipeline_arm = None
    for pipeline_arm in optimizer.search_space.pipeline_arms:
        pipeline_arm.compute_metrics()
        if pipeline_arm.quality_mean is not None:
            selected_pipeline_arm = pipeline_arm
            break
    assert selected_pipeline_arm

    # get pipelines
    pipelines = selected_pipeline_arm.get_next_pipelines(use_bayesian_optimization=True, pipelines_num=10)
    assert len(pipelines) == 10

    # check search space
    assert optimizer.search_space.num_searched_pipelines > 0
