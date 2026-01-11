# pylint: disable=missing-docstring
"""Tests for metadata management."""

from alpine_meadow.common import Config, PerformanceMetric, Task, TaskKeyword
from alpine_meadow.core.meta_learning import MetaLearningManager, PipelineHistory
from alpine_meadow.core.rule import RuleExecutor
from alpine_meadow.core import Optimizer


def test_meta_learning(df_185):
    # meta learning manager
    config = Config()
    metadata_manager = MetaLearningManager(config)

    # make task
    df = df_185
    metric = PerformanceMetric.Value('F1_MACRO')
    target_column = "Hall_of_Fame"
    task = Task([TaskKeyword.Value('CLASSIFICATION')], [metric], [target_column], df)
    task.compute_meta_features()
    task.config = config

    assert metadata_manager.num_task_traces == 0
    assert metadata_manager.num_pipeline_runs == 0

    # get similar tasks
    similar_tasks = metadata_manager.register_task(task, config)
    assert similar_tasks

    # check similar tasks
    assert metadata_manager.get_num_similar_tasks(task) > 0
    assert metadata_manager.get_num_reference_pipelines(task) > 0

    # get pipeline arms
    rule_executor = RuleExecutor(config)
    pipeline_history = PipelineHistory()
    search_space = rule_executor.execute(task, pipeline_history)
    pipeline_arms = search_space.pipeline_arms
    assert pipeline_arms

    # update pipeline arms
    metadata_manager.update_pipeline_arms_from_history(task, pipeline_arms)
    external_num = 0
    for pipeline_arm in pipeline_arms:
        if pipeline_arm.external_quality_mean is not None and pipeline_arm.external_quality_std is not None:
            external_num += 1
    assert external_num > 100

    # check meta learning info
    assert metadata_manager.num_task_traces == 298
    assert metadata_manager.num_pipeline_runs == 102340

    meta_learning_info = task.meta_learning_info
    assert meta_learning_info['num_similar_tasks'] > 0
    assert meta_learning_info['num_updated_runs'] > 0


def test_task_trace(df_185):
    # make task
    df = df_185
    metric = PerformanceMetric.Value('F1_MACRO')
    target_column = "Hall_of_Fame"
    task = Task([TaskKeyword.Value('CLASSIFICATION')], [metric], [target_column], df)

    # run optimizer
    config = Config(log_trace=True, timeout_seconds=5)
    optimizer = Optimizer(task, config=config)
    for result in optimizer.optimize():
        print(f'Time: {result.elapsed_time}, Score: {result.score}')

    # check dump
    dump = optimizer.dumps()
    assert 'task' in dump
    assert 'pipelines' in dump
    assert 'pipeline_runs' in dump
    assert isinstance(dump['pipelines'], list)
    assert isinstance(dump['pipeline_runs'], list)
