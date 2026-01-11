# pylint: disable=missing-docstring
"""Tests for tasks."""

from alpine_meadow.common import Task, TaskKeyword, PerformanceMetric


def test_naive_sore(df_185):
    # make task
    df = df_185
    metrics = [PerformanceMetric.Value('PRECISION'), PerformanceMetric.Value('ACCURACY')]
    target_column = "Hall_of_Fame"
    task = Task([TaskKeyword.Value('CLASSIFICATION')], metrics, [target_column], df,
                pos_label='0')
    naive_score = task.compute_naive_score()
    assert abs(naive_score - 0.9068) < 1e-4
