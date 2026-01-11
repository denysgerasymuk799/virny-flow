# pylint: disable=missing-docstring
"""Davos default backend tests."""

import os
import tempfile

import numpy as np

from alpine_meadow.backend import DefaultBackend
from alpine_meadow.common import PerformanceMetric, Config, Dataset


def test_backend(df_185, pipeline_185, df_185_path):
    # set up backend
    config = Config()
    backend = DefaultBackend(config)

    # preprocess data
    df_185['Hall_of_Fame'] = df_185['Hall_of_Fame'].astype(np.int64)

    # get pipeline executor
    metrics = [PerformanceMetric.Value('ACCURACY'),
               PerformanceMetric.Value('F1_MACRO')]
    pipeline_executor = backend.get_pipeline_executor(pipeline_185, metrics)

    # train
    dataset = Dataset(df_185)
    train_result = pipeline_executor.train([dataset])
    assert len(train_result.outputs) == len(df_185)

    # test
    test_result = pipeline_executor.test([dataset])
    assert len(test_result.outputs) == len(df_185)

    # score
    targets = [df_185['Hall_of_Fame']]
    score_result = pipeline_executor.score([dataset], targets)
    scores = score_result.scores
    assert len(scores) == 2
    assert scores[0] >= 0.9
    assert scores[1] >= 0.7

    # export
    with tempfile.NamedTemporaryFile(suffix='.py') as tf:
        with open(tf.name, 'w') as f:
            f.write(pipeline_executor.export())
        ret = os.system(f'python {tf.name} {df_185_path}')
        assert ret == 0
