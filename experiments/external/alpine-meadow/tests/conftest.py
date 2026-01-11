# pylint: disable=redefined-outer-name
"""Alpine Meadow tests configurations and fixtures."""

import os
import json

import pytest

from alpine_meadow.common import Pipeline


@pytest.fixture(scope='session')
def df_185_path():
    return os.path.join(os.path.abspath(os.path.dirname(__file__)), "datasets", "d3m",
                        "baseball", "tables", "learningData.csv")


@pytest.fixture(scope='session')
def df_185(df_185_path):
    import pandas as pd

    df = pd.read_csv(df_185_path)
    return df


@pytest.fixture(scope='session')
def df_196():
    import pandas as pd

    df = pd.read_csv(os.path.join(os.path.abspath(os.path.dirname(__file__)), "datasets", "d3m",
                                  "autoMpg", "tables", "learningData.csv"))
    return df


@pytest.fixture(scope='session')
def pipeline_185():
    with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), "files", "185_pipeline.json")) as f:
        pipeline_json = json.load(f)
        pipeline_str = json.dumps(pipeline_json['pipeline'])
    pipeline_desc = Pipeline.from_pipeline_desc(pipeline_str, is_json=True, human_readable=True)
    return pipeline_desc
