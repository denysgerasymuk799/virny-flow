# pylint: disable=missing-docstring, invalid-name
"""Tests for facade classes."""
from alpine_meadow.common import PerformanceMetric
from alpine_meadow.facade import AMClassifier
from alpine_meadow.common.metric import get_score


def test_facade(df_185):
    # prepare X and y
    df = df_185
    X = df.drop(['Hall_of_Fame'], axis=1)
    y = df[['Hall_of_Fame']]

    # fit
    classifier = AMClassifier()
    classifier.fit(X, y)

    # predict
    yy = classifier.predict(X)
    score = get_score(PerformanceMetric.Value('F1_MACRO'), y.values, yy.values)
    print(f'Score: {score}')
