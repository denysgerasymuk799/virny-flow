"""Evaluation utility functions."""

import traceback

import numpy as np
import pandas as pd

from alpine_meadow.common import PerformanceMetric
from alpine_meadow.common.metric import get_score


CLASSIFICATION_METRICS = (
    PerformanceMetric.ACCURACY, PerformanceMetric.PRECISION,
    PerformanceMetric.RECALL, PerformanceMetric.F1,
    PerformanceMetric.F1_MICRO, PerformanceMetric.F1_MACRO,
    PerformanceMetric.ROC_AUC, PerformanceMetric.ROC_AUC_MICRO,
    PerformanceMetric.ROC_AUC_MACRO, PerformanceMetric.COHEN_KAPPA,
    PerformanceMetric.SUPPORT
)

REGRESSION_METRICS = (
    PerformanceMetric.MEAN_SQUARED_ERROR, PerformanceMetric.ROOT_MEAN_SQUARED_ERROR,
    PerformanceMetric.MEAN_ABSOLUTE_ERROR, PerformanceMetric.R_SQUARED
)


def compute_metrics(to_be_computed_metrics, y_true, y_pred, **scoring_kwargs):
    metrics = {}
    for metric in to_be_computed_metrics:
        try:
            score = get_score(metric, y_true, y_pred, **scoring_kwargs)
            if not np.isnan(score):
                metrics[metric] = score
        except:  # pylint: disable=bare-except # noqa: E722
            traceback.print_exc()
    return metrics


def compute_classification_metrics(y_true, y_pred, y_pred_proba=None, classes=None, **scoring_kwargs):
    """
    Compute classification metrics given predications and ground truth.
    """

    from sklearn.metrics import classification_report

    to_be_computed_metrics = (PerformanceMetric.ACCURACY, PerformanceMetric.F1_MICRO,
                              PerformanceMetric.F1_MACRO, PerformanceMetric.COHEN_KAPPA)
    metrics = compute_metrics(to_be_computed_metrics, y_true, y_pred)

    # classification report (accuracy, precision, recall, F1 and support)
    classification_report = classification_report(y_true, y_pred, output_dict=True)
    pos_label = scoring_kwargs.get('pos_label', None)
    for class_, class_report in classification_report.items():
        if pos_label is not None and str(class_) != str(pos_label):
            continue
        if not isinstance(class_report, dict):
            continue
        for key in class_report:
            metric_name = key.upper()
            # change F1-SCORE to F1 to be consistent with Protobuf
            if metric_name == 'F1-SCORE':
                metric_name = 'F1'

            metric = PerformanceMetric.Value(metric_name)
            if metric not in metrics:
                metrics[metric] = {}
            if pos_label is not None:
                metrics[metric] = class_report[key]
            else:
                metrics[metric][class_] = class_report[key]

    # prepare data for ROC_AUC
    if y_pred_proba is not None:
        try:
            if len(y_pred_proba.values.shape) > 1 and y_pred_proba.values.shape[1] > 1:
                y_true_proba = pd.DataFrame()
                for label in classes:
                    y_true_proba[label] = np.array(y_true.values == label).flatten()
            else:
                y_true_proba = y_true

            for metric in [PerformanceMetric.ROC_AUC_MICRO, PerformanceMetric.ROC_AUC_MACRO]:
                metrics[metric] = get_score(metric, y_true_proba, y_pred_proba, **scoring_kwargs)
        except:  # pylint: disable=bare-except # noqa: E722
            traceback.print_exc()

    return metrics


def compute_regression_metrics(y_true, y_pred, **scoring_kwargs):
    """
    Compute regression metrics given predications and ground truth.
    :param y_true: Ground truth (correct) target values or labels
    :param y_pred: Estimated target values or labels
    :return: metrics as a dict
    """

    return compute_metrics(REGRESSION_METRICS, y_true, y_pred, **scoring_kwargs)
