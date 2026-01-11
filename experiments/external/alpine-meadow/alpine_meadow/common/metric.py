"""Scoring utility functions."""

from alpine_meadow.utils import ignore_warnings, AMException
from .task import PerformanceMetric


def is_better_score(metric, score1, score2):
    """
    Given the metric and two scores, return True if score1 is better than score2 under the metric
    """

    metric_name = PerformanceMetric.Name(metric)

    if metric_name in ['ACCURACY', 'PRECISION', 'RECALL', 'F1', 'F1_MICRO', 'F1_MACRO', 'ROC_AUC',
                       'ROC_AUC_MICRO', 'ROC_AUC_MACRO', 'R_SQUARED', 'NORMALIZED_MUTUAL_INFORMATION',
                       'COHEN_KAPPA']:
        return score1 > score2
    return score1 < score2


def score_to_error(metric, score):
    """
    Convert the raw score into error-based metric (the lower the better).
    """

    if is_better_score(metric, score + 1, score):
        return 1.0 - score
    return score


@ignore_warnings
def get_score(metric, true_results, predicted_results, logger=None, **kwargs):
    """
    Compute the score based on true_results and predicted_results
    """

    if metric == PerformanceMetric.Value('ACCURACY'):
        from sklearn.metrics import accuracy_score

        score = accuracy_score(true_results, predicted_results)
    elif metric == PerformanceMetric.Value('PRECISION'):
        from sklearn.metrics import precision_score

        pos_label = kwargs.get('pos_label', None)
        if pos_label is None:
            raise AMException("Expect pos_label for precision")
        score = precision_score(true_results, predicted_results, pos_label=pos_label, labels=[pos_label],
                                average=None)[0]
    elif metric == PerformanceMetric.Value('RECALL'):
        from sklearn.metrics import recall_score

        pos_label = kwargs.get('pos_label', None)
        if pos_label is None:
            raise AMException("Expect pos_label for recall")
        score = recall_score(true_results, predicted_results, pos_label=pos_label, labels=[pos_label],
                             average=None)[0]
    elif metric == PerformanceMetric.Value('F1'):
        if kwargs.get('class_weights', None):
            try:
                from sklearn.metrics import classification_report
                import numpy as np

                class_weights = kwargs['class_weights']
                dict_ = classification_report(true_results, predicted_results, output_dict=True)
                dict_ = {str(key): value for key, value in dict_.items()}
                weights = []
                scores = []
                for class_ in class_weights:
                    if class_ in dict_:
                        weights.append(class_weights[class_])
                        scores.append(dict_[str(class_)]['f1-score'])
                return np.average(scores, weights=weights)
            except BaseException:  # pylint: disable=broad-except
                if logger:
                    logger.error(msg='', exc_info=True)

        from sklearn.metrics import f1_score

        pos_label = kwargs.get('pos_label', None)
        if pos_label is None:
            raise AMException("Expect pos_label for f1")
        score = f1_score(true_results, predicted_results, pos_label=pos_label, labels=[pos_label],
                         average=None)[0]
    elif metric == PerformanceMetric.Value('F1_MICRO'):
        from sklearn.metrics import f1_score

        score = f1_score(true_results, predicted_results, average='micro')
    elif metric == PerformanceMetric.Value('F1_MACRO'):
        from sklearn.metrics import f1_score

        score = f1_score(true_results, predicted_results, average='macro')
    elif metric == PerformanceMetric.Value('ROC_AUC'):
        from sklearn.metrics import roc_auc_score

        if 'all_labels' in kwargs:
            return roc_auc_score(true_results, predicted_results, average=None,
                                 labels=kwargs['all_labels'])

        pos_label = kwargs.get('pos_label', None)
        if pos_label is None:
            raise AMException("Expect pos_label for roc_auc")
        labels = list(true_results.columns)
        if pos_label not in labels:
            raise AMException("pos_label is not in ground truth")

        score = roc_auc_score(true_results, predicted_results, average=None,
                              labels=labels)[labels.index(pos_label)]
    elif metric == PerformanceMetric.Value('ROC_AUC_MICRO'):
        from sklearn.metrics import roc_auc_score

        score = roc_auc_score(true_results, predicted_results, average='micro', multi_class='ovr')
    elif metric == PerformanceMetric.Value('ROC_AUC_MACRO'):
        from sklearn.metrics import roc_auc_score

        score = roc_auc_score(true_results, predicted_results, average='macro', multi_class='ovr')
    elif metric == PerformanceMetric.Value('ROOT_MEAN_SQUARED_ERROR'):
        from sklearn.metrics import mean_squared_error
        import numpy as np

        score = np.sqrt(mean_squared_error(true_results, predicted_results))
    elif metric == PerformanceMetric.Value('MEAN_ABSOLUTE_ERROR'):
        from sklearn.metrics import mean_absolute_error

        score = mean_absolute_error(true_results, predicted_results)
    elif metric == PerformanceMetric.Value('R_SQUARED'):
        from sklearn.metrics import r2_score

        score = r2_score(true_results, predicted_results)
    elif metric == PerformanceMetric.Value('MEAN_SQUARED_ERROR'):
        from sklearn.metrics import mean_squared_error

        score = mean_squared_error(true_results, predicted_results)
    elif metric == PerformanceMetric.Value('NORMALIZED_MUTUAL_INFORMATION'):
        from sklearn.metrics import normalized_mutual_info_score

        score = normalized_mutual_info_score(true_results, predicted_results)
    elif metric == PerformanceMetric.Value('COHEN_KAPPA'):
        from sklearn.metrics import cohen_kappa_score

        score = cohen_kappa_score(true_results, predicted_results)
    else:
        raise AMException(f"Unknown metric: {metric}")

    return score
