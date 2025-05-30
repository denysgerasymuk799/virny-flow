import sys
import numpy as np
import pandas as pd

from virny.configs.constants import ALEATORIC_UNCERTAINTY, EPISTEMIC_UNCERTAINTY, OVERALL_UNCERTAINTY, ComputationMode
from virny.metrics import METRIC_TO_FUNCTION, METRICS_FOR_PREDICT_PROBA, METRICS_FOR_LABELS


def combine_bootstrap_predictions(bootstrap_predictions: dict, y_test_indexes: np.ndarray):
    """
    Combine predictions generated by estimators in the bootstrap to get final 1D array of predictions.

    Return a pandas series of predictions for each test sample.

    Parameters
    ----------
    bootstrap_predictions
        A dictionary where keys are indexes of bootstrap estimators and values are their predictions for the test set.
    y_test_indexes
        Indexes of the initial test set to keep original row indexes.

    """
    if isinstance(bootstrap_predictions, np.ndarray):
        results = pd.DataFrame(bootstrap_predictions)
    else:
        results = pd.DataFrame(bootstrap_predictions).transpose()

    main_prediction = results.mean().values
    y_preds = np.array([int(x<0.5) for x in main_prediction])

    return pd.Series(y_preds, index=y_test_indexes)


def count_prediction_metrics(y_true, uq_results, with_predict_proba: bool = True, computation_mode = None):
    """
    Compute means, stds, iqr, entropy, jitter, label stability, and transform predictions to pd.Dataframe.

    Return a 1D numpy array of predictions, 2D array of each model prediction for y_test, a data structure of metrics.

    Parameters
    ----------
    y_true
        True labels
    uq_results
        2D array of prediction proba for the zero value label by each model
    with_predict_proba
        [Optional] A flag if model can return probabilities for its predictions.
         If no, only metrics based on labels (not labels and probabilities) will be computed.
    computation_mode
        [Optional] A mode for computing metrics

    """
    if isinstance(uq_results, np.ndarray):
        results = pd.DataFrame(uq_results)
    else:
        results = pd.DataFrame(uq_results).transpose()

    metrics_dct = dict()
    if computation_mode != ComputationMode.NO_BOOTSTRAP.value: # Do not compute stability and uncertainty metrics for NO_BOOTSTRAP
        # Compute metrics for prediction probabilities
        if not with_predict_proba:
            uq_labels = results
        else:
            uq_predict_probas = results
            for metric in METRICS_FOR_PREDICT_PROBA:
                if metric == EPISTEMIC_UNCERTAINTY: # skip computation for a metric that is based on two other metrics
                    continue

                metrics_dct[metric] = METRIC_TO_FUNCTION[metric](y_true, uq_predict_probas)

            metrics_dct[EPISTEMIC_UNCERTAINTY] = metrics_dct[OVERALL_UNCERTAINTY] - metrics_dct[ALEATORIC_UNCERTAINTY]

            # Convert predict proba results of each model to correspondent labels.
            # Here we use int(x<0.5) since we use predict_prob()[:, 0] to make predictions.
            # Hence, if a value is, for example, 0.3 --> label == 1, 0.6 -- > label == 0
            uq_labels = (results < 0.5).astype(int)

        # Compute metrics for prediction labels
        for metric in METRICS_FOR_LABELS:
            metrics_dct[metric] = METRIC_TO_FUNCTION[metric](y_true, uq_labels)

    if with_predict_proba:
        y_preds = np.array([int(x<0.5) for x in results.mean().values])
    else:
        y_preds = np.array([int(x>0.5) for x in results.mean().values])

    return y_preds, metrics_dct


def generate_bootstrap(features, labels, boostrap_size, with_replacement=True, random_state=None):
    if boostrap_size == features.shape[0]:
        return pd.DataFrame(features), pd.DataFrame(labels)

    # Create a local random state.
    # Note that to keep reverse compatibility we need to use different generators for different python versions
    # since random number generation was changed in Python 3.12
    if sys.version_info.major == 3 and sys.version_info.minor >= 12:
        rng = np.random.default_rng(seed=random_state)
    else:
        rng = np.random.RandomState(random_state)

    # Generate bootstrapped indexes
    bootstrap_index = rng.choice(features.shape[0], size=boostrap_size, replace=with_replacement)
    bootstrap_features = pd.DataFrame(features).iloc[bootstrap_index]
    bootstrap_labels = pd.DataFrame(labels).iloc[bootstrap_index]
    if len(bootstrap_features) == boostrap_size:
        return bootstrap_features, bootstrap_labels
    else:
        raise ValueError('Bootstrap samples are not of the size requested')
