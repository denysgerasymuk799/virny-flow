import copy
import numpy as np

from virny.utils.postprocessing_intervention_utils import construct_binary_label_dataset_from_df


class ExpGradientReductionWrapper:
    """
    A wrapper for fair inprocessors from aif360. The wrapper aligns fit, predict, and predict_proba methods
    to be compatible with sklearn models.

    Parameters
    ----------
    inprocessor
        An initialized inprocessor from aif360.
    sensitive_attr_for_intervention
        A sensitive attribute name to use in the fairness in-processing intervention.

    """

    def __init__(self, inprocessor, sensitive_attr_for_intervention):
        self.sensitive_attr_for_intervention = sensitive_attr_for_intervention
        self.inprocessor = inprocessor

    def fit(self, X, y):
        train_binary_dataset = construct_binary_label_dataset_from_df(X_sample=X,
                                                                      y_sample=y,
                                                                      target_column='target',
                                                                      sensitive_attribute=self.sensitive_attr_for_intervention)
        # Fit an inprocessor
        self.inprocessor.fit(train_binary_dataset)
        return self

    def predict_proba(self, X):
        y_empty = np.zeros(X.shape[0])
        test_binary_dataset = construct_binary_label_dataset_from_df(X_sample=X,
                                                                     y_sample=y_empty,
                                                                     target_column='target',
                                                                     sensitive_attribute=self.sensitive_attr_for_intervention)
        test_dataset_pred = self.inprocessor.predict(test_binary_dataset)
        # Set 1.0 since ExponentiatedGradientReduction can return probabilities slightly higher than 1.0.
        # This can cause Infinity values for entropy.
        test_dataset_pred.scores[test_dataset_pred.scores > 1.0] = 1.0
        # Return 1 - test_dataset_pred.scores since scores are probabilities for label 1, not for label 0
        return 1 - test_dataset_pred.scores.flatten()

    def predict(self, X):
        y_empty = np.zeros(shape=X.shape[0])
        test_binary_dataset = construct_binary_label_dataset_from_df(X_sample=X,
                                                                     y_sample=y_empty,
                                                                     target_column='target',
                                                                     sensitive_attribute=self.sensitive_attr_for_intervention)
        test_dataset_pred = self.inprocessor.predict(test_binary_dataset)
        return test_dataset_pred.labels.flatten()

    def __copy__(self):
        new_inprocessor = copy.copy(self.inprocessor)
        return ExpGradientReductionWrapper(inprocessor=new_inprocessor,
                                           sensitive_attr_for_intervention=self.sensitive_attr_for_intervention)

    def __deepcopy__(self, memo):
        new_inprocessor = copy.deepcopy(self.inprocessor)
        return ExpGradientReductionWrapper(inprocessor=new_inprocessor,
                                           sensitive_attr_for_intervention=self.sensitive_attr_for_intervention)

    def get_params(self):
        return {'sensitive_attr_for_intervention': self.sensitive_attr_for_intervention}
