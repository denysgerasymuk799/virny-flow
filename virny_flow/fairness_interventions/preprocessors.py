import copy
import numpy as np

from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.preprocessing import DisparateImpactRemover, LFR


def remove_disparate_impact(init_base_flow_dataset, repair_level, sensitive_attribute):
    """
    Based on this documentation:
     https://aif360.readthedocs.io/en/latest/modules/generated/aif360.algorithms.preprocessing.DisparateImpactRemover.html

    """
    base_flow_dataset = copy.deepcopy(init_base_flow_dataset)
    if str(repair_level) == '0.0':
        print('Skip preprocessing')
        base_flow_dataset.X_train_val = base_flow_dataset.X_train_val.drop([sensitive_attribute], axis=1)
        base_flow_dataset.X_test = base_flow_dataset.X_test.drop([sensitive_attribute], axis=1)
        return base_flow_dataset

    train_df = base_flow_dataset.X_train_val
    train_df[base_flow_dataset.target] = base_flow_dataset.y_train_val
    test_df = base_flow_dataset.X_test
    test_df[base_flow_dataset.target] = base_flow_dataset.y_test

    train_binary_dataset = BinaryLabelDataset(df=train_df,
                                              label_names=[base_flow_dataset.target],
                                              protected_attribute_names=[sensitive_attribute],
                                              favorable_label=1,
                                              unfavorable_label=0)
    test_binary_dataset = BinaryLabelDataset(df=test_df,
                                             label_names=[base_flow_dataset.target],
                                             protected_attribute_names=[sensitive_attribute],
                                             favorable_label=1,
                                             unfavorable_label=0)
    # Set labels (aka y_test) to zeros since we do not know labels during inference
    test_binary_dataset.labels = np.zeros(shape=np.shape(test_binary_dataset.labels))

    di = DisparateImpactRemover(repair_level=repair_level, sensitive_attribute=sensitive_attribute)
    train_repaired_df, _ = di.fit_transform(train_binary_dataset).convert_to_dataframe()
    test_repaired_df , _ = di.fit_transform(test_binary_dataset).convert_to_dataframe()
    train_repaired_df.index = train_repaired_df.index.astype(dtype='int64')
    test_repaired_df.index = test_repaired_df.index.astype(dtype='int64')

    base_flow_dataset.X_train_val = train_repaired_df.drop([base_flow_dataset.target, sensitive_attribute], axis=1)
    base_flow_dataset.X_test = test_repaired_df.drop([base_flow_dataset.target, sensitive_attribute], axis=1)

    return base_flow_dataset, di


def apply_learning_fair_representations(init_base_flow_dataset, intervention_options, sensitive_attribute):
    """
    Based on this documentation:
     https://aif360.readthedocs.io/en/v0.2.3/modules/preprocessing.html#learning-fair-representations

    Reference source code:
     https://github.com/giandos200/Zemel-et-al.-2013-Learning-Fair-Representations-/blob/main/main.py

    """
    base_flow_dataset = copy.deepcopy(init_base_flow_dataset)
    train_df = base_flow_dataset.X_train_val
    train_df[base_flow_dataset.target] = base_flow_dataset.y_train_val
    test_df = base_flow_dataset.X_test
    test_df[base_flow_dataset.target] = base_flow_dataset.y_test

    train_binary_dataset = BinaryLabelDataset(df=train_df,
                                              label_names=[base_flow_dataset.target],
                                              protected_attribute_names=[sensitive_attribute],
                                              favorable_label=1,
                                              unfavorable_label=0)
    test_binary_dataset = BinaryLabelDataset(df=test_df,
                                             label_names=[base_flow_dataset.target],
                                             protected_attribute_names=[sensitive_attribute],
                                             favorable_label=1,
                                             unfavorable_label=0)
    # Set labels (aka y_test) to zeros since we do not know labels during inference
    test_binary_dataset.labels = np.zeros(shape=np.shape(test_binary_dataset.labels))

    # Fair preprocessing.
    # Fit and transform only train and validation sets since the intervention changes also labels,
    # which we do not know for a test set in the production case.
    privileged_groups = [{sensitive_attribute: 1}]
    unprivileged_groups = [{sensitive_attribute: 0}]
    lfr_model = LFR(unprivileged_groups=unprivileged_groups,
                    privileged_groups=privileged_groups,
                    k=intervention_options['k'],
                    Ax=intervention_options['Ax'],
                    Ay=intervention_options['Ay'],
                    Az=intervention_options['Az'],
                    seed=42,
                    verbose=1)
    lfr_model = lfr_model.fit(train_binary_dataset, maxiter=5000, maxfun=5000)
    train_repaired_df, _ = lfr_model.transform(train_binary_dataset).convert_to_dataframe()
    test_repaired_df, _ = lfr_model.transform(test_binary_dataset).convert_to_dataframe()
    train_repaired_df.index = train_repaired_df.index.astype(dtype='int64')
    test_repaired_df.index = test_repaired_df.index.astype(dtype='int64')

    # Do NOT change base_flow_dataset.y_train_val and base_flow_dataset.y_test, keep original ones.
    # Use preprocessed X_train_val and X_test with original y_train_val and y_test for model fitting.
    # More details are in this part of the code from this repo:
    # https://github.com/giandos200/Zemel-et-al.-2013-Learning-Fair-Representations-/blob/main/main.py#L62
    base_flow_dataset.X_train_val = train_repaired_df.drop([base_flow_dataset.target, sensitive_attribute], axis=1)
    base_flow_dataset.X_test = test_repaired_df.drop([base_flow_dataset.target, sensitive_attribute], axis=1)

    return base_flow_dataset, lfr_model
