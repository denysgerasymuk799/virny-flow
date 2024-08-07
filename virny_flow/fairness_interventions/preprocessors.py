import copy
import numpy as np

from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.preprocessing import DisparateImpactRemover


def remove_disparate_impact(init_base_flow_dataset, alpha, sensitive_attribute):
    """
    Based on this documentation:
     https://aif360.readthedocs.io/en/latest/modules/generated/aif360.algorithms.preprocessing.DisparateImpactRemover.html

    """
    base_flow_dataset = copy.deepcopy(init_base_flow_dataset)
    if str(alpha) == '0.0':
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

    di = DisparateImpactRemover(repair_level=alpha, sensitive_attribute=sensitive_attribute)
    train_repaired_df, _ = di.fit_transform(train_binary_dataset).convert_to_dataframe()
    test_repaired_df , _ = di.fit_transform(test_binary_dataset).convert_to_dataframe()
    train_repaired_df.index = train_repaired_df.index.astype(dtype='int64')
    test_repaired_df.index = test_repaired_df.index.astype(dtype='int64')

    base_flow_dataset.X_train_val = train_repaired_df.drop([base_flow_dataset.target, sensitive_attribute], axis=1)
    base_flow_dataset.X_test = test_repaired_df.drop([base_flow_dataset.target, sensitive_attribute], axis=1)

    return base_flow_dataset
