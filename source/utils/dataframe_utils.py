import numpy as np
import pandas as pd
from scipy.stats import entropy

from source.preprocessing import get_simple_preprocessor


def preprocess_base_flow_dataset(base_flow_dataset):
    column_transformer = get_simple_preprocessor(base_flow_dataset)
    column_transformer = column_transformer.set_output(transform="pandas")  # Set transformer output to a pandas df
    base_flow_dataset.X_train_val = column_transformer.fit_transform(base_flow_dataset.X_train_val)
    base_flow_dataset.X_test = column_transformer.transform(base_flow_dataset.X_test)

    return base_flow_dataset


def preprocess_mult_base_flow_datasets(main_base_flow_dataset, extra_base_flow_datasets):
    column_transformer = get_simple_preprocessor(main_base_flow_dataset)
    column_transformer = column_transformer.set_output(transform="pandas")  # Set transformer output to a pandas df

    # Preprocess main_base_flow_dataset
    main_base_flow_dataset.X_train_val = column_transformer.fit_transform(main_base_flow_dataset.X_train_val)
    main_base_flow_dataset.X_test = column_transformer.transform(main_base_flow_dataset.X_test)

    print('preprocessed ordinal columns\n',
          main_base_flow_dataset.X_train_val[
              ['ord__' + col for col in list(main_base_flow_dataset.ordered_categories_dct.keys())]
          ].head(20))

    # Preprocess extra_base_flow_datasets
    extra_test_sets = []
    for i in range(len(extra_base_flow_datasets)):
        extra_base_flow_datasets[i].X_test = column_transformer.transform(extra_base_flow_datasets[i].X_test)
        extra_test_sets.append((extra_base_flow_datasets[i].X_test,
                                extra_base_flow_datasets[i].y_test,
                                extra_base_flow_datasets[i].init_sensitive_attrs_df))

    return main_base_flow_dataset, extra_test_sets


def get_object_columns_indexes(df):
    """
    Get the indexes of columns with object dtype in a pandas DataFrame.

    Parameters:
    df (pd.DataFrame): Input pandas DataFrame.

    Returns:
    list: Indexes of columns with object dtype.
    """
    object_columns = df.select_dtypes(include=['object']).columns
    object_indexes = [df.columns.get_loc(col) for col in object_columns]
    
    return object_indexes


def get_numerical_columns_indexes(df):
    """
    Get the indexes of columns with numerical dtype in a pandas DataFrame.

    Parameters:
    df (pd.DataFrame): Input pandas DataFrame.

    Returns:
    list: Indexes of columns with numerical dtype.
    """
    numerical_columns = df.select_dtypes(include=['int', 'float']).columns
    numerical_indexes = [df.columns.get_loc(col) for col in numerical_columns]
    
    return numerical_indexes


def _get_mask(X, value_to_mask):
    """Compute the boolean mask X == missing_values."""
    if value_to_mask == "NaN" or np.isnan(value_to_mask):
        return np.isnan(X)
    else:
        return X == value_to_mask


def get_columns_sorted_by_nulls(mask):
    # Calculate the number of null values in each column
    null_counts = mask.sum()

    # Sort columns based on the number of null values
    sorted_columns = null_counts.sort_values(ascending=True)

    # Get the column names as a list
    sorted_columns_names = sorted_columns.index.tolist()

    return sorted_columns_names


def calculate_kl_divergence(true: pd.DataFrame, pred: pd.DataFrame):
    # Get the value counts normalized to probability distributions
    true_dist = true.value_counts(normalize=True)
    pred_dist = pred.value_counts(normalize=True)

    # Ensure both distributions have the same index
    all_categories = true_dist.index.union(pred_dist.index)
    true_dist = true_dist.reindex(all_categories, fill_value=0.000000001).sort_index()
    pred_dist = pred_dist.reindex(all_categories, fill_value=0.000000001).sort_index()

    # Calculate KL divergence from true_dist to pred_dist
    # KL(P || Q) where P is the true distribution and Q is the approximation
    kl_div = entropy(true_dist, pred_dist)

    return kl_div
