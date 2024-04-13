import numpy as np

from source.preprocessing import get_simple_preprocessor


def preprocess_base_flow_dataset(base_flow_dataset):
    column_transformer = get_simple_preprocessor(base_flow_dataset)
    column_transformer = column_transformer.set_output(transform="pandas")  # Set transformer output to a pandas df
    base_flow_dataset.X_train_val = column_transformer.fit_transform(base_flow_dataset.X_train_val)
    base_flow_dataset.X_test = column_transformer.transform(base_flow_dataset.X_test)

    return base_flow_dataset


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