import pytest
import numpy as np
import pandas as pd

from source.null_imputers.kmeans_imputer import KMeansImputer
from source.utils.dataframe_utils import get_object_columns_indexes, get_numerical_columns_indexes
from sklearn.impute import SimpleImputer


# Test if output of KMeansImputer does not contain nulls
def test_kmeans_imputer_no_nulls(acs_income_dataset_categorical_columns_idxs, common_seed, kmeans_acs_income_hyperparams):
    injected_df, categorical_columns_idxs = acs_income_dataset_categorical_columns_idxs
    # Initialize KMeansImputer
    imputer = KMeansImputer(seed=common_seed, imputer_mode="kprototypes", hyperparameters=kmeans_acs_income_hyperparams)

    # Fit and transform the dataset
    X_imputed = imputer.fit_transform(injected_df, cat_vars=categorical_columns_idxs)

    # Check if there are any nulls in the output
    assert not np.isnan(X_imputed).any(), "Output contains null values"


# Test if KMeansImputer returns same results with the same seed
def test_kmeans_imputer_same_seed(acs_income_dataset_categorical_columns_idxs, common_seed, kmeans_acs_income_hyperparams):
    injected_df, categorical_columns_idxs = acs_income_dataset_categorical_columns_idxs
    
    # Initialize KMeansImputer with seed
    imputer1 = KMeansImputer(seed=common_seed, imputer_mode="kprototypes", hyperparameters=kmeans_acs_income_hyperparams)
    imputer2 = KMeansImputer(seed=common_seed, imputer_mode="kprototypes", hyperparameters=kmeans_acs_income_hyperparams)

    # Fit and transform the sample data with imputer1
    X_imputed1 = imputer1.fit_transform(injected_df, cat_vars=categorical_columns_idxs)

    # Fit and transform the sample data with imputer2
    X_imputed2 = imputer2.fit_transform(injected_df, cat_vars=categorical_columns_idxs)

    # Check if the results are identical
    np.testing.assert_allclose(
        X_imputed1, X_imputed2, 
        atol=1e-9, rtol=1e-9, err_msg="Results from KMeansImputer are not identical"
    )


def test_kmeans_imputer_no_change(acs_income_dataset_categorical_columns_idxs, common_seed, kmeans_acs_income_hyperparams):
    injected_df, categorical_columns_idxs = acs_income_dataset_categorical_columns_idxs
    not_null_indices = injected_df.index[injected_df.notnull().all(axis=1)]
    categorical_columns_idxs = get_object_columns_indexes(injected_df)
    
    # Initialize KMeansImputer
    imputer = KMeansImputer(seed=common_seed, imputer_mode="kprototypes", hyperparameters=kmeans_acs_income_hyperparams)
    
    # Fit and transform the dataset
    X_imputed = imputer.fit_transform(injected_df, cat_vars=categorical_columns_idxs)
    
    X_imputed_df = pd.DataFrame(X_imputed, columns=injected_df.columns, index=injected_df.index)
    X_imputed_df.iloc[:, categorical_columns_idxs] = X_imputed_df.iloc[:, categorical_columns_idxs].astype(int).astype('str')
    
    # check if the rows that had no nulls are the same
    np.testing.assert_array_equal(injected_df.loc[not_null_indices].values, X_imputed_df.loc[not_null_indices].values, "Rows with no nulls have changed")


# Run the tests
if __name__ == "__main__":
    pytest.main([__file__])
