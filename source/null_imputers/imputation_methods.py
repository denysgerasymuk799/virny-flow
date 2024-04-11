import copy
import pandas as pd
from sklearn.impute import SimpleImputer

from source.null_imputers.automl_imputer import AutoMLImputer
from source.null_imputers.missforest_imputer import MissForestImputer
from source.null_imputers.kmeans_imputer import KMeansImputer
from source.utils.dataframe_utils import get_object_columns_indexes


def impute_with_simple_imputer(X_train_with_nulls: pd.DataFrame, X_test_with_nulls: pd.DataFrame,
                               numeric_columns_with_nulls: list, categorical_columns_with_nulls: list,
                               hyperparams: dict, **kwargs):
    X_train_imputed = copy.deepcopy(X_train_with_nulls)
    X_test_imputed = copy.deepcopy(X_test_with_nulls)

    # Impute numerical columns
    num_imputer = SimpleImputer(strategy=kwargs['num'])
    X_train_imputed[numeric_columns_with_nulls] = num_imputer.fit_transform(X_train_imputed[numeric_columns_with_nulls])
    X_test_imputed[numeric_columns_with_nulls] = num_imputer.transform(X_test_imputed[numeric_columns_with_nulls])

    # Impute categorical columns
    cat_imputer = SimpleImputer(strategy=kwargs['cat'], fill_value='missing') \
        if kwargs['cat'] == 'constant' else SimpleImputer(strategy=kwargs['cat'])
    X_train_imputed[categorical_columns_with_nulls] = cat_imputer.fit_transform(X_train_imputed[categorical_columns_with_nulls])
    X_test_imputed[categorical_columns_with_nulls] = cat_imputer.transform(X_test_imputed[categorical_columns_with_nulls])

    null_imputer_params_dct = None
    return X_train_imputed, X_test_imputed, null_imputer_params_dct


def impute_with_automl(X_train_with_nulls: pd.DataFrame, X_test_with_nulls: pd.DataFrame,
                       numeric_columns_with_nulls: list, categorical_columns_with_nulls: list,
                       hyperparams: dict, **kwargs):
    directory = kwargs['directory']
    seed = kwargs['experiment_seed']
    target_columns = list(set(numeric_columns_with_nulls) | set(categorical_columns_with_nulls))

    X_train_imputed = copy.deepcopy(X_train_with_nulls)
    X_test_imputed = copy.deepcopy(X_test_with_nulls)

    imputer = AutoMLImputer(max_trials=kwargs["max_trials"],
                            tuner=kwargs["tuner"],
                            validation_split=kwargs["validation_split"],
                            epochs=kwargs["epochs"],
                            seed=seed,
                            directory=directory)
    imputer.fit(X=X_train_imputed,
                target_columns=target_columns,
                verbose=0)

    X_train_imputed = imputer.transform(X_train_imputed)
    X_test_imputed = imputer.transform(X_test_imputed)

    null_imputer_params_dct = imputer.get_best_hyperparameters()
    return X_train_imputed, X_test_imputed, null_imputer_params_dct


def impute_with_missforest(X_train_with_nulls: pd.DataFrame, X_test_with_nulls: pd.DataFrame,
                            numeric_columns_with_nulls: list, categorical_columns_with_nulls: list,
                            hyperparams: dict, **kwargs):
    seed = kwargs['experiment_seed']
    
    X_train_imputed = copy.deepcopy(X_train_with_nulls)
    X_test_imputed = copy.deepcopy(X_test_with_nulls)
    
    # Impute numerical columns
    missforest_imputer = MissForestImputer(seed=seed)
    # Get indices of categorical columns
    categorical_columns_idxs = get_object_columns_indexes(X_train_imputed)
    
    X_train_imputed_values = missforest_imputer.fit_transform(X_train_imputed, cat_vars=categorical_columns_idxs)
    X_train_imputed = pd.DataFrame(X_train_imputed_values, columns=X_train_imputed.columns, index=X_train_imputed.index)
    # set the same columns types as in the original dataset
    X_train_imputed[categorical_columns_with_nulls] = X_train_imputed[categorical_columns_with_nulls].astype('int').astype(str)
    
    X_test_imputed_values = missforest_imputer.transform(X_test_imputed)
    X_test_imputed = pd.DataFrame(X_test_imputed_values, columns=X_test_imputed.columns, index=X_test_imputed.index)
    X_test_imputed[categorical_columns_with_nulls] = X_test_imputed[categorical_columns_with_nulls].astype('int').astype(str)
    
    null_imp_params_dct = None
    return X_train_imputed, X_test_imputed, null_imp_params_dct


def impute_with_kmeans(X_train_with_nulls: pd.DataFrame, X_test_with_nulls: pd.DataFrame,
                       numeric_columns_with_nulls: list, categorical_columns_with_nulls: list,
                       hyperparams: dict, **kwargs):
    seed = kwargs['experiment_seed']
    
    X_train_imputed = copy.deepcopy(X_train_with_nulls)
    X_test_imputed = copy.deepcopy(X_test_with_nulls)
    
    # Impute numerical columns
    kmeans_imputer = KMeansImputer(n_clusters=2, seed=seed)
    categorical_columns_idxs = get_object_columns_indexes(X_train_imputed)
    
    X_train_imputed_values = kmeans_imputer.fit_transform(X_train_imputed, cat_vars=categorical_columns_idxs)
    X_train_imputed = pd.DataFrame(X_train_imputed_values, columns=X_train_imputed.columns, index=X_train_imputed.index)
    # set the same columns types as in the original dataset
    X_train_imputed[categorical_columns_with_nulls] = X_train_imputed[categorical_columns_with_nulls].astype(int).astype('str')
    
    X_test_imputed_values = kmeans_imputer.transform(X_test_imputed)
    X_test_imputed = pd.DataFrame(X_test_imputed_values, columns=X_test_imputed.columns, index=X_test_imputed.index)
    X_test_imputed[categorical_columns_with_nulls] = X_test_imputed[categorical_columns_with_nulls].astype(int).astype('str')
    
    null_imp_params_dct = None
    return X_train_imputed, X_test_imputed, null_imp_params_dct
