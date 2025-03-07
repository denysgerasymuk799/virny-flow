import copy
import pandas as pd
from sklearn.impute import SimpleImputer

from virny_flow.configs.structs import MixedImputer
from virny_flow.core.utils.pipeline_utils import encode_dataset_for_missforest, decode_dataset_for_missforest
from virny_flow.core.utils.dataframe_utils import get_numerical_columns_indexes

from .automl_imputer import AutoMLImputer
from .missforest_imputer import MissForestImputer
from .kmeans_imputer import KMeansImputer


def impute_with_deletion(X_train_with_nulls: pd.DataFrame, X_tests_with_nulls_lst: list,
                         numeric_columns_with_nulls: list, categorical_columns_with_nulls: list,
                         hyperparams: dict, **kwargs):
    X_train_imputed = copy.deepcopy(X_train_with_nulls)
    X_tests_imputed_lst = list(map(lambda X_test_with_nulls: copy.deepcopy(X_test_with_nulls), X_tests_with_nulls_lst))

    # Apply deletion for a train set
    X_train_imputed = X_train_imputed.dropna()

    # Apply median-mode for a test set
    num_imputer = SimpleImputer(strategy='median')
    num_imputer.fit(X_train_imputed[numeric_columns_with_nulls])
    for i in range(len(X_tests_imputed_lst)):
        X_tests_imputed_lst[i][numeric_columns_with_nulls] = num_imputer.transform(X_tests_imputed_lst[i][numeric_columns_with_nulls])

    cat_imputer = SimpleImputer(strategy='most_frequent')
    cat_imputer.fit(X_train_imputed[categorical_columns_with_nulls])
    for i in range(len(X_tests_imputed_lst)):
        X_tests_imputed_lst[i][categorical_columns_with_nulls] = cat_imputer.transform(X_tests_imputed_lst[i][categorical_columns_with_nulls])

    null_imputer_params_dct = None
    return X_train_imputed, X_tests_imputed_lst, null_imputer_params_dct


def impute_with_simple_imputer(X_train_with_nulls: pd.DataFrame, X_tests_with_nulls_lst: list,
                               numeric_columns_with_nulls: list, categorical_columns_with_nulls: list,
                               hyperparams: dict, **kwargs):
    mixed_imputer = MixedImputer(num_imputer=None, cat_imputer=None)

    X_train_imputed = copy.deepcopy(X_train_with_nulls)
    X_tests_imputed_lst = list(map(lambda X_test_with_nulls: copy.deepcopy(X_test_with_nulls), X_tests_with_nulls_lst))

    # Impute numerical columns
    if X_train_imputed[numeric_columns_with_nulls].isnull().values.any():
        mixed_imputer.num_imputer = SimpleImputer(strategy=kwargs['num'])
        X_train_imputed[numeric_columns_with_nulls] =\
            mixed_imputer.num_imputer.fit_transform(X_train_imputed[numeric_columns_with_nulls])
        for i in range(len(X_tests_imputed_lst)):
            X_tests_imputed_lst[i][numeric_columns_with_nulls] =\
                mixed_imputer.num_imputer.transform(X_tests_imputed_lst[i][numeric_columns_with_nulls])

    # Impute categorical columns
    if X_train_imputed[categorical_columns_with_nulls].isnull().values.any():
        mixed_imputer.cat_imputer = SimpleImputer(strategy=kwargs['cat'], fill_value='missing') \
            if kwargs['cat'] == 'constant' else SimpleImputer(strategy=kwargs['cat'])
        X_train_imputed[categorical_columns_with_nulls] =\
            mixed_imputer.cat_imputer.fit_transform(X_train_imputed[categorical_columns_with_nulls])
        for i in range(len(X_tests_imputed_lst)):
            X_tests_imputed_lst[i][categorical_columns_with_nulls] =\
                mixed_imputer.cat_imputer.transform(X_tests_imputed_lst[i][categorical_columns_with_nulls])

    null_imputer_params_dct = {'num_imputer_strategy': kwargs['num'], 'cat_imputer_strategy': kwargs['cat']}
    return X_train_imputed, X_tests_imputed_lst, null_imputer_params_dct


def impute_with_automl(X_train_with_nulls: pd.DataFrame, X_tests_with_nulls_lst: list,
                       numeric_columns_with_nulls: list, categorical_columns_with_nulls: list,
                       hyperparams: dict, **kwargs):
    directory = kwargs['directory']
    seed = kwargs['experiment_seed']
    target_columns = list(set(numeric_columns_with_nulls) | set(categorical_columns_with_nulls))

    X_train_imputed = copy.deepcopy(X_train_with_nulls)
    X_tests_imputed_lst = list(map(lambda X_test_with_nulls: copy.deepcopy(X_test_with_nulls), X_tests_with_nulls_lst))

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
    X_tests_imputed_lst = list(map(lambda X_test_imputed: imputer.transform(X_test_imputed), X_tests_imputed_lst))

    null_imputer_params_dct = imputer.get_best_hyperparameters()
    return X_train_imputed, X_tests_imputed_lst, null_imputer_params_dct


def impute_with_missforest(X_train_with_nulls: pd.DataFrame, X_tests_with_nulls_lst: list,
                           numeric_columns_with_nulls: list, categorical_columns_with_nulls: list,
                           hyperparams: dict, **kwargs):
    seed = kwargs['experiment_seed']
    dataset_name = kwargs['dataset_name']

    # Impute numerical columns
    missforest_imputer = MissForestImputer(seed=seed, hyperparams=hyperparams)

    X_train_encoded, cat_encoders, categorical_columns_idxs = encode_dataset_for_missforest(X_train_with_nulls,
                                                                                            dataset_name=dataset_name,
                                                                                            categorical_columns_with_nulls=categorical_columns_with_nulls)
    X_train_repaired_values = missforest_imputer.fit_transform(X_train_encoded.values.astype(float), cat_vars=categorical_columns_idxs)
    X_train_repaired = pd.DataFrame(X_train_repaired_values, columns=X_train_encoded.columns, index=X_train_encoded.index)
    X_train_imputed = decode_dataset_for_missforest(X_train_repaired, cat_encoders, dataset_name=dataset_name)

    X_tests_imputed_lst = []
    for i in range(len(X_tests_with_nulls_lst)):
        X_test_with_nulls = X_tests_with_nulls_lst[i]

        X_test_encoded, _, _ = encode_dataset_for_missforest(X_test_with_nulls,
                                                             cat_encoders=cat_encoders,
                                                             dataset_name=dataset_name,
                                                             categorical_columns_with_nulls=categorical_columns_with_nulls)
        X_test_repaired_values = missforest_imputer.transform(X_test_encoded.values.astype(float))
        X_test_repaired = pd.DataFrame(X_test_repaired_values, columns=X_test_encoded.columns, index=X_test_encoded.index)
        X_test_imputed = decode_dataset_for_missforest(X_test_repaired, cat_encoders, dataset_name=dataset_name)

        X_tests_imputed_lst.append(X_test_imputed)

    if hyperparams is not None:
        null_imp_params_dct = {col: hyperparams for col in X_train_with_nulls.columns}
    else:
        predictor_params = missforest_imputer.get_predictors_params()
        null_imp_params_dct = {X_train_with_nulls.columns[i]: {str(k): predictor_params[i][k] for k in predictor_params[i]} for i in predictor_params}
    
    return X_train_imputed, X_tests_imputed_lst, null_imp_params_dct


def impute_with_kmeans(X_train_with_nulls: pd.DataFrame, X_tests_with_nulls_lst: list,
                       numeric_columns_with_nulls: list, categorical_columns_with_nulls: list,
                       hyperparams: dict, **kwargs):
    seed = kwargs['experiment_seed']
    dataset_name = kwargs['dataset_name']

    # Set an appropriate kmeans_imputer_mode type
    numerical_columns_idxs = get_numerical_columns_indexes(X_train_with_nulls)
    if len(numerical_columns_idxs) == len(numeric_columns_with_nulls):
        kmeans_imputer_mode = "kmodes"
    else:
        kmeans_imputer_mode = "kprototypes"

    X_train_encoded, cat_encoders, categorical_columns_idxs = \
        encode_dataset_for_missforest(X_train_with_nulls,
                                      dataset_name=dataset_name,
                                      categorical_columns_with_nulls=categorical_columns_with_nulls)

    # Impute numerical columns
    kmeans_imputer = KMeansImputer(seed=seed, imputer_mode=kmeans_imputer_mode, hyperparameters=hyperparams)
    
    X_train_repaired_values = kmeans_imputer.fit_transform(X_train_encoded.values.astype(float), cat_vars=categorical_columns_idxs)
    X_train_repaired = pd.DataFrame(X_train_repaired_values, columns=X_train_encoded.columns, index=X_train_encoded.index)
    X_train_imputed = decode_dataset_for_missforest(X_train_repaired, cat_encoders, dataset_name=dataset_name)

    X_tests_imputed_lst = []
    for i in range(len(X_tests_with_nulls_lst)):
        X_test_with_nulls = X_tests_with_nulls_lst[i]

        X_test_encoded, _, _ = encode_dataset_for_missforest(X_test_with_nulls,
                                                             cat_encoders=cat_encoders,
                                                             dataset_name=dataset_name,
                                                             categorical_columns_with_nulls=categorical_columns_with_nulls)
        X_test_repaired_values = kmeans_imputer.transform(X_test_encoded.values.astype(float))
        X_test_repaired = pd.DataFrame(X_test_repaired_values, columns=X_test_encoded.columns, index=X_test_encoded.index)
        X_test_imputed = decode_dataset_for_missforest(X_test_repaired, cat_encoders, dataset_name=dataset_name)

        X_tests_imputed_lst.append(X_test_imputed)
    
    null_imp_params_dct = {col: kmeans_imputer.get_predictors_params() for col in X_train_with_nulls.columns}
    return X_train_imputed, X_tests_imputed_lst, null_imp_params_dct
