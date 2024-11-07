import lightgbm
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pprint import pprint, pformat
from copy import deepcopy
from datetime import datetime

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import make_scorer, accuracy_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

from virny.custom_classes.base_dataset import BaseFlowDataset
from virny_flow.custom_classes.grid_search_cv_with_early_stopping import GridSearchCVWithEarlyStopping


def validate_model(model, x, y, params, n_folds):
    """
    Use GridSearchCV for a special model to find the best hyperparameters based on validation set
    """
    if isinstance(model, lightgbm.LGBMClassifier):
        grid_search_cv_class = GridSearchCVWithEarlyStopping
    else:
        grid_search_cv_class = GridSearchCV

    grid_search = grid_search_cv_class(estimator=model,
                                       param_grid=params,
                                       scoring={
                                           "F1_Score": make_scorer(f1_score, average='macro'),
                                           "Accuracy_Score": make_scorer(accuracy_score),
                                       },
                                       refit="F1_Score",
                                       n_jobs=-1,
                                       cv=n_folds,
                                       verbose=0)
    grid_search.fit(x, y.values.ravel())
    best_index = grid_search.best_index_

    return grid_search.best_estimator_, \
           grid_search.cv_results_["mean_test_F1_Score"][best_index], \
           grid_search.cv_results_["mean_test_Accuracy_Score"][best_index], \
           grid_search.best_params_


def test_evaluation(cur_best_model, model_name, cur_best_params,
                    cur_x_train, cur_y_train, cur_x_test, cur_y_test,
                    dataset_title, show_plots, debug_mode):
    """
    Evaluate model on test set.

    :return: F1 score, accuracy and predicted values, which we use to visualisations for model comparison later.
    """
    cur_best_model.fit(cur_x_train, cur_y_train.values.ravel()) # refit model on the whole train set
    cur_model_pred = cur_best_model.predict(cur_x_test)
    test_f1_score = f1_score(cur_y_test, cur_model_pred, average='macro')
    test_accuracy = accuracy_score(cur_y_test, cur_model_pred)

    if debug_mode:
        print("#" * 20, f' {dataset_title} ', "#" * 20)
        print('Test model: ', model_name)
        print('Test model parameters:')
        pprint(cur_best_params)

        # print the scores
        print()
        print(classification_report(cur_y_test, cur_model_pred, digits=3))

    if show_plots:
        # plot the confusion matrix
        sns.set_style("white")
        cm = confusion_matrix(cur_y_test, cur_model_pred, labels=cur_best_model.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Employed", "Not Employed"])
        disp.plot()
        plt.show()

    return test_f1_score, test_accuracy, cur_model_pred


def evaluate_and_select_top_configs(model_name: str, model_params: dict, X_train, y_train, 
                                    dataset_name: str, round_num: int, selection_rate: float, 
                                    tuned_params_df: pd.DataFrame):
    """
    Evaluates each configuration of the model, logs the results, and selects the top configurations 
    based on F1 score for the next round.

    Parameters:
    - model_name: Name of the model being evaluated.
    - model_params: Dictionary containing the model instance and list of parameter configurations.
    - X_train, y_train: Training data.
    - dataset_name: Name of the dataset.
    - round_num: Current round of evaluation.
    - selection_rate: Percentage of top configurations to select for the next round.
    - tuned_params_df: DataFrame to log evaluation results.

    Returns:
    - top_configs: List of top configurations for the next round.
    - tuned_params_df: Updated DataFrame with logged results.
    """
    scores = []
    print(f"\nEvaluating configurations for model: {model_name}")

    for config_idx, params in enumerate(model_params['params']):
        try:
            tuning_start_time = datetime.now()
            print(f"{tuning_start_time.strftime('%Y/%m/%d, %H:%M:%S')}: Evaluating configuration {config_idx + 1}...", flush=True)

            # Validate model with current parameters
            cur_model, cur_f1_score, cur_accuracy, cur_params = validate_model(
                deepcopy(model_params['model']), X_train, y_train, [params], n_folds=1)

            tuning_end_time = datetime.now()
            tuning_duration = (tuning_end_time - tuning_start_time).total_seconds() / 60.0

            # Log results for this configuration
            tuned_params_df = pd.concat([
                tuned_params_df,
                pd.DataFrame({
                    'Dataset_Name': [dataset_name],
                    'Model_Name': [model_name],
                    'F1_Score': [cur_f1_score],
                    'Accuracy_Score': [cur_accuracy],
                    'Runtime_In_Mins': [tuning_duration],
                    'Model_Best_Params': [cur_params],
                    'Round': [round_num]
                })
            ], ignore_index=True)

            # Append score for selection
            scores.append((cur_f1_score, cur_accuracy, cur_params, deepcopy(model_params['model'])))

        except Exception as err:
            print(f"ERROR with configuration {config_idx + 1} for {model_name}: ", err)
            continue

    # Select top configurations based on F1 Score
    scores.sort(reverse=True, key=lambda x: x[0])  # Sort by F1 Score (highest first)
    top_configs = scores[:max(1, int(len(scores) * selection_rate))]  # Select top configurations

    return top_configs, tuned_params_df


def adaptive_model_selection(models_params_for_adaptive_selection: dict, base_flow_dataset: BaseFlowDataset,
                              dataset_name: str, dataset_fraction_per_round: list[float], selection_rate: float = 0.25):
    """
    Adaptive tuning pipeline to iteratively evaluate hyperparameter configurations with an increasing
    training dataset size and select the best-performing configurations for each model separately at each round.
    """
    models_config = dict()
    tuned_params_df = pd.DataFrame(columns=['Dataset_Name', 'Model_Name', 'F1_Score', 'Accuracy_Score', 
                                            'Runtime_In_Mins', 'Model_Best_Params', 'Round'])
    
    for round_num, data_fraction in enumerate(dataset_fraction_per_round, start=1):
        print(f"\nRound {round_num}: Evaluating configurations on {int(data_fraction * 100)}% of the dataset.")
        
        # Split dataset based on current fraction
        X_train, _, y_train, _ = train_test_split(base_flow_dataset.X_train_val, base_flow_dataset.y_train_val, 
                                                  train_size=data_fraction, stratify=base_flow_dataset.y_train_val)
        
        # Evaluate each model's configurations separately
        for model_name, model_params in models_params_for_adaptive_selection.items():
            top_configs, tuned_params_df = evaluate_and_select_top_configs(
                model_name, model_params, X_train, y_train, dataset_name, round_num, selection_rate, tuned_params_df
            )

            # Update configurations for the next round
            models_params_for_adaptive_selection[model_name]['params'] = [config[2] for config in top_configs]  # Keep only the top params
            models_params_for_adaptive_selection[model_name]['model'] = deepcopy(top_configs[0][3])  # Update model with best params

            print(f"{model_name}: Selected {len(top_configs)} configurations for the next round.")

        print(f"Round {round_num} completed.")

    # After the final round, save the best configuration for each model
    for model_name, model_info in models_params_for_adaptive_selection.items():
        models_config[model_name] = model_info['model']

    return tuned_params_df, models_config


def tune_ML_models(models_params_for_tuning: dict, base_flow_dataset: BaseFlowDataset,
                   dataset_name: str, n_folds: int = 3):
    """
    Tune each model on a validation set with GridSearchCV.

    Return each model with its best hyperparameters that have the highest F1 score and Accuracy.
     results_df is a dataframe with metrics and tuned parameters;
     models_config is a dict with model tuned params for the metrics computation stage
    """
    models_config = dict()
    tuned_params_df = pd.DataFrame(columns=('Dataset_Name', 'Model_Name', 'F1_Score', 'Accuracy_Score', 'Runtime_In_Mins', 'Model_Best_Params'))
    # Find the most optimal hyperparameters based on accuracy and F1-score for each model in models_config
    for model_idx, (model_name, model_params) in enumerate(models_params_for_tuning.items()):
        try:
            tuning_start_time = datetime.now()
            print(f"{tuning_start_time.strftime('%Y/%m/%d, %H:%M:%S')}: Tuning {model_name}...", flush=True)
            cur_model, cur_f1_score, cur_accuracy, cur_params = validate_model(deepcopy(model_params['model']),
                                                                               base_flow_dataset.X_train_val,
                                                                               base_flow_dataset.y_train_val,
                                                                               model_params['params'],
                                                                               n_folds)
            tuning_end_time = datetime.now()
            print(f'{tuning_end_time.strftime("%Y/%m/%d, %H:%M:%S")}: Tuning for {model_name} is finished '
                  f'[F1 score = {cur_f1_score}, Accuracy = {cur_accuracy}]\n', flush=True)
            print(f'Best hyper-parameters for {model_name}:\n{pformat(cur_params)}', flush=True)

        except Exception as err:
            print(f"ERROR with {model_name}: ", err)
            continue

        # Save test results of each model in dataframe
        tuning_duration = (tuning_end_time - tuning_start_time).total_seconds() / 60.0
        tuned_params_df.loc[model_idx] = [dataset_name, model_name, cur_f1_score, cur_accuracy, tuning_duration, cur_params]
        models_config[model_name] = model_params['model'].set_params(**cur_params)

    return tuned_params_df, models_config


def test_ML_models(best_results_df, models_config, n_folds, X_train, y_train, X_test, y_test,
                   dataset_title, show_plots, debug_mode):
    """
    Find the best model from defined list.
    Tune each model on a validation set with GridSearchCV and
    return best_model with its hyperparameters, which has the highest F1 score
    """
    results_df = pd.DataFrame(columns=('Dataset_Name', 'Model_Name', 'F1_Score',
                                       'Accuracy_Score',
                                       'Model_Best_Params'))
    best_f1_score = -np.Inf
    best_accuracy = -np.Inf
    best_model_pred = []
    best_model_name = 'No model'
    best_params = None
    idx = 0
    # find the best model among defined in models_config
    for model_config in models_config:
        try:
            print(f"{datetime.now().strftime('%Y/%m/%d, %H:%M:%S')}: Tuning {model_config['model_name']}...")
            cur_model, cur_f1_score, cur_accuracy, cur_params = validate_model(deepcopy(model_config['model']),
                                                                               X_train, y_train, model_config['params'],
                                                                               n_folds)
            print(f'{datetime.now().strftime("%Y/%m/%d, %H:%M:%S")}: Tuning for {model_config["model_name"]} is finished')

            test_f1_score, test_accuracy, cur_model_pred = test_evaluation(cur_model, model_config['model_name'], cur_params,
                                                                           X_train, y_train, X_test, y_test, dataset_title, show_plots, debug_mode)
        except Exception as err:
            print(f"ERROR with {model_config['model_name']}: ", err)
            continue

        # save test results of each model in dataframe
        results_df.loc[idx] = [dataset_title,
                               model_config['model_name'],
                               test_f1_score,
                               test_accuracy,
                               cur_params]
        idx += 1

        if test_f1_score > best_f1_score:
            best_f1_score = test_f1_score
            best_accuracy = test_accuracy
            best_model_name = model_config['model_name']
            best_params = cur_params
            best_model_pred = cur_model_pred

    # append results of best model in best_results_df
    best_results_df.loc[best_results_df.shape[0]] = [dataset_title,
                                                     best_model_name,
                                                     best_f1_score,
                                                     best_accuracy,
                                                     best_params,
                                                     best_model_pred]

    return results_df, best_results_df
