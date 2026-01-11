# pylint: disable=protected-access
"""Integration of automatic feature engineering."""

from alpine_meadow.common import TaskKeyword
import random

from alpine_meadow.utils import AMException


def automatic_feature_engineering(optimizer, dataset):
    """
    Apply automatic feature engineering.
    :param optimizer:
    :param dataset:
    :return:
    """

    if optimizer.state.time_limit is None:
        return

    # initialize feature engineering parameters
    import rl_feature_eng as fe

    logger = optimizer.logger
    logger.info("Feature engineering enabled!")
    fe_budget = optimizer.state.time_limit * optimizer.config.fe_budget_proportion
    if fe_budget <= 0:
        raise AMException(f"Feature engineering budget must be larger than 0: {fe_budget}")

    fe_hyperparams = optimizer.config.fe_hyperparams.copy()
    fe_hyperparams["test_time_budget"] = fe_budget

    task = optimizer.task
    task_type = optimizer.task.type
    if task_type == TaskKeyword.Value('CLASSIFICATION'):
        fe_type = "categorical"
    elif task_type == TaskKeyword.Value('REGRESSION'):
        fe_type = "regression"
    else:
        raise ValueError("Can't do FE with invalid FE dataset type!")

    # prepare train/test data
    res = fe.DirectInputInfoEncaser.DirectInputInfoEncaser(num_to_store=optimizer.config.fe_num_engineered_datasets)
    training_data = dataset.to_data_frame()
    target_data = training_data[task.target_columns]
    training_data = training_data.drop(task.target_columns, axis=1)

    # run feature engineering
    fe.Main.main(input_dataset=training_data,
                 input_test_target=target_data, input_test_target_type=fe_type,
                 res_storage=res, hyperparams=fe_hyperparams)

    # Retrieve new datasets from feature engineering output, with corresponding weights based on performance
    weighted_feature_sets = res.export_weighted_feature_sets()

    # Skip editing the search space if we didn't find any promising engineered features
    if weighted_feature_sets is not None:
        # Populate new search spaces with feature engineering pipelines
        fe_search_spaces = []
        for wfs in weighted_feature_sets:
            task._engineered_features = wfs[1]
            fe_search_spaces.append(optimizer.rule_executor.execute(task, optimizer.pipeline_history))

        # Select pipeline arms from new search spaces based on the given weights
        for i, ss in enumerate(fe_search_spaces):
            ss_weight = weighted_feature_sets[i][0]
            for arm in ss.pipeline_arms:
                if random.random() < ss_weight:
                    optimizer.search_space.add_pipeline_arm(arm)
