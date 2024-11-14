import numpy as np
from virny_flow.core.custom_classes.evaluation_scenario_builder import EvaluationScenarioBuilder


def test_parsing_in_evaluation_scenario_builder(german_data_loader, common_seed):
    df = german_data_loader.X_data
    evaluation_scenario = [
        {'columns_to_inject': 'checking-account,savings-account', 'error_type': 'missing_value', 'condition': '`checking-account` == "no account"', 'error_rate': 0.35},
        {'columns_to_inject': 'duration', 'error_type': 'missing_value', 'condition': 'duration <= 20', 'error_rate': 0.30},
        {'columns_to_inject': 'employment-since', 'error_type': 'missing_value', 'condition': '`employment-since` in ["<1 years", "unemployed"]', 'error_rate': 0.20},
    ]
    scenario_builder = EvaluationScenarioBuilder(exp_config_name='exp_config_name',
                                                 dataset_name='dataset_name',
                                                 seed=common_seed)
    df_with_nulls = scenario_builder.implement_evaluation_scenario(df=df, evaluation_scenario=evaluation_scenario)

    # Check error injection
    mask1_2 = df["checking-account"] == "no account"

    print('df.shape --', df.shape)
    print('df[mask1_2].shape --', df[mask1_2].shape)
    print("df_with_nulls[mask1_2]['checking-account'].isnull().sum() --", df_with_nulls[mask1_2]['checking-account'].isnull().sum())
    print("mask1_2.sum() --", mask1_2.sum())

    np.testing.assert_almost_equal(df_with_nulls[mask1_2]['checking-account'].isnull().sum() / mask1_2.sum(), evaluation_scenario[0]['error_rate'], decimal=3, err_msg="Injection scenario is not satisfied")
