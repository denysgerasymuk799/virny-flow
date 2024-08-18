import os

from virny_flow.utils.common_helpers import create_config_obj
from virny_flow.custom_classes.evaluation_scenario_builder import EvaluationScenarioBuilder
from virny_flow_demo.configs.datasets_config import DATASET_CONFIG


if __name__ == '__main__':
    # Read an experimental config
    exp_config_yaml_path = os.path.join('.', 'configs', 'exp_config.yaml')
    exp_config = create_config_obj(exp_config_yaml_path=exp_config_yaml_path)

    evaluation_scenario = [
        {'columns_to_inject': 'savings-account', 'error_type': 'missing_value', 'condition': '`checking-account` == "no account"', 'error_rate': 0.35},
        # {'columns_to_inject': 'checking-account,savings-account', 'error_type': 'missing_value', 'condition': '`checking-account` == "no account"', 'error_rate': 0.35},
        # {'columns_to_inject': 'duration', 'error_type': 'missing_value', 'condition': 'duration <= 20', 'error_rate': 0.30},
        # {'columns_to_inject': 'employment-since', 'error_type': 'missing_value', 'condition': '`employment-since` in ["<1 years", "unemployed"]', 'error_rate': 0.20},
    ]

    # Implement the defined evaluation scenario
    scenario_builder = EvaluationScenarioBuilder(exp_config=exp_config,
                                                 dataset_config=DATASET_CONFIG)
    scenario_builder.test_evaluation_scenario(evaluation_scenario_name='test_scenario',
                                              evaluation_scenario=evaluation_scenario,
                                              experiment_seed=exp_config.random_state)
