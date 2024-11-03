import os
from virny_flow.core.utils.common_helpers import create_config_obj
from virny_flow.user_interfaces.worker_interface import worker_interface

from virny_flow_demo.configs.datasets_config import DATASET_CONFIG
from virny_flow_demo.configs.params_for_tuning import get_models_params_for_tuning, FAIRNESS_INTERVENTION_HYPERPARAMS


if __name__ == '__main__':
    # Read an experimental config
    exp_config_yaml_path = os.path.join('.', 'configs', 'exp_config.yaml')
    # exp_config_yaml_path = os.path.join('virny_flow_demo', 'configs', 'exp_config.yaml')
    exp_config = create_config_obj(exp_config_yaml_path=exp_config_yaml_path)
    worker_interface(exp_config=exp_config,
                     virny_flow_address="http://127.0.0.1:8000",
                     dataset_config=DATASET_CONFIG,
                     fairness_intervention_config=FAIRNESS_INTERVENTION_HYPERPARAMS,
                     models_config=get_models_params_for_tuning(exp_config.random_state))
