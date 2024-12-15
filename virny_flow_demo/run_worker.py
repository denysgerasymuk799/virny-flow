import pathlib
from virny_flow.core.utils.common_helpers import create_exp_config_obj
from virny_flow.user_interfaces.worker_interface import worker_interface
from virny_flow.configs.component_configs import (NULL_IMPUTATION_CONFIG, FAIRNESS_INTERVENTION_CONFIG_SPACE,
                                                  get_models_params_for_tuning)

from virny_flow_demo.configs.datasets_config import DATASET_CONFIG


if __name__ == '__main__':
    # Read an experimental config
    exp_config_yaml_path = pathlib.Path(__file__).parent.joinpath('configs').joinpath('exp_config.yaml')
    exp_config = create_exp_config_obj(exp_config_yaml_path=exp_config_yaml_path)

    worker_interface(exp_config=exp_config,
                     virny_flow_address="http://127.0.0.1:8000",
                     dataset_config=DATASET_CONFIG,
                     null_imputation_config=NULL_IMPUTATION_CONFIG,
                     fairness_intervention_config=FAIRNESS_INTERVENTION_CONFIG_SPACE,
                     models_config=get_models_params_for_tuning())
