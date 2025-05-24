import sys
import argparse
import warnings
from pathlib import Path

# Suppress all warnings
warnings.filterwarnings("ignore")

# Define a correct root path
sys.path.append(str(Path(f"{__file__}").parent.parent))


from virny_flow.core.utils.common_helpers import create_exp_config_obj
from virny_flow.user_interfaces.worker_interface import worker_interface
from virny_flow.configs.component_configs import (NULL_IMPUTATION_CONFIG, FAIRNESS_INTERVENTION_CONFIG_SPACE,
                                                  get_models_params_for_tuning)

from scripts.configs.datasets_config import DATASET_CONFIG


if __name__ == '__main__':
    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_config_yaml_path",
                        type=str,
                        required=True,
                        help="Path to experiment config file")
    parser.add_argument("--kafka_broker_address",
                        type=str,
                        required=False,
                        default="localhost:9092",
                        help="Hostname and port of the Kafka broker")
    args = parser.parse_args()

    # Read an experimental config
    exp_config_yaml_path = args.exp_config_yaml_path
    exp_config = create_exp_config_obj(exp_config_yaml_path=exp_config_yaml_path)

    worker_interface(exp_config=exp_config,
                     dataset_config=DATASET_CONFIG,
                     null_imputation_config=NULL_IMPUTATION_CONFIG,
                     fairness_intervention_config=FAIRNESS_INTERVENTION_CONFIG_SPACE,
                     models_config=get_models_params_for_tuning(),
                     kafka_broker_address=args.kafka_broker_address)
