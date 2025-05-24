import sys
import argparse
import warnings
from pathlib import Path

# Suppress all warnings
warnings.filterwarnings("ignore")

# Define a correct root path
sys.path.append(str(Path(f"{__file__}").parent.parent))

from virny_flow.task_manager import TaskManager
from virny_flow.core.utils.common_helpers import create_exp_config_obj


if __name__ == "__main__":
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

    task_manager = TaskManager(secrets_path=exp_config.common_args.secrets_path,
                               host='127.0.0.1',
                               port=8000,
                               kafka_broker_address=args.kafka_broker_address,
                               exp_config=exp_config)
    task_manager.run()
