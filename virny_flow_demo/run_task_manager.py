import pathlib
from virny_flow.task_manager import TaskManager
from virny_flow.configs.structs import BOAdvisorConfig
from virny_flow.core.utils.common_helpers import create_exp_config_obj


if __name__ == "__main__":
    bo_advisor_config = BOAdvisorConfig()

    # Read an experimental config
    exp_config_yaml_path = pathlib.Path(__file__).parent.joinpath('configs').joinpath('exp_config.yaml')
    exp_config = create_exp_config_obj(exp_config_yaml_path=exp_config_yaml_path)

    task_manager = TaskManager(secrets_path=pathlib.Path(__file__).parent.joinpath('configs').joinpath('secrets.env'),
                               host='127.0.0.1',
                               port=8000,
                               exp_config=exp_config,
                               bo_advisor_config=bo_advisor_config)
    task_manager.run()
