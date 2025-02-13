import os
import json
import pathlib
from openbox import History
from datetime import datetime
from ConfigSpace import ConfigurationSpace
from openbox.utils.history import Observation

from virny_flow.configs.structs import BOAdvisorConfig
from virny_flow.core.utils.common_helpers import create_exp_config_obj, read_history_from_db
from virny_flow.visualizations.viz_utils import build_visualizer, create_config_space


def prepare_history(data: dict, config_space: ConfigurationSpace, defined_objectives: list) -> 'History':
    """
    Prepare history object from raw data read from the database.

    Args:
        data (dict): Data read from the database.
        config_space (ConfigurationSpace): Configuration space object.
        defined_objectives (list): List of defined objectives.

    Returns:
        History: History object for visualizer.
    """

    # Get original losses from weighted losses
    for obs in data["observations"]:
        obs["objectives"] = [obs["objectives"][0] / defined_objectives[0]['weight'],
                             obs["objectives"][1] / defined_objectives[1]['weight']]

    global_start_time = data.pop('global_start_time')
    global_start_time = datetime.fromisoformat(global_start_time)
    observations = data.pop('observations')
    observations = [Observation.from_dict(obs, config_space) for obs in observations]

    history = History(**data)
    history.global_start_time = global_start_time
    history.update_observations(observations)

    return history


if __name__ == '__main__':
    # Input variables
    exp_config_name = 'case_studies_exp_folk_emp_cs1_w_acc_0_25_w_fair_0_75'
    lp_name = 'None&NO_FAIRNESS_INTERVENTION&lgbm_clf'
    run_num = 2

    # Read an experimental config
    exp_config_yaml_path = pathlib.Path(__file__).parent.joinpath('configs').joinpath('exp_config.yaml')
    exp_config = create_exp_config_obj(exp_config_yaml_path=exp_config_yaml_path)
    db_secrets_path = pathlib.Path(__file__).parent.joinpath('configs').joinpath('secrets.env')

    # Prepare a History object
    bo_advisor_config = BOAdvisorConfig()
    config_space = create_config_space(lp_name)
    raw_history, defined_objectives, surrogate_model_type = read_history_from_db(db_secrets_path, exp_config_name, lp_name, run_num)
    history = prepare_history(data=raw_history, 
                              config_space=config_space,
                              defined_objectives=defined_objectives)

    task_info = {
        'advisor_type': 'default',
        'max_runs': exp_config.optimisation_args.max_trials,
        'max_runtime_per_trial': bo_advisor_config.max_runtime_per_trial,
        'surrogate_type': surrogate_model_type,
        'constraint_surrogate_type': None,
        'transfer_learning_history': bo_advisor_config.transfer_learning_history,
    }

    visualizer = build_visualizer(
        task_info=task_info,
        option='advanced',
        history=history,
        auto_open_html=True,
    )
    visualizer.setup()
    visualizer.update()
