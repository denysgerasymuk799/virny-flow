import os
import json
import pathlib
from openbox import History
from datetime import datetime
from ConfigSpace import ConfigurationSpace
from openbox.utils.history import Observation

from virny_flow.configs.structs import BOAdvisorConfig
from virny_flow.core.utils.common_helpers import create_exp_config_obj
from virny_flow.visualizations.viz_utils import build_visualizer, create_config_space


def load_history(filename: str, config_space: ConfigurationSpace, defined_objectives: list) -> 'History':
    if not os.path.exists(filename):
        raise FileNotFoundError(f'File not found: {filename}')
    with open(filename, 'r') as f:
        data = json.load(f)

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
    exp_config_name = 'cost_model_exp1_folk_emp_w_acc_0_25_w_fair_0_75_w_openbox_weights'
    defined_objectives = [
        { "name": "objective_1", "metric": "F1", "group": "overall", "weight": 0.25 },
        { "name": "objective_2", "metric": "Equalized_Odds_TPR", "group": "SEX&RAC1P", "weight": 0.75 }
    ]
    lp_name = 'None&DIR&lgbm_clf'
    run_num = 2
    history_filename = 'history_2024-12-15-18-30-18-476914.json'
    surrogate_model_type = 'prf'  # 'gp' or 'prf'
    bo_advisor_config = BOAdvisorConfig()

    # Read an experimental config
    exp_config_yaml_path = pathlib.Path(__file__).parent.joinpath('configs').joinpath('exp_config.yaml')
    exp_config = create_exp_config_obj(exp_config_yaml_path=exp_config_yaml_path)

    config_space = create_config_space(lp_name)
    history_path = f'../history/{exp_config.common_args.exp_config_name}/run_num_{str(run_num)}/{lp_name}/' + history_filename
    history = load_history(history_path, config_space, defined_objectives)

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
