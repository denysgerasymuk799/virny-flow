import pathlib
from openbox import History
from datetime import datetime
from ConfigSpace import ConfigurationSpace
from openbox.utils.history import Observation

from virny_flow.configs.structs import BOAdvisorConfig
from virny_flow.core.utils.common_helpers import read_history_from_db
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
        if len(defined_objectives) == 3:
            obs["objectives"] = [obs["objectives"][0] / defined_objectives[0]['weight'],
                                 obs["objectives"][1] / defined_objectives[1]['weight'],
                                 obs["objectives"][2] / defined_objectives[2]['weight']]
        else:
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
    exp_config_name = 'case_studies_exp_folk_pubcov_cs2_w_acc_0_5_w_fair_0_5'
    lp_name = 'None&NO_FAIRNESS_INTERVENTION&lgbm_clf'
    run_num = 1
    max_trials = 100
    ref_point = [0.50, 0.15]

    # Read an experimental config
    db_secrets_path = pathlib.Path(__file__).parent.joinpath('configs').joinpath('secrets.env')

    # Prepare a History object
    bo_advisor_config = BOAdvisorConfig()
    config_space = create_config_space(lp_name)
    raw_history, defined_objectives, surrogate_model_type = read_history_from_db(secrets_path=db_secrets_path,
                                                                                 exp_config_name=exp_config_name,
                                                                                 lp_name=lp_name,
                                                                                 run_num=run_num,
                                                                                 ref_point=ref_point)
    history = prepare_history(data=raw_history,
                              config_space=config_space,
                              defined_objectives=defined_objectives)

    task_info = {
        'advisor_type': 'default',
        'max_runs': max_trials,
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
