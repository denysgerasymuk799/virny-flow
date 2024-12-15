import pathlib
from openbox import History

from virny_flow.configs.structs import BOAdvisorConfig
from virny_flow.core.utils.common_helpers import create_exp_config_obj
from virny_flow.visualizations.viz_utils import build_visualizer, create_config_space


if __name__ == '__main__':
    # Input variables
    lp_name = 'miss_forest&DIR&lr_clf'
    run_num = 1
    history_filename = 'history_2024-12-14-23-54-45-330002.json'
    surrogate_model_type = 'gp'  # 'gp' or 'prf'
    bo_advisor_config = BOAdvisorConfig()

    # Read an experimental config
    exp_config_yaml_path = pathlib.Path(__file__).parent.joinpath('configs').joinpath('exp_config.yaml')
    exp_config = create_exp_config_obj(exp_config_yaml_path=exp_config_yaml_path)

    config_space = create_config_space(lp_name)
    history_path = f'../history/{exp_config.common_args.exp_config_name}/run_num_{str(run_num)}/{lp_name}/' + history_filename
    history = History.load_json(history_path, config_space)

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
