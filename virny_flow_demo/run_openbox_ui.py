import pathlib
from typing import Union
from openbox import History
from openbox.visualization.base_visualizer import _parse_option, NullVisualizer
from openbox.utils.config_space import ConfigurationSpace

from virny_flow.core.utils.common_helpers import create_exp_config_obj
from virny_flow.configs.constants import StageName, NO_FAIRNESS_INTERVENTION, STAGE_SEPARATOR
from virny_flow.configs.structs import BOAdvisorConfig
from virny_flow.configs.component_configs import (NULL_IMPUTATION_CONFIG, FAIRNESS_INTERVENTION_CONFIG_SPACE,
                                                  get_models_params_for_tuning)


def build_visualizer(
        option: Union[str, bool],
        history: History,
        *,
        logging_dir='logs/',
        task_info=None,
        **kwargs,
):
    """
    Build visualizer for optimizer.

    Parameters
    ----------
    option : ['none', 'basic', 'advanced']
        Visualizer option.
    history : History
        History to visualize.
    logging_dir : str, default='logs/'
        The directory to save the visualization.
    task_info : dict, optional
        Task information for visualizer to use.
    optimizer : Optimizer, optional
        Optimizer to extract task_info from.
    advisor : Advisor, optional
        Advisor to extract task_info from.
    kwargs : dict
        Other arguments for visualizer.
        For HTMLVisualizer, available arguments are:
        - auto_open_html : bool, default=False
            Whether to open html file automatically.
        - advanced_analysis_options : dict, default=None
            Advanced analysis options. See `HTMLVisualizer` for details.

    Returns
    -------
    visualizer : BaseVisualizer
        Visualizer.
    """
    option = _parse_option(option)

    if option == 'none':
        visualizer = NullVisualizer()
    elif option in ['basic', 'advanced']:
        from openbox.visualization.html_visualizer import HTMLVisualizer
        visualizer = HTMLVisualizer(
            logging_dir=logging_dir,
            history=history,
            task_info=task_info,
            auto_open_html=kwargs.get('auto_open_html', False),
            advanced_analysis=(option == 'advanced'),
            advanced_analysis_options=kwargs.get('advanced_analysis_options'),
        )
    else:
        raise ValueError('Unknown visualizer option: %s' % option)

    return visualizer


def create_config_space(logical_pipeline_name: str):
    # Create a config space based on config spaces of each stage
    config_space = ConfigurationSpace()
    lp_stages = logical_pipeline_name.split(STAGE_SEPARATOR)
    components = {
        StageName.null_imputation.value: lp_stages[0],
        StageName.fairness_intervention.value: lp_stages[1],
        StageName.model_evaluation.value: lp_stages[2],
    }
    for stage in StageName:
        if stage == StageName.null_imputation:
            stage_config_space = NULL_IMPUTATION_CONFIG[components[stage.value]]['config_space']
        elif stage == StageName.fairness_intervention:
            # NO_FAIRNESS_INTERVENTION does not have config space
            if components[stage.value] == NO_FAIRNESS_INTERVENTION:
                continue

            stage_config_space = FAIRNESS_INTERVENTION_CONFIG_SPACE[components[stage.value]]
        else:
            # We set seed to None since we need only a config space
            stage_config_space = get_models_params_for_tuning(models_tuning_seed=None)[components[stage.value]]['config_space']

        config_space.add_hyperparameters(list(stage_config_space.values()))

    return config_space


if __name__ == '__main__':
    # Input variables
    # lp_name = 'miss_forest&DIR&lr_clf'
    # history_filename = 'history_2024-11-21-00-39-13-139162.json'
    lp_name = 'miss_forest&NO_FAIRNESS_INTERVENTION&lr_clf'
    history_filename = 'history_2024-11-21-00-39-13-139162.json'
    surrogate_model_type = 'gp'  # 'gp' or 'prf'

    bo_advisor_config = BOAdvisorConfig()

    # Read an experimental config
    exp_config_yaml_path = pathlib.Path(__file__).parent.joinpath('configs').joinpath('exp_config.yaml')
    exp_config = create_exp_config_obj(exp_config_yaml_path=exp_config_yaml_path)

    config_space = create_config_space(lp_name)
    history_path = f'../history/{exp_config.exp_config_name}/{lp_name}/' + history_filename
    history = History.load_json(history_path, config_space)

    task_info = {
        'advisor_type': 'default',
        'max_runs': exp_config.max_trials,
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
