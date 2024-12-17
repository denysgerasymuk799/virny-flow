from typing import Union
from openbox import History
from openbox.visualization.base_visualizer import _parse_option, NullVisualizer
from openbox.utils.config_space import ConfigurationSpace

from virny_flow.configs.constants import StageName, NO_FAIRNESS_INTERVENTION, STAGE_SEPARATOR
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
            # None imputer does not have config space
            if components[stage.value] == 'None':
                continue

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
