import pathlib
from virny.custom_classes.metrics_interactive_visualizer import MetricsInteractiveVisualizer

from virny_flow.core.utils.common_helpers import create_exp_config_obj
from virny_flow.visualizations.use_case_queries import prepare_metrics_for_virnyview
from virny_flow_demo.configs.datasets_config import DATASET_CONFIG


if __name__ == '__main__':
    # Read an experimental config
    # exp_config_yaml_path = pathlib.Path(__file__).parent.joinpath('configs').joinpath('exp_config.yaml')
    exp_config_yaml_path = pathlib.Path(__file__).parent.joinpath('configs').joinpath('exp_config_folk_emp.yaml')
    exp_config = create_exp_config_obj(exp_config_yaml_path=exp_config_yaml_path)

    data_loader = DATASET_CONFIG[exp_config.pipeline_args.dataset]['data_loader'](**DATASET_CONFIG[exp_config.pipeline_args.dataset]['data_loader_kwargs'])
    virny_config = exp_config.virny_args
    all_metrics_df = prepare_metrics_for_virnyview(secrets_path=exp_config.common_args.secrets_path,
                                                   exp_config_name=exp_config.common_args.exp_config_name,
                                                   groups=list(virny_config.sensitive_attributes_dct.keys()))

    interactive_metrics_visualizer = MetricsInteractiveVisualizer(X_data=data_loader.X_data,
                                                                  y_data=data_loader.y_data,
                                                                  model_metrics=all_metrics_df,
                                                                  sensitive_attributes_dct=virny_config.sensitive_attributes_dct)
    interactive_metrics_visualizer.create_web_app()
