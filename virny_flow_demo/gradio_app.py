import os

import pandas as pd
from virny.custom_classes.metrics_interactive_visualizer import MetricsInteractiveVisualizer

from virny_flow.configs.constants import S3Folder, NO_FAIRNESS_INTERVENTION
from virny_flow.custom_classes.s3_client import S3Client
from virny_flow.custom_classes.database_client import DatabaseClient
from virny_flow.utils.common_helpers import create_config_obj
from virny_flow_demo.configs.datasets_config import DATASET_CONFIG


def read_model_metrics_from_s3(exp_config, evaluation_scenario_name: str):
    db = DatabaseClient(exp_config.secrets_path)
    db.connect()

    s3_client = S3Client(exp_config.secrets_path)
    save_sets_dir_path = f'{S3Folder.experiments.value}/{exp_config.exp_config_name}/{S3Folder.evaluation_scenarios.value}/{evaluation_scenario_name}'

    # Read pipeline names from the database
    pipeline_names = db.read_pipeline_names(exp_config_name=exp_config.exp_config_name)

    all_metrics_df = pd.DataFrame()
    for pipeline_name in pipeline_names:
        # Read metrics of the pipeline from S3 as a CSV
        metrics_filename = f'{pipeline_name}_metrics.csv'
        pipeline_metrics_df = s3_client.read_csv( f'{save_sets_dir_path}/{metrics_filename}', index=True)
        pipeline_metrics_df['Model_Name'] = pipeline_name.replace(NO_FAIRNESS_INTERVENTION, 'NO_FI')
        all_metrics_df = pd.concat([all_metrics_df, pipeline_metrics_df])

    db.close()

    return all_metrics_df


if __name__ == '__main__':
    # Read an experimental config
    # exp_config_yaml_path = os.path.join('.', 'configs', 'exp_config.yaml')
    exp_config_yaml_path = os.path.join('virny_flow_demo', 'configs', 'exp_config.yaml')
    exp_config = create_config_obj(exp_config_yaml_path=exp_config_yaml_path)

    evaluation_scenario_name = 'test_scenario'
    data_loader = DATASET_CONFIG[exp_config.dataset]['data_loader'](**DATASET_CONFIG[exp_config.dataset]['data_loader_kwargs'])
    virny_config = create_config_obj(DATASET_CONFIG[exp_config.dataset]['virny_config_path'])
    all_metrics_df = read_model_metrics_from_s3(exp_config, evaluation_scenario_name)

    interactive_metrics_visualizer = MetricsInteractiveVisualizer(X_data=data_loader.X_data,
                                                                  y_data=data_loader.y_data,
                                                                  model_metrics=all_metrics_df,
                                                                  sensitive_attributes_dct=virny_config.sensitive_attributes_dct)
    interactive_metrics_visualizer.create_web_app()
