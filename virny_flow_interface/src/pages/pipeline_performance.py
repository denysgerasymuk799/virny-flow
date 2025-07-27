import gradio as gr
from virny.configs.constants import *

from scripts.configs.datasets_config import DATASET_CONFIG
from src.utils.common_utils import load_exp_config
from src.custom_classes.metrics_interactive_visualizer import MetricsInteractiveVisualizer
from src.utils.db_utils import get_tuned_lps_for_exp_config


def generate_lp_name_combinations(config):
    """
    Generate all possible learning pipeline name combinations from config.
    
    Args:
        config (dict): Configuration dictionary from YAML
        
    Returns:
        list: List of all possible lp_name combinations
    """
    pipeline_args = config.get('pipeline_args', {})
    
    # Get components from config
    null_imputers = pipeline_args.get('null_imputers', [])
    fairness_interventions = pipeline_args.get('fairness_interventions', [])
    models = pipeline_args.get('models', [])
    
    null_imputers += ['None']
    fairness_interventions += ['NO_FAIRNESS_INTERVENTION']
    
    # Generate all combinations
    combinations = []
    for null_imp in null_imputers:
        for fair_int in fairness_interventions:
            for model in models:
                lp_name = f"{null_imp}&{fair_int}&{model}"
                combinations.append(lp_name)
    
    return combinations


def create_pipeline_performance_page(exp_config_name: str, run_num: int):
    """Create the Pipeline Performance page components"""
    
    # Load config to get lp_name combinations
    config = load_exp_config()
    dataset_name = config['pipeline_args']['dataset']
    
    # Default lp_name (first combination or a specific one)
    data_loader = DATASET_CONFIG[dataset_name]['data_loader'](**DATASET_CONFIG[dataset_name]['data_loader_kwargs'])
    sensitive_attributes_dct = config['virny_args']['sensitive_attributes_dct']
    model_metrics_df = get_tuned_lps_for_exp_config(secrets_path=config['common_args']['secrets_path'],
                                                    exp_config_name=config['common_args']['exp_config_name'])
    model_metrics_df.rename(columns={'model_name': 'Model_Name', 'metric': 'Metric'}, inplace=True)
    model_metrics_df['Model_Name'] = model_metrics_df['logical_pipeline_name'] # Replace Model_Name with logical_pipeline_name to reuse MetricsInteractiveVisualizer
    visualizer = MetricsInteractiveVisualizer(
        X_data=data_loader.X_data,
        y_data=data_loader.y_data,
        model_metrics=model_metrics_df,
        sensitive_attributes_dct=sensitive_attributes_dct,
    )
    
    gr.Markdown("# Pipeline Performance")

    # ============================ Group Specific and Disparity Metrics Bar Charts ============================
    gr.Markdown(
        """
        ## Group Specific and Disparity Metrics Bar Charts
        """)
    model_name_vw3 = gr.Dropdown(
        sorted(visualizer.model_names), value=sorted(visualizer.model_names)[0], multiselect=False, scale=1,
        label="Select Learning Pipeline", info="Choose the learning pipeline configuration to visualize:",
    )
    with gr.Row():
        with gr.Column():
            gr.Markdown(
                """
                ### Group Specific Metrics
                """)
            accuracy_metrics = gr.Dropdown(
                sorted(visualizer.all_accuracy_metrics),
                value=[ACCURACY, F1], multiselect=True, label="Correctness Metrics", info="Select correctness metrics to display on the heatmap:",
            )
            uncertainty_metrics = gr.Dropdown(
                sorted(visualizer.all_uncertainty_metrics),
                value=[ALEATORIC_UNCERTAINTY], multiselect=True, label="Uncertainty Metrics", info="Select uncertainty metrics to display on the heatmap:",
            )
            subgroup_stability_metrics = gr.Dropdown(
                sorted(visualizer.all_stability_metrics),
                value=[STD, LABEL_STABILITY], multiselect=True, label="Stability Metrics", info="Select stability metrics to display on the heatmap:",
            )
            btn_view3 = gr.Button("Submit")
        with gr.Column():
            gr.Markdown(
                """
                ### Disparity Metrics
                """)
            fairness_metrics_vw3 = gr.Dropdown(
                sorted(visualizer.all_error_disparity_metrics),
                value=[EQUALIZED_ODDS_FPR, EQUALIZED_ODDS_TPR], multiselect=True, label="Error Disparity Metrics", info="Select error disparity metrics to display on the heatmap:",
            )
            group_uncertainty_metrics_vw3 = gr.Dropdown(
                sorted(visualizer.all_uncertainty_disparity_metrics),
                value=[ALEATORIC_UNCERTAINTY_RATIO], multiselect=True, label="Uncertainty Disparity Metrics", info="Select uncertainty disparity metrics to display on the heatmap:",
            )
            group_stability_metrics_vw3 = gr.Dropdown(
                sorted(visualizer.all_stability_disparity_metrics),
                value=[LABEL_STABILITY_RATIO, STD_DIFFERENCE], multiselect=True, label="Stability Disparity Metrics", info="Select stability disparity metrics to display on the heatmap:",
            )
    with gr.Row():
        with gr.Column():
            subgroup_metrics_bar_chart = gr.Plot(label="Group Specific Bar Chart")
        with gr.Column():
            group_metrics_bar_chart = gr.Plot(label="Disparity Bar Chart")

    btn_view3.click(visualizer._create_subgroup_metrics_bar_chart_per_one_model,
                    inputs=[model_name_vw3, accuracy_metrics, uncertainty_metrics, subgroup_stability_metrics],
                    outputs=[subgroup_metrics_bar_chart])
    btn_view3.click(visualizer._create_group_metrics_bar_chart_per_one_model,
                    inputs=[model_name_vw3, fairness_metrics_vw3, group_uncertainty_metrics_vw3, group_stability_metrics_vw3],
                    outputs=[group_metrics_bar_chart])

    # ==================================== Dataset Statistics ====================================
    gr.Markdown("")
    gr.Markdown("")
    gr.Markdown(
        """
        ## Dataset Statistics
        """)
    with gr.Row():
        with gr.Column(scale=2):
            default_val = 5
            s = gr.Slider(1, visualizer.max_groups, value=default_val, step=1, label="How many groups to show:")
            grp_names = []
            grp_dis_values = []
            sensitive_attr_items = list(visualizer.sensitive_attributes_dct.items())
            for i in range(visualizer.max_groups):
                visibility = True if i + 1 <= default_val else False
                with gr.Row():
                    if visibility and i + 1 <= len(sensitive_attr_items):
                        grp, dis_value = sensitive_attr_items[i]
                        if dis_value is None:
                            dis_value = '-'
                        elif isinstance(dis_value, str):
                            dis_value = f"'{dis_value}'"
                        grp_name = gr.Text(label=f"Group {i + 1}", value=grp, interactive=True, visible=visibility)
                        grp_dis_value = gr.Text(label="Disadvantage val.", value=dis_value, interactive=True, visible=visibility)
                    else:
                        grp_name = gr.Text(label=f"Group {i + 1}", interactive=True, visible=visibility)
                        grp_dis_value = gr.Text(label="Disadvantage val.", interactive=True, visible=visibility)
                grp_names.append(grp_name)
                grp_dis_values.append(grp_dis_value)

            s.change(visualizer._variable_inputs, s, grp_names)
            s.change(visualizer._variable_inputs, s, grp_dis_values)
            btn_view0 = gr.Button("Submit")
        with gr.Column(scale=4):
            dataset_proportions_bar_chart = gr.Plot(label="Group Proportions and Base Rates")

    btn_view0.click(visualizer._create_dataset_proportions_bar_chart,
                    inputs=[grp_names[0], grp_names[1], grp_names[2], grp_names[3], grp_names[4], grp_names[5], grp_names[6], grp_names[7],
                            grp_dis_values[0], grp_dis_values[1], grp_dis_values[2], grp_dis_values[3], grp_dis_values[4], grp_dis_values[5], grp_dis_values[6], grp_dis_values[7]],
                    outputs=[dataset_proportions_bar_chart])


# For standalone testing
if __name__ == "__main__":
    # Load configuration from YAML file
    config = load_exp_config()
    
    # Extract parameters from config
    exp_config_name = config['common_args']['exp_config_name']
    run_num = config['common_args']['run_nums'][0]  # Take first run number
    
    with gr.Blocks() as demo:
        create_pipeline_performance_page(
            exp_config_name=exp_config_name,
            run_num=run_num
        )
    demo.launch()
