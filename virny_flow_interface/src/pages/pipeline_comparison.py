import gradio as gr
import pandas as pd
from virny.configs.constants import *
import pprint

from scripts.configs.datasets_config import DATASET_CONFIG
from src.utils.common_utils import load_exp_config
from src.custom_classes.metrics_interactive_visualizer import MetricsInteractiveVisualizer
from src.utils.db_utils import get_tuned_lps_for_exp_config
from src.database.mongodb_client import MongoDBClient


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


def create_pipeline_comparison_page(exp_config_name: str = '', run_num: int = 0):
    """Create the Pipeline Performance page components"""
    gr.Markdown("# Pipeline Comparison")

    # Fetch all exp config names from MongoDB
    db_client = MongoDBClient()
    exp_config_names = db_client.get_all_exp_config_names()
    db_client.close()
    if not exp_config_names:
        gr.Markdown("No experiment configurations found in the database.")
        return
    if not exp_config_name:
        exp_config_name = exp_config_names[0]
    exp_config_name = str(exp_config_name)

    # Load config for the selected exp config name
    config = load_exp_config()
    dataset_name = config['pipeline_args']['dataset']
    data_loader = DATASET_CONFIG[dataset_name]['data_loader'](**DATASET_CONFIG[dataset_name]['data_loader_kwargs'])
    sensitive_attributes_dct = config['virny_args']['sensitive_attributes_dct']

    # Fetch logical pipeline names from logical_pipeline_scores for the default exp config
    initial_model_names_val = []
    if exp_config_name:
        db_client = MongoDBClient()
        lp_scores_df = db_client.get_logical_pipeline_scores(exp_config_name, run_num=1)
        db_client.close()
        if 'logical_pipeline_name' in lp_scores_df.columns:
            logical_names = [str(name) for name in sorted(lp_scores_df['logical_pipeline_name'].unique())]
            initial_model_names_val = [f"E1: {name}" for name in logical_names[:4]] if len(logical_names) >= 4 else [f"E1: {name}" for name in logical_names]

    # Add exp config selector
    with gr.Row():
        selected_exp_configs = gr.Dropdown(
            choices=exp_config_names,
            value=[exp_config_name],
            label="Experiment Configurations",
            info="Select one or more experiment configurations to compare pipelines.",
            multiselect=True,
            max_choices=len(exp_config_names)
        )

    # Markdown table component
    exp_config_table = gr.Markdown(value="")

    # Markdown table for best logical pipelines
    gr.Markdown("")
    gr.Markdown("## Best Logical Pipelines")
    best_lp_table = gr.Markdown(value="")

    # Helper to get alias for each exp config
    def get_alias(idx):
        return f"E{idx+1}"

    # Helper to get best logical pipeline for each exp config
    def get_best_logical_pipelines(exp_config_names):
        db_client = MongoDBClient()
        rows = []
        for exp_config in exp_config_names:
            lp_scores_df = db_client.get_logical_pipeline_scores(exp_config)
            if not lp_scores_df.empty and 'logical_pipeline_name' in lp_scores_df.columns and 'best_compound_pp_quality' in lp_scores_df.columns:
                # Find the row with the max best_compound_pp_quality
                best_row = lp_scores_df.loc[lp_scores_df['best_compound_pp_quality'].idxmax()]
                best_lp = best_row['logical_pipeline_name']
                best_score = round(best_row['best_compound_pp_quality'], 4)
                rows.append((exp_config, best_lp, best_score))
            else:
                rows.append((exp_config, '-', '-'))
        db_client.close()
        # Build markdown table
        table = "| Experiment Configuration | Best Logical Pipeline | Performance Score |\n|---|---|---|\n"
        for exp_config, best_lp, best_score in rows:
            table += f"| {exp_config} | {best_lp} | {best_score} |\n"
        return table

    # Helper to get concatenated metrics for selected exp configs
    def get_concat_metrics(selected_exp_configs):
        metrics_dfs = []
        for idx, exp_name in enumerate(selected_exp_configs):
            metrics_df = get_tuned_lps_for_exp_config(
                secrets_path=config['common_args']['secrets_path'],
                exp_config_name=exp_name
            )
            metrics_df = metrics_df.rename(columns={'model_name': 'Model_Name', 'metric': 'Metric'})
            metrics_df['Model_Name'] = metrics_df['logical_pipeline_name']
            alias = get_alias(idx)
            # Add alias to logical_pipeline_name values
            metrics_df['logical_pipeline_name'] = metrics_df['logical_pipeline_name'].apply(lambda x: f"{alias}: {x}")
            metrics_df['Model_Name'] = metrics_df['logical_pipeline_name']
            metrics_dfs.append(metrics_df)
        if metrics_dfs:
            model_metrics_df = pd.concat(metrics_dfs, ignore_index=True)
        else:
            model_metrics_df = pd.DataFrame()
        return model_metrics_df

    # Function to update the exp config table
    def make_exp_config_table(selected_names):
        if not selected_names:
            return ""
        table = "| Exp. Config Name | Alias |\n|---|---|\n"
        for idx, name in enumerate(selected_names):
            table += f"| {name} | E{idx+1} |\n"
        return table

    def update_exp_config_table(selected_names):
        return make_exp_config_table(selected_names)

    # UI components for metrics heatmap
    gr.Markdown("")
    gr.Markdown(
        """
        ## Metrics Heatmap
        Select input arguments to create a metrics heatmap.
        """)
    with gr.Row():
        with gr.Column(scale=1):
            tolerance = gr.Text(value="0.005", label="Tolerance", info="Define an acceptable tolerance for metric dense ranking.")
        with gr.Column(scale=1):
            gr.Markdown("")
    with gr.Row():
        model_names = gr.Dropdown(
            choices=initial_model_names_val, value=initial_model_names_val, max_choices=5, multiselect=True,
            label="Logical Pipelines", info="Select logical pipelines to display on the heatmap:",
        )
    with gr.Row():
        with gr.Column(scale=1):
            accuracy_metrics = gr.Dropdown(
                choices=[ACCURACY, F1],
                value=[ACCURACY, F1], multiselect=True, label="Correctness Metrics", info="Select correctness metrics to display on the heatmap:",
            )
            uncertainty_metrics = gr.Dropdown(
                choices=[ALEATORIC_UNCERTAINTY],
                value=[ALEATORIC_UNCERTAINTY], multiselect=True, label="Uncertainty Metrics", info="Select uncertainty metrics to display on the heatmap:",
            )
            subgroup_stability_metrics = gr.Dropdown(
                choices=[STD, LABEL_STABILITY],
                value=[STD, LABEL_STABILITY], multiselect=True, label="Stability Metrics", info="Select stability metrics to display on the heatmap:",
            )
            ranking_heatmap_btn_view2 = gr.Button("Submit")
        with gr.Column(scale=1):
            fairness_metrics_vw2 = gr.Dropdown(
                choices=[EQUALIZED_ODDS_FPR, EQUALIZED_ODDS_TPR],
                value=[EQUALIZED_ODDS_FPR, EQUALIZED_ODDS_TPR], multiselect=True, label="Error Disparity Metrics", info="Select error disparity metrics to display on the heatmap:",
            )
            group_uncertainty_metrics_vw2 = gr.Dropdown(
                choices=[ALEATORIC_UNCERTAINTY_DIFFERENCE],
                value=[ALEATORIC_UNCERTAINTY_DIFFERENCE], multiselect=True, label="Uncertainty Disparity Metrics", info="Select uncertainty disparity metrics to display on the heatmap:",
            )
            group_stability_metrics_vw2 = gr.Dropdown(
                choices=[LABEL_STABILITY_RATIO, STD_DIFFERENCE],
                value=[LABEL_STABILITY_RATIO, STD_DIFFERENCE], multiselect=True, label="Stability Disparity Metrics", info="Select stability disparity metrics to display on the heatmap:",
            )

    with gr.Row():
        model_ranking_heatmap = gr.Plot(label="Heatmap")

    def on_exp_config_selector_change(selected_exp_configs):
        # Update exp config table
        table_md = update_exp_config_table(selected_exp_configs)
        # Update best logical pipeline table
        best_lp_md = get_best_logical_pipelines(selected_exp_configs)
        # Update model_names dropdown (choices and value)
        model_metrics_df = get_concat_metrics(selected_exp_configs)
        visualizer = MetricsInteractiveVisualizer(
            X_data=data_loader.X_data,
            y_data=data_loader.y_data,
            model_metrics=model_metrics_df,
            sensitive_attributes_dct=sensitive_attributes_dct,
        )
        all_model_names = [str(name) for name in sorted(visualizer.model_names)]
        model_names_val = all_model_names[:4] if len(all_model_names) >= 4 else all_model_names
        return table_md, best_lp_md, gr.update(choices=list(all_model_names), value=list(model_names_val))

    selected_exp_configs.change(
        on_exp_config_selector_change,
        inputs=[selected_exp_configs],
        outputs=[exp_config_table, best_lp_table, model_names]
    )

    # Only update the heatmap when the Submit button is clicked
    def on_submit(selected_exp_configs, tolerance_val, acc_metrics, unc_metrics, stab_metrics, fair_metrics, group_unc_metrics, group_stab_metrics, model_names_val):
        model_metrics_df = get_concat_metrics(selected_exp_configs)
        visualizer = MetricsInteractiveVisualizer(
            X_data=data_loader.X_data,
            y_data=data_loader.y_data,
            model_metrics=model_metrics_df,
            sensitive_attributes_dct=sensitive_attributes_dct,
        )
        heatmap = visualizer._create_model_rank_heatmap(
            model_names_val, tolerance_val, acc_metrics, unc_metrics, stab_metrics,
            fair_metrics, group_unc_metrics, group_stab_metrics
        )
        return heatmap

    ranking_heatmap_btn_view2.click(
        on_submit,
        inputs=[selected_exp_configs, tolerance, accuracy_metrics, uncertainty_metrics, subgroup_stability_metrics, fairness_metrics_vw2, group_uncertainty_metrics_vw2, group_stability_metrics_vw2, model_names],
        outputs=[model_ranking_heatmap]
    )

    # ===================== System Configuration Section =====================
    gr.Markdown("")
    gr.Markdown("## System Configuration")

    # Helper to get config keys from exp_config.yaml (flattened)
    def flatten_dict(d, parent_key='', sep='.'):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    # Get all config keys from exp_config.yaml (flattened)
    config_keys = []
    if config is not None:
        config_flat = flatten_dict(config)
        # All keys from config
        config_keys = list(config_flat.keys())
        # Add all keys starting with 'virny.args' (if not already included)
        virny_args_keys = [k for k in config_flat.keys() if k.startswith('virny_args') or k.startswith('virny.args')]
        config_keys = list(dict.fromkeys(config_keys + virny_args_keys))
        # Add special keys
        config_keys += ['exp_config_execution_time', 'best_compound_pp_quality']
        # Exclude all keys starting with 'common_args.'
        config_keys = [k for k in config_keys if not k.startswith('common_args.')]

    # Always include these keys from exp_config_history if not present
    extra_virny_keys = [
        'virny_args.bootstrap_fraction',
        'virny_args.n_estimators',
        'virny_args.computation_mode',
    ]
    for k in extra_virny_keys:
        if k not in config_keys:
            config_keys.append(k)

    def get_system_config_df(selected_exp_configs, config_keys=config_keys, selected_keys=None):
        config_keys = config_keys or []
        if selected_keys is not None:
            config_keys = [k for k in config_keys if k in selected_keys]
        if selected_exp_configs is None:
            selected_exp_configs = []
        if isinstance(selected_exp_configs, str):
            selected_exp_configs = [selected_exp_configs]
        if not selected_exp_configs:
            # Return an empty DataFrame with config_keys as index and no columns
            df = pd.DataFrame(index=config_keys)
            df.index.name = 'Config Key'
            return df.reset_index()
        db_client = MongoDBClient()
        records = []
        for exp_name in selected_exp_configs:
            # Get the latest record for this exp_config_name
            doc = db_client.db['exp_config_history'].find_one({'exp_config_name': exp_name}, sort=[('create_datetime', -1)])
            # Get exp_config_execution_time and best_compound_pp_quality from doc
            exp_config_execution_time = None
            best_compound_pp_quality = '-'
            if doc is not None:
                exp_config_execution_time = doc.get('exp_config_execution_time', '-')
                best_compound_pp_quality = doc.get('best_compound_pp_quality', '-')
            # Flatten the document and filter keys
            if doc is None:
                filtered = {k: '-' for k in config_keys}
            else:
                flat_doc = flatten_dict(doc)
                filtered = {}
                for k in config_keys:
                    if k == 'exp_config_execution_time':
                        v = exp_config_execution_time
                    elif k == 'best_compound_pp_quality':
                        v = best_compound_pp_quality
                    else:
                        v = flat_doc.get(k, '-')
                    if k == 'optimisation_args.objectives' and v != '-':
                        filtered[k] = pprint.pformat(v, indent=4, width=40)
                    elif isinstance(v, list):
                        filtered[k] = str(v)
                    elif isinstance(v, dict):
                        filtered[k] = str(v)
                    else:
                        filtered[k] = v
            records.append(filtered)
        db_client.close()
        df = pd.DataFrame(records, index=selected_exp_configs).T
        df.index.name = 'Config Key'
        return df.reset_index()

    # Dropdown for config key selection
    config_key_selector = gr.Dropdown(
        choices=config_keys,
        value=config_keys,
        label="Config Keys to Display",
        info="Select which config keys to show in the table.",
        multiselect=True,
        max_choices=len(config_keys)
    )

    # System config DataFrame component
    def get_column_widths(num_exp_configs):
        # First column (Config Key) gets 200px, rest are distributed evenly
        if num_exp_configs == 0:
            return ["200px"]
        if num_exp_configs == 1:
            return ["200px", "200px"]
        first_col = "200px"
        rest_width = f"{int(600 / num_exp_configs)}px"
        rest = [rest_width for _ in range(num_exp_configs)]
        return [first_col] + rest

    system_config_df = gr.DataFrame(
        value=get_system_config_df([exp_config_name], config_keys=config_keys, selected_keys=config_keys),
        interactive=False,
        wrap=True,
        column_widths=get_column_widths(1),
    )

    def on_exp_config_selector_change_with_system_config(selected_exp_configs, selected_keys):
        table_md = update_exp_config_table(selected_exp_configs)
        best_lp_md = get_best_logical_pipelines(selected_exp_configs)
        model_metrics_df = get_concat_metrics(selected_exp_configs)
        visualizer = MetricsInteractiveVisualizer(
            X_data=data_loader.X_data,
            y_data=data_loader.y_data,
            model_metrics=model_metrics_df,
            sensitive_attributes_dct=sensitive_attributes_dct,
        )
        all_model_names = [str(name) for name in sorted(visualizer.model_names)]
        model_names_val = all_model_names[:4] if len(all_model_names) >= 4 else all_model_names
        sys_config_df = get_system_config_df(selected_exp_configs, config_keys=config_keys, selected_keys=selected_keys)
        column_widths = get_column_widths(len(selected_exp_configs))
        return table_md, best_lp_md, gr.update(choices=list(all_model_names), value=list(model_names_val)), gr.update(value=sys_config_df, column_widths=column_widths)

    # Update outputs to include system config table and config key selector
    selected_exp_configs.change(
        on_exp_config_selector_change_with_system_config,
        inputs=[selected_exp_configs, config_key_selector],
        outputs=[exp_config_table, best_lp_table, model_names, system_config_df]
    )
    def on_config_key_selector_change(selected_keys, selected_exp_configs):
        return gr.update(
            value=get_system_config_df(selected_exp_configs, config_keys=config_keys, selected_keys=selected_keys),
            column_widths=get_column_widths(len(selected_exp_configs) if selected_exp_configs else 1)
        )
    config_key_selector.change(
        on_config_key_selector_change,
        inputs=[config_key_selector, selected_exp_configs],
        outputs=[system_config_df]
    )

    # Show initial tables
    exp_config_table.value = make_exp_config_table([exp_config_name])
    best_lp_table.value = get_best_logical_pipelines([exp_config_name])


# For standalone testing
if __name__ == "__main__":
    # Load configuration from YAML file
    config = load_exp_config()
    # Extract parameters from config
    exp_config_name = config['common_args']['exp_config_name']
    run_num = config['common_args']['run_nums'][0]  # Take first run number
    with gr.Blocks() as demo:
        create_pipeline_comparison_page(
            exp_config_name=str(exp_config_name),
            run_num=int(run_num)
        )
    demo.launch()
