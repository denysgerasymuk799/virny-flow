import gradio as gr
import pathlib
import yaml
import pandas as pd
import altair as alt
import numpy as np
from openbox import History
from datetime import datetime
from ConfigSpace import ConfigurationSpace
from openbox.utils.history import Observation
import json
import plotly.graph_objects as go
from scipy.spatial import ConvexHull
import plotly.express as px  # <-- Add this import

from virny_flow.configs.structs import BOAdvisorConfig
from virny_flow.visualizations.viz_utils import build_visualizer, create_config_space
from src.utils.db_utils import read_history_from_db, get_tuned_lp_for_exp_config
from src.configs.component_configs import get_models_params_for_tuning, NULL_IMPUTATION_CONFIG, FAIRNESS_INTERVENTION_CONFIG_SPACE


def load_exp_config():
    """Load experimental configuration from YAML file"""
    config_path = pathlib.Path(__file__).parent.parent.parent.joinpath('scripts').joinpath('configs').joinpath('exp_config.yaml')
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def create_optimization_plot(draw_data, objectives_config):
    """
    Create interactive optimization plots using Altair - one plot per objective.
    
    Args:
        draw_data (dict): Data containing line_data for plotting
        objectives_config (list): List of objectives from config
        
    Returns:
        list: List of interactive Altair charts (one per objective)
    """
    line_data = draw_data.get('line_data', [])
    
    if not line_data:
        return [alt.Chart().mark_text(text="No data available for plotting")]
    
    plots = []
    
    for obj_idx, obj_data in enumerate(line_data):
        best_data = obj_data.get('best', [])
        infeasible_data = obj_data.get('infeasible', [])
        feasible_data = obj_data.get('feasible', [])
        
        if not best_data and not infeasible_data and not feasible_data:
            objective_name = objectives_config[obj_idx]['metric'] if obj_idx < len(objectives_config) else f"Objective {obj_idx + 1}"
            plots.append(alt.Chart().mark_text(text=f"No data for {objective_name}"))
            continue
        
        # Create DataFrame for this objective
        plot_data = []
        
        # Add best data (for the line)
        for point in best_data:
            if len(point) >= 2:
                plot_data.append({
                    'Iteration': point[0],
                    'Objective Value': point[1],
                    'Type': 'Best'
                })
        
        # Add infeasible data (scatter points only)
        for point in infeasible_data:
            if len(point) >= 2:
                plot_data.append({
                    'Iteration': point[0],
                    'Objective Value': point[1],
                    'Type': 'Infeasible'
                })
        
        # Add feasible data (scatter points only)
        for point in feasible_data:
            if len(point) >= 2:
                plot_data.append({
                    'Iteration': point[0],
                    'Objective Value': point[1],
                    'Type': 'Feasible'
                })
        
        if not plot_data:
            objective_name = objectives_config[obj_idx]['metric'] if obj_idx < len(objectives_config) else f"Objective {obj_idx + 1}"
            plots.append(alt.Chart().mark_text(text=f"No valid data points for {objective_name}"))
            continue
        
        df = pd.DataFrame(plot_data)
        
        # Selection tools
        brush_x = alt.selection_interval(encodings=['x'], name=f'brushX_obj{obj_idx}')
        brush_y = alt.selection_interval(encodings=['y'], name=f'brushY_obj{obj_idx}')
        
        # Color scheme for different point types
        color_scheme = {
            'Best': 'steelblue',
            'Infeasible': 'grey',
            'Feasible': 'grey'
        }
        
        # Get objective name for title
        objective_name = objectives_config[obj_idx]['metric'] if obj_idx < len(objectives_config) else f"Objective {obj_idx + 1}"
        objective_name = f"{objective_name} (Minimized)"
        
        # Main scatter plot for all points
        base_font_size = 18
        points = alt.Chart(df).mark_point(filled=True, size=60).encode(
            x=alt.X('Iteration:Q', title='Iterations',
                    scale=alt.Scale(domain=brush_x),
                    axis=alt.Axis(labelFontSize=base_font_size, titleFontSize=base_font_size + 2, format='d')),
            y=alt.Y('Objective Value:Q', title=f'{objective_name}',
                    scale=alt.Scale(domain=brush_y, zero=True),
                    axis=alt.Axis(labelFontSize=base_font_size, titleFontSize=base_font_size + 2)),
            color=alt.Color('Type:N', scale=alt.Scale(domain=list(color_scheme.keys()), range=list(color_scheme.values())), legend=None),
            tooltip=[
                alt.Tooltip('Iteration', title='Iter'),
                alt.Tooltip('Objective Value', format='.5f', title=objective_name),
            ]
        ).properties(
            width=800,
            height=400,
        )
        
        # Line connecting only the best points
        best_df = df[df['Type'] == 'Best']
        line = alt.Chart(best_df).mark_line(color='steelblue', strokeWidth=2).encode(
            x=alt.X('Iteration:Q', scale=alt.Scale(domain=brush_x), axis=alt.Axis(format='d')),
            y=alt.Y('Objective Value:Q', scale=alt.Scale(domain=brush_y, zero=True))
        )
        
        # Bottom context (x-axis zoom)
        x_brush = alt.Chart(df).mark_area(opacity=0.3, color='#bbbbbb').encode(
            x=alt.X('Iteration:Q', axis=None),
            y=alt.Y('Objective Value:Q', axis=None)
        ).properties(
            width=800,
            height=40
        ).add_selection(
            brush_x
        )
        
        # Right density plot (y-axis zoom)
        y_density = alt.Chart(df).transform_density(
            'Objective Value',
            as_=['Objective Value', 'density']
        ).mark_area(opacity=0.3, color='#bbbbbb', interpolate='monotone', orient='horizontal').encode(
            y=alt.Y('Objective Value:Q', axis=None),
            x=alt.X('density:Q', axis=None)
        ).properties(
            width=40,
            height=400
        ).add_selection(
            brush_y
        )
        
        # Combine layout for this objective
        objective_plot = alt.hconcat(
            alt.vconcat(points + line, x_brush),
            y_density
        ).configure_axis(
            labelFontSize=base_font_size,
            titleFontSize=base_font_size + 2,
        )
        
        plots.append(objective_plot)
    
    return plots


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


def load_optimization_data(exp_config_name: str, lp_name: str, run_num: int, max_trials: int, ref_point: list):
    """
    Load optimization data from database and prepare for visualization.
    
    Args:
        exp_config_name (str): Name of the experimental configuration
        lp_name (str): Learning pipeline name
        run_num (int): Run number
        max_trials (int): Maximum number of trials
        ref_point (list): Reference point for optimization
        
    Returns:
        dict: Task info and history data
    """
    # Read an experimental config
    db_secrets_path = pathlib.Path(__file__).parent.parent.parent.joinpath('scripts').joinpath('configs').joinpath('secrets_gmail.env')

    # Prepare a History object
    bo_advisor_config = BOAdvisorConfig()
    config_space = create_config_space(lp_name)
    raw_history, defined_objectives, surrogate_model_type, acq_type, num_completed_pps = (
        read_history_from_db(secrets_path=str(db_secrets_path),
                              exp_config_name=exp_config_name,
                              lp_name=lp_name,
                              run_num=run_num,
                              ref_point=ref_point))
    history = prepare_history(data=raw_history,
                              config_space=config_space,
                              defined_objectives=defined_objectives)

    task_info = {
        'advisor_type': 'default',
        'max_runs': max_trials,
        'max_runtime_per_trial': bo_advisor_config.max_runtime_per_trial,
        'surrogate_type': surrogate_model_type,
        'acq_type': acq_type,
        'num_completed_pps': num_completed_pps,
        'defined_objectives': defined_objectives,
        'constraint_surrogate_type': None,
        'transfer_learning_history': bo_advisor_config.transfer_learning_history,
    }
    
    return {
        'task_info': task_info,
        'history': history,
        'config_space': config_space
    }


def generate_draw_data(visualizer, update_importance=False, verify_surrogate=False):
    # Basic data
    draw_data = visualizer.generate_basic_data()

    # Importance data
    importance = visualizer._cache_advanced_data.get('importance')
    if update_importance:
        importance = visualizer.generate_importance_data(method=visualizer.advanced_analysis_options['importance_method'])
        visualizer._cache_advanced_data['importance'] = importance
    draw_data['importance_data'] = importance

    # Verify surrogate data
    if verify_surrogate:
        pred_label_data, grade_data, cons_pred_label_data = visualizer.generate_verify_surrogate_data()
        draw_data['pred_label_data'] = pred_label_data
        draw_data['grade_data'] = grade_data
        draw_data['cons_pred_label_data'] = cons_pred_label_data

    return draw_data


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


def is_pareto_efficient(costs, maximise=False):
    """
    :param costs: An (n_points, n_costs) array
    :maximise: boolean. True for maximising, False for minimising
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            if maximise:
                is_efficient[is_efficient] = np.any(costs[is_efficient]>=c, axis=1)  # Remove dominated points
            else:
                is_efficient[is_efficient] = np.any(costs[is_efficient]<=c, axis=1)  # Remove dominated points
    return is_efficient


def create_pareto_plot(draw_data, objectives_config):
    """
    Create a Pareto frontier plot using Altair (2D) or Plotly (3D).
    Args:
        draw_data (dict): Data containing pareto_data for plotting
        objectives_config (list): List of objectives from config
    Returns:
        alt.Chart or plotly Figure: Pareto plot
    """
    pareto_data = draw_data.get('pareto_data', {})
    all_points = pareto_data.get('all_points', [])
    pareto_points = pareto_data.get('pareto_point', [])

    # Dynamically determine the number of objectives
    num_objectives = len(objectives_config)
    if all_points and isinstance(all_points[0], (list, tuple)):
        num_objectives = len(all_points[0])
    elif pareto_points and isinstance(pareto_points[0], (list, tuple)):
        num_objectives = len(pareto_points[0])
    elif objectives_config:
        num_objectives = len(objectives_config)
    else:
        num_objectives = 2  # fallback

    # If no data, return a placeholder chart
    if not all_points or not pareto_points:
        return alt.Chart(pd.DataFrame({'x': [], 'y': []})).mark_text(text="No Pareto data available")

    # Build column names and axis labels dynamically
    col_names = [f"obj{i+1}" for i in range(num_objectives)]
    obj_labels = []
    for i in range(num_objectives):
        if i < len(objectives_config):
            label = objectives_config[i].get('metric', f'obj{i+1}')
        else:
            label = f'obj{i+1}'
        obj_labels.append(f"{label} (Minimized)" if num_objectives == 2 else label)

    try:
        df_all = pd.DataFrame(all_points, columns=col_names)
    except Exception as e:
        return alt.Chart(pd.DataFrame({'x': [], 'y': []})).mark_text(text=f"Pareto data shape error: {e}")

    base_font_size = 16
    if num_objectives == 2:
        # 2D scatter + line plot (Altair)
        try:
            df_pareto = pd.DataFrame(pareto_points, columns=col_names)
        except Exception as e:
            return alt.Chart(pd.DataFrame({'x': [], 'y': []})).mark_text(text=f"Pareto data shape error: {e}")
        points = alt.Chart(df_all).mark_circle(size=80, color='#91CC75', opacity=0.6).encode(
            x=alt.X(col_names[0], title=obj_labels[0], scale=alt.Scale(zero=True)),
            y=alt.Y(col_names[1], title=obj_labels[1], scale=alt.Scale(zero=True)),
            tooltip=col_names
        )
        pareto_line = alt.Chart(df_pareto).mark_line(point=True, color='steelblue').encode(
            x=col_names[0],
            y=col_names[1]
        )
        chart = (points + pareto_line).properties(
            width=400,
            height=400,
        ).configure_axis(
            labelFontSize=base_font_size,
            titleFontSize=base_font_size + 2,
        ).configure_title(
            fontSize=24
        )
        return chart
    elif num_objectives == 3:
        # 3D scatter plot (Plotly) using notebook logic
        data_np = df_all[col_names].values
        pareto_mask = is_pareto_efficient(data_np)
        pareto_points = data_np[pareto_mask]
        non_pareto_points = data_np[~pareto_mask]
        fig = go.Figure()
        # Non-Pareto points
        if len(non_pareto_points) > 0:
            fig.add_trace(go.Scatter3d(
                x=non_pareto_points[:, 0],
                y=non_pareto_points[:, 1],
                z=non_pareto_points[:, 2],
                mode='markers',
                marker=dict(size=4, color='lightgray'),
                name='Non-Pareto Points'
            ))
        # Pareto points
        if len(pareto_points) > 0:
            fig.add_trace(go.Scatter3d(
                x=pareto_points[:, 0],
                y=pareto_points[:, 1],
                z=pareto_points[:, 2],
                mode='markers',
                marker=dict(size=5, color='red'),
                name='Pareto Frontier Points'
            ))
        # Pareto surface using convex hull
        if len(pareto_points) >= 4:
            try:
                hull = ConvexHull(pareto_points)
                for simplex in hull.simplices:
                    fig.add_trace(go.Mesh3d(
                        x=pareto_points[simplex, 0],
                        y=pareto_points[simplex, 1],
                        z=pareto_points[simplex, 2],
                        color='red',
                        opacity=0.3,
                        name='Pareto Surface',
                        showscale=False
                    ))
            except Exception as e:
                # QhullError or other: skip surface, just show points
                pass
        fig.update_layout(
            title='3D Pareto Frontier Plot (Minimization)',
            scene=dict(
                xaxis_title=obj_labels[0],
                yaxis_title=obj_labels[1],
                zaxis_title=obj_labels[2]
            ),
            legend=dict(x=0, y=1),
            height=700
        )
        return fig
    else:
        # Not supported
        return alt.Chart(pd.DataFrame({'x': [], 'y': []})).mark_text(text="Pareto plot only supported for 2 or 3 objectives.")


def create_hypervolume_plot(draw_data):
    """
    Create a Hypervolume plot using Altair.
    Args:
        draw_data (dict): Data containing pareto_data["hv"] for plotting
    Returns:
        alt.Chart: Hypervolume plot
    """
    pareto_data = draw_data.get('pareto_data', {})
    hv = pareto_data.get('hv', [])
    data = pd.DataFrame(data=hv, columns=['Iterations', 'Hypervolume'])

    base_font_size = 16
    chart = alt.Chart(data).mark_line(point=alt.OverlayMarkDef(size=50)).encode(
        x=alt.X('Iterations:Q', title='Iterations'),
        y=alt.Y('Hypervolume:Q', title='Hypervolume'),
        tooltip=['Iterations', 'Hypervolume']
    ).properties(
        width=600,
        height=400,
        title=alt.TitleParams("Hypervolume Over Iterations", fontSize=20, fontWeight='bold')
    ).configure_axis(
        labelFontSize=base_font_size,
        titleFontSize=base_font_size + 2,
    )
    return chart


def create_parameter_importance_plot(draw_data, objectives_config):
    """
    Create an Overall Parameter Importance plot using Altair.
    Args:
        draw_data (dict): Data containing importance_data for plotting
        objectives_config (list): List of objectives from config
    Returns:
        alt.Chart: Parameter importance plot
    """
    importance_data = draw_data.get('importance_data', {})
    x_labels = importance_data.get('x', [])
    # Dynamically collect all available objectives from importance_data['data']
    obj_keys = [k for k in importance_data.get('data', {}).keys() if k.startswith('obj')]
    obj_names = []
    obj_values = []
    for i, k in enumerate(obj_keys):
        if i < len(objectives_config):
            name = objectives_config[i]['metric']
        else:
            name = k
        obj_names.append(name)
        obj_values.append(importance_data['data'][k])

    # If no data, return placeholder
    if not x_labels or not obj_values or any(len(vals) == 0 for vals in obj_values):
        return alt.Chart(pd.DataFrame({'Parameter': [], 'Importance': []})).mark_text(text="No parameter importance data available")

    # Build DataFrame
    data_dict = {'Parameter': x_labels}
    for name, vals in zip(obj_names, obj_values):
        data_dict[name] = vals
    data = pd.DataFrame(data_dict)
    data_long = data.melt(id_vars='Parameter', var_name='Objective', value_name='Importance')

    # Color palette for up to 3 objectives
    color_palette = ['#4c78a8', '#72b37e', '#e45756']
    base_font_size = 16
    chart = alt.Chart(data_long).mark_bar().encode(
        x=alt.X(
            'Parameter:N',
            title=None,
            axis=alt.Axis(labelFontSize=base_font_size, titleFontSize=base_font_size + 2, labelAngle=-30),
            scale=alt.Scale(paddingInner=0.5)
        ),
        y=alt.Y('Importance:Q', title='Importance', axis=alt.Axis(labelFontSize=base_font_size, titleFontSize=base_font_size + 2)),
        color=alt.Color('Objective:N', legend=None, scale=alt.Scale(range=color_palette[:len(obj_names)]), sort=obj_names),
        column=alt.Column('Objective:N', title=None, sort=obj_names, header=alt.Header(labelFontSize=base_font_size, titleFontSize=base_font_size + 2))
    ).properties(
        width=300,
        height=300
    ).configure_axis(
        labelLimit=400,
        titleLimit=300,
    ).configure_legend(
        titleFontSize=base_font_size + 2,
        labelFontSize=base_font_size,
        symbolStrokeWidth=10,
        labelLimit=400,
        titleLimit=300,
        orient='top',
        direction='horizontal',
        columns=1,
    )
    return chart


def get_config_space_bounds(parameter_name, lp_name):
    """
    Get min and max bounds for a parameter from the config spaces.
    
    Args:
        parameter_name (str): Name of the parameter (e.g., 'model__max_depth')
        lp_name (str): Name of a logical pipeline

    Returns:
        tuple: (min_val, max_val) or None if not found
    """
    # Get all model config spaces
    mvi_name, fi_name, model_name = lp_name.split('&')
    models_config = get_models_params_for_tuning()
    
    # Search in model config spaces
    if parameter_name.startswith('model__'):
        model_config = models_config.get(model_name, {})
        if 'config_space' in model_config:
            for param_name, param_config in model_config['config_space'].items():
                if param_name == parameter_name:
                    if hasattr(param_config, 'choices'):  # CategoricalHyperparameter
                        if isinstance(param_config.choices[0], str) or isinstance(param_config.choices[0], bool):
                            return 0, len(param_config.choices) - 1

                        param_config.choices = tuple([0 if x == 'None' else x for x in param_config.choices])
                        return min(param_config.choices), max(param_config.choices)
                    elif hasattr(param_config, 'lower') and hasattr(param_config, 'upper'):  # UniformHyperparameter
                        return param_config.lower, param_config.upper
    
    # Search in null imputation config spaces
    if parameter_name.startswith('mvi__'):
        method_config = NULL_IMPUTATION_CONFIG.get(mvi_name, {})
        if 'config_space' in method_config:
            for param_name, param_config in method_config['config_space'].items():
                if param_name == parameter_name:
                    if hasattr(param_config, 'choices'):  # CategoricalHyperparameter
                        if isinstance(param_config.choices[0], str) or isinstance(param_config.choices[0], bool):
                            return 0, len(param_config.choices) - 1

                        param_config.choices = tuple([0 if x == 'None' else x for x in param_config.choices])
                        return min(param_config.choices), max(param_config.choices)
                    elif hasattr(param_config, 'lower') and hasattr(param_config, 'upper'):  # UniformHyperparameter
                        return param_config.lower, param_config.upper
    
    # Search in fairness intervention config spaces
    if parameter_name.startswith('fi__'):
        method_config = FAIRNESS_INTERVENTION_CONFIG_SPACE.get(fi_name, {})
        if 'config_space' in method_config:
            for param_name, param_config in method_config['config_space'].items():
                if param_name == parameter_name:
                    if hasattr(param_config, 'choices'):  # CategoricalHyperparameter
                        if isinstance(param_config.choices[0], str) or isinstance(param_config.choices[0], bool):
                            return 0, len(param_config.choices) - 1

                        param_config.choices = tuple([0 if x == 'None' else x for x in param_config.choices])
                        return min(param_config.choices), max(param_config.choices)
                    elif hasattr(param_config, 'lower') and hasattr(param_config, 'upper'):  # UniformHyperparameter
                        return param_config.lower, param_config.upper
    
    return None


def get_config_space_ticks(parameter_name):
    """
    Get tick information for a parameter from the config spaces.
    
    Args:
        parameter_name (str): Name of the parameter (e.g., 'model__max_depth')
        
    Returns:
        list: List of tick values for the parameter
    """
    # Get all model config spaces
    models_config = get_models_params_for_tuning()
    
    # Search in model config spaces
    for model_name, model_config in models_config.items():
        if 'config_space' in model_config:
            for param_name, param_config in model_config['config_space'].items():
                if param_name == parameter_name:
                    if hasattr(param_config, 'choices'):  # CategoricalHyperparameter
                        return param_config.choices
                    elif hasattr(param_config, 'lower') and hasattr(param_config, 'upper'):  # UniformHyperparameter
                        if hasattr(param_config, 'q') and param_config.q is not None:  # Integer with quantization
                            q = param_config.q
                            lower, upper = param_config.lower, param_config.upper
                            return list(range(lower, upper + 1, q))
                        else:  # Float or continuous
                            lower, upper = param_config.lower, param_config.upper
                            return [lower, (lower + upper) / 2, upper]
    
    # Search in null imputation config spaces
    for method_name, method_config in NULL_IMPUTATION_CONFIG.items():
        if 'config_space' in method_config:
            for param_name, param_config in method_config['config_space'].items():
                if param_name == parameter_name:
                    if hasattr(param_config, 'choices'):  # CategoricalHyperparameter
                        return param_config.choices
                    elif hasattr(param_config, 'lower') and hasattr(param_config, 'upper'):  # UniformHyperparameter
                        if hasattr(param_config, 'q') and param_config.q is not None:  # Integer with quantization
                            q = param_config.q
                            lower, upper = param_config.lower, param_config.upper
                            return list(range(lower, upper + 1, q))
                        else:  # Float or continuous
                            lower, upper = param_config.lower, param_config.upper
                            return [lower, (lower + upper) / 2, upper]
    
    # Search in fairness intervention config spaces
    for method_name, method_config in FAIRNESS_INTERVENTION_CONFIG_SPACE.items():
        if 'config_space' in method_config:
            for param_name, param_config in method_config['config_space'].items():
                if param_name == parameter_name:
                    if hasattr(param_config, 'choices'):  # CategoricalHyperparameter
                        return param_config.choices
                    elif hasattr(param_config, 'lower') and hasattr(param_config, 'upper'):  # UniformHyperparameter
                        if hasattr(param_config, 'q') and param_config.q is not None:  # Integer with quantization
                            q = param_config.q
                            lower, upper = param_config.lower, param_config.upper
                            return list(range(lower, upper + 1, q))
                        else:  # Float or continuous
                            lower, upper = param_config.lower, param_config.upper
                            return [lower, (lower + upper) / 2, upper]
    
    return None


def create_parallel_coordinates_plot(draw_data, objectives_config, selected_lp_name):
    """
    Create a parallel coordinates plot using the parallel_data from draw_data.
    
    Args:
        draw_data (dict): Data containing parallel_data for plotting
        objectives_config (list): List of objectives from config
        selected_lp_name (str): Name of a logical pipeline

    Returns:
        alt.Chart: Interactive parallel coordinates chart
    """
    parallel_data = draw_data.get('parallel_data', {})

    if not parallel_data or 'data' not in parallel_data or 'schema' not in parallel_data:
        return alt.Chart().mark_text(text="No parallel coordinates data available")
    
    # Extract data and schema
    data = parallel_data['data']
    schema = parallel_data['schema']
    
    if not data or not schema:
        return alt.Chart().mark_text(text="No parallel coordinates data available")
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=schema)
    
    # Get feature columns (exclude Uid and objective columns)
    objective_names = [f"Objs {i+1}" for i in range(len(objectives_config))]
    feature_columns = [col for col in schema if col not in ['Uid'] + objective_names]
    
    # Convert string values to numeric where possible, and handle categorical variables
    for col in feature_columns + objective_names:
        if col in df.columns:
            # Handle boolean values first
            if df[col].dtype == bool:
                df[col] = df[col].astype(int)  # Convert boolean to int (0/1)
            else:
                # Try to convert to numeric, if it fails, treat as categorical
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    # If all values are NaN after conversion, try categorical
                    if df[col].isna().all():
                        # Create numeric mapping for categorical values
                        unique_vals = df[col].astype(str).unique()
                        val_mapping = {val: i for i, val in enumerate(unique_vals)}
                        df[col] = df[col].astype(str).map(val_mapping)
                    else:
                        # Fill NaN values with 0 for numeric columns
                        df[col] = df[col].fillna(0)
                except (ValueError, TypeError):
                    # Create numeric mapping for categorical values
                    unique_vals = df[col].astype(str).unique()
                    val_mapping = {val: i for i, val in enumerate(unique_vals)}
                    df[col] = df[col].astype(str).map(val_mapping)
    
    # Create a mapping for integer-to-string conversion based on config spaces
    int_to_string_mapping = {}
    for feature in feature_columns:
        config_ticks = get_config_space_ticks(feature)
        if config_ticks is not None and len(config_ticks) > 0:
            # Check if config ticks are all strings
            if all(isinstance(tick, str) for tick in config_ticks):
                # Create mapping from integer index to string value
                int_to_string_mapping[feature] = {i: tick for i, tick in enumerate(config_ticks)}
    
    # Melt to long format for parallel coordinates
    df_long = df.melt(id_vars=['Uid'], value_vars=feature_columns + objective_names,
                      var_name='Feature', value_name='Value')
    
    # Normalize values for alignment - handle both numeric and categorical data
    def normalize_group(group):
        feature_name = group.name
        if group.dtype in ['int64', 'float64']:
            # Try to get bounds from config space first
            config_bounds = get_config_space_bounds(feature_name, selected_lp_name)
            if config_bounds is not None:
                config_min, config_max = config_bounds
                if config_max != config_min:
                    return (group - config_min) / (config_max - config_min)
                else:
                    return pd.Series([0.5] * len(group), index=group.index)
            else:
                # Fall back to data-based normalization
                min_val = group.min()
                max_val = group.max()
                if max_val != min_val:
                    return (group - min_val) / (max_val - min_val)
                else:
                    return pd.Series([0.5] * len(group), index=group.index)
        else:
            # For categorical data, use the values as they are (already mapped to 0, 1, 2, etc.)
            return group
    
    df_long['NormValue'] = df_long.groupby('Feature')['Value'].transform(normalize_group)
    
    # Create tick positions for each feature using config spaces
    tick_positions = {}
    for feature in feature_columns + objective_names:
        if feature in objective_names:
            # For objectives, create more granular ticks
            min_val, max_val = df[feature].min(), df[feature].max()
            if max_val - min_val > 0:
                tick_positions[feature] = np.linspace(min_val, max_val, 5)
            else:
                tick_positions[feature] = [min_val]
        else:
            # Try to get ticks from config spaces first
            config_ticks = get_config_space_ticks(feature)
            if config_ticks is not None:
                # Check if config ticks are all strings
                if all(isinstance(tick, str) for tick in config_ticks):
                    # Use integer indices for positioning, but store string values for display
                    tick_positions[feature] = list(range(len(config_ticks)))
                else:
                    # Use config space ticks directly for static positioning
                    tick_positions[feature] = config_ticks
            else:
                # Fall back to data-based ticks with static spacing
                unique_vals = sorted(df[feature].unique())
                if len(unique_vals) <= 10:
                    tick_positions[feature] = unique_vals
                else:
                    # For continuous features, create 5-6 ticks with static spacing
                    min_val, max_val = df[feature].min(), df[feature].max()
                    tick_positions[feature] = np.linspace(min_val, max_val, 6)
    
    # Convert to normalized scale and build tick+label layers
    tick_rows = []
    label_rows = []
    for feature, ticks in tick_positions.items():
        feature_data = df[feature]
        
        # Try to get bounds from config space first
        config_bounds = get_config_space_bounds(feature, selected_lp_name)
        if config_bounds is not None:
            min_val, max_val = config_bounds
        else:
            min_val, max_val = feature_data.min(), feature_data.max()
        
        # Handle boolean values by converting to integers
        if feature_data.dtype == bool:
            min_val, max_val = 0, 1
            # Convert boolean ticks to integers
            ticks = [1 if tick else 0 for tick in ticks]
        
        for val in ticks:
            # Handle boolean values in ticks
            if isinstance(val, bool):
                norm = 0.0 if not val else 1.0
                tick_rows.append({'Feature': feature, 'NormValue': norm})
                label_rows.append({'Feature': feature, 'NormValue': norm, 'Label': str(val)})
            else:
                # Check if this feature has string config values
                if feature in int_to_string_mapping:
                    # Use integer index for positioning, string value for display
                    norm = val / (len(int_to_string_mapping[feature]) - 1) if len(int_to_string_mapping[feature]) > 1 else 0.5
                    display_val = int_to_string_mapping[feature].get(int(val), str(val))
                    tick_rows.append({'Feature': feature, 'NormValue': norm})
                    label_rows.append({'Feature': feature, 'NormValue': norm, 'Label': str(display_val)})
                else:
                    # For static ticks, normalize based on the full range of possible values from config
                    # If we have config ticks, use their min/max for normalization
                    config_ticks = get_config_space_ticks(feature)
                    if config_ticks is not None and len(config_ticks) > 0:
                        # Filter out non-numeric values for normalization
                        numeric_config_ticks = []
                        for tick in config_ticks:
                            if isinstance(tick, (int, float)) or (isinstance(tick, str) and tick.replace('.', '').replace('-', '').isdigit()):
                                numeric_config_ticks.append(float(tick))
                        
                        if numeric_config_ticks:
                            config_min = min(numeric_config_ticks)
                            config_max = max(numeric_config_ticks)
                            # Check if the current value is numeric
                            if isinstance(val, (int, float)) or (isinstance(val, str) and val.replace('.', '').replace('-', '').isdigit()):
                                val_numeric = float(val)
                                norm = (val_numeric - config_min) / (config_max - config_min) if config_max != config_min else 0.5
                            else:
                                # For non-numeric values, use a default position
                                norm = 0.5
                        else:
                            # No numeric config ticks, fall back to data-based normalization
                            norm = (val - min_val) / (max_val - min_val) if max_val != min_val else 0.5
                    else:
                        # Fall back to data-based normalization
                        norm = (val - min_val) / (max_val - min_val) if max_val != min_val else 0.5
                    
                    tick_rows.append({'Feature': feature, 'NormValue': norm})
                    label_rows.append({'Feature': feature, 'NormValue': norm, 'Label': f"{val:.3f}" if isinstance(val, float) else str(val)})
    
    tick_df = pd.DataFrame(tick_rows)
    label_df = pd.DataFrame(label_rows)

    # Smoothed lines for each trial
    base_font_size = 14
    lines_chart = alt.Chart(df_long).mark_line(
        interpolate='monotone', opacity=0.35,
    ).encode(
        x=alt.X('Feature:N', sort=feature_columns + objective_names,
                axis=alt.Axis(orient='top', title=None, labelAngle=-15)),
        y=alt.Y('NormValue:Q', axis=None),
        color=alt.Color('Uid:Q', scale=alt.Scale(scheme='turbo')),
        detail='Uid:N'
    )
    
    # Background vertical config lines
    vertical_lines = pd.DataFrame([
        {'Feature': f, 'NormValue': y} for f in feature_columns + objective_names 
        for y in np.linspace(0, 1, 25)
    ])
    rule_chart = alt.Chart(vertical_lines).mark_rule(color='black', opacity=0.1).encode(
        x=alt.X('Feature:N', sort=feature_columns + objective_names, 
                axis=alt.Axis(orient='top', title=None, labelAngle=-15)),
        y=alt.Y('NormValue:Q', axis=None),
    )
    
    # Tick marks (short horizontal lines)
    tick_marks = alt.Chart(tick_df).mark_tick(
        color='black', thickness=1, size=base_font_size, orient='horizontal'
    ).encode(
        x=alt.X('Feature:N', sort=feature_columns + objective_names, 
                axis=alt.Axis(orient='top', title=None, labelAngle=-15)),
        y=alt.Y('NormValue:Q', axis=None),
    )
    
    # Tick labels
    tick_labels = alt.Chart(label_df).mark_text(
        align='left', dx=5, dy=0, fontSize=base_font_size, color='black'
    ).encode(
        x=alt.X('Feature:N', sort=feature_columns + objective_names, 
                axis=alt.Axis(orient='top', title=None, labelAngle=-15)),
        y=alt.Y('NormValue:Q', axis=None),
        text='Label:N'
    )
    
    # Final composed chart
    final_plot = alt.layer(rule_chart, tick_marks, tick_labels, lines_chart).properties(
        width=950,
        height=450
    ).configure_view(
        stroke=None  # Removes the outer border
    ).configure_axis(
        labelFontSize=base_font_size + 2,
        titleFontSize=base_font_size + 4,
        labelFontWeight='normal',
        titleFontWeight='normal',
        labelLimit=400,
        titleLimit=300,
    ).configure_title(
        fontSize=base_font_size + 2
    ).configure_legend(
        titleFontSize=base_font_size + 4,
        labelFontSize=base_font_size + 2,
        symbolStrokeWidth=10,
        labelLimit=400,
        titleLimit=300,
    )
    
    return final_plot


def create_ternary_plot(draw_data, objectives_config):
    """
    Create a ternary plot using plotly.express if there are three objectives.
    Args:
        draw_data (dict): Data containing pareto_data for plotting
        objectives_config (list): List of objectives from config
    Returns:
        plotly Figure: Ternary plot or None if not applicable
    """
    pareto_data = draw_data.get('pareto_data', {})
    all_points = pareto_data.get('all_points', [])
    if not all_points or len(objectives_config) != 3:
        return None

    # Build column names and axis labels dynamically
    col_names = [f"obj{i+1}" for i in range(3)]
    obj_labels = []
    for i in range(3):
        if i < len(objectives_config):
            label = objectives_config[i].get('metric', f'obj{i+1}')
        else:
            label = f'obj{i+1}'
        obj_labels.append(label + " (Minimized)")

    df = pd.DataFrame(all_points, columns=col_names)
    # Normalize each row to sum to 1 (ternary requirement)
    sums = df[col_names].sum(axis=1)
    for i, col in enumerate(col_names):
        df[f'{col}_norm'] = df[col] / sums
    # Round for tooltip
    for col in col_names:
        df[f'{col}_rounded'] = df[col].round(4)
        df[f'{col}_norm_rounded'] = df[f'{col}_norm'].round(4)

    # Add a column for fixed dot size to avoid name conflict
    df['_dot_size'] = 5

    # Title with legend for obj1, obj2, obj3
    legend_lines = [f"{col_names[i]} = {obj_labels[i]}" for i in range(3)]
    legend_text = "<br>".join(legend_lines)
    full_title = (f"Ternary Plot: {obj_labels[0]} vs {obj_labels[1]} vs {obj_labels[2]}".replace(" (Minimized)", "") +
                  f"<br><br><span style='font-size:14px'>{legend_text}</span>")

    fig = px.scatter_ternary(
        df,
        a=f'{col_names[0]}_norm',
        b=f'{col_names[1]}_norm',
        c=f'{col_names[2]}_norm',
        size='_dot_size',  # Use a unique column name for size
        size_max=10,
        opacity=0.7,
        hover_data={
            '_dot_size': False,
            f'{col_names[0]}_norm': False,
            f'{col_names[1]}_norm': False,
            f'{col_names[2]}_norm': False,
            f'{col_names[0]}_rounded': True,
            f'{col_names[1]}_rounded': True,
            f'{col_names[2]}_rounded': True,
            f'{col_names[0]}_norm_rounded': True,
            f'{col_names[1]}_norm_rounded': True,
            f'{col_names[2]}_norm_rounded': True,
        }
    )
    fig.update_layout(
        title=full_title,
        ternary=dict(
            aaxis_title=obj_labels[0],
            baxis_title=obj_labels[1],
            caxis_title=obj_labels[2]
        ),
        height=600
    )
    return fig


def create_hyperparams_markdown(pipeline_hyperparams, selected_lp_name: str):
    # Group hyperparameters by prefix
    model_hparams = []
    mvi_hparams = []
    fi_hparams = []
    for k, v in pipeline_hyperparams.items():
        if k.startswith('model__'):
            model_hparams.append((k.replace('model__', ''), v))
        elif k.startswith('mvi__'):
            mvi_hparams.append((k.replace('mvi__', ''), v))
        elif k.startswith('fi__'):
            fi_hparams.append((k.replace('fi__', ''), v))

    # Extract component names from logical pipeline name
    mvi_name, fi_name, model_name = selected_lp_name.split('&') if '&' in selected_lp_name else (None, None, None)

    hparams_md = ""
    if mvi_hparams:
        mvi_title = f"**Null Imputer Hyperparameters (`{mvi_name}`)**:\n" if mvi_name else "**Null Imputer Hyperparameters:**\n"
        hparams_md += mvi_title
        for k, v in mvi_hparams:
            hparams_md += f"  - `{k}`: {v}\n"
    if fi_hparams:
        fi_title = f"**Fairness Intervention Hyperparameters (`{fi_name}`)**:\n" if fi_name else "**Fairness Intervention Hyperparameters:**\n"
        hparams_md += fi_title
        for k, v in fi_hparams:
            hparams_md += f"  - `{k}`: {v}\n"
    if model_hparams:
        model_title = f"**Model Hyperparameters (`{model_name}`)**:\n" if model_name else "**Model Hyperparameters:**\n"
        hparams_md += model_title
        for k, v in model_hparams:
            hparams_md += f"  - `{k}`: {v}\n"
    if not (model_hparams or mvi_hparams or fi_hparams):
        hparams_md += "- _No hyperparameters found for this pipeline._\n"

    return hparams_md


def create_hyperparams_json(pipeline_hyperparams, selected_lp_name: str):
    # Group hyperparameters by prefix
    model_hparams = {}
    mvi_hparams = {}
    fi_hparams = {}
    for k, v in pipeline_hyperparams.items():
        if k.startswith('model__'):
            model_hparams[k.replace('model__', '')] = v
        elif k.startswith('mvi__'):
            mvi_hparams[k.replace('mvi__', '')] = v
        elif k.startswith('fi__'):
            fi_hparams[k.replace('fi__', '')] = v

    # Extract component names from logical pipeline name
    mvi_name, fi_name, model_name = selected_lp_name.split('&') if '&' in selected_lp_name else (None, None, None)

    json_obj = {
        "logical_pipeline": selected_lp_name,
        "null_imputer": {"name": mvi_name, "hyperparameters": mvi_hparams} if mvi_hparams else None,
        "fairness_intervention": {"name": fi_name, "hyperparameters": fi_hparams} if fi_hparams else None,
        "model": {"name": model_name, "hyperparameters": model_hparams} if model_hparams else None
    }
    # Remove None values for cleaner JSON
    json_obj = {k: v for k, v in json_obj.items() if v is not None}
    return json.dumps(json_obj, indent=2)


def create_pipeline_optimization_page(exp_config_name: str, run_num: int, max_trials: int, ref_point: list):
    """Create the pipeline optimization page components"""
    
    # Load config to get lp_name combinations and objectives
    config = load_exp_config()
    secrets_path = pathlib.Path(__file__).parent.parent.parent.joinpath('scripts').joinpath('configs').joinpath('secrets_gmail.env')
    lp_name_combinations = generate_lp_name_combinations(config)
    objectives_config = config.get('optimisation_args', {}).get('objectives', [])
    
    # Default lp_name (first combination or a specific one)
    default_lp_name = lp_name_combinations[0]
    
    # Create objective choices from config
    objective_choices = [obj.get('metric', f'Objective {i+1}') for i, obj in enumerate(objectives_config)]
    default_objective = objective_choices[0] if objective_choices else "No objectives available"
    
    # Create objectives display from config
    objectives_display = "## Defined Objectives\n\n"
    for i, obj in enumerate(objectives_config):
        objectives_display += f"**{i+1}. {obj.get('metric', f'Objective {i+1}')}**\n"
        objectives_display += f"   - Group: {obj.get('group', 'N/A')}\n"
        objectives_display += f"   - Weight: {obj.get('weight', 'N/A')}\n"
        if 'minimize' in obj:
            objectives_display += f"   - Minimize: {obj['minimize']}\n"
        objectives_display += "\n"

    def update_plots(selected_lp_name, selected_objective):
        """Update plots based on selected lp_name and objective"""
        # Load optimization data with selected lp_name
        optimization_data = load_optimization_data(
            exp_config_name=exp_config_name,
            lp_name=selected_lp_name,
            run_num=run_num,
            max_trials=max_trials,
            ref_point=ref_point
        )

        # Build visualizer
        visualizer = build_visualizer(
            task_info=optimization_data['task_info'],
            option='advanced',
            history=optimization_data['history'],
            auto_open_html=False,
        )
        draw_data = generate_draw_data(visualizer, update_importance=True, verify_surrogate=True)

        # Create interactive plots using objectives from config
        plots = create_optimization_plot(draw_data, objectives_config)
        parallel_coordinates_plot = create_parallel_coordinates_plot(draw_data, objectives_config, selected_lp_name)
        pareto_plot = create_pareto_plot(draw_data, objectives_config)
        hypervolume_plot = create_hypervolume_plot(draw_data)
        parameter_importance_plot = create_parameter_importance_plot(draw_data, objectives_config)
        ternary_plot = create_ternary_plot(draw_data, objectives_config) if len(objectives_config) == 3 else None

        # Create updated table with task_info
        task_info = optimization_data['task_info']
        table_markdown = f"""
        | Experiment Config | Acquisition Type | Surrogate Type | Total Trials | Reference Point |
        |------------------|------------------|----------------|--------------|-----------------|
        | {exp_config_name} | {task_info.get('acq_type', 'N/A')} | {task_info.get('surrogate_type', 'N/A')} | {task_info.get('num_completed_pps', 'N/A')} | {ref_point} |
        """

        # Hyperparameters JSON for Code block
        pipeline_hyperparams = get_tuned_lp_for_exp_config(
            secrets_path=str(secrets_path),
            exp_config_name=exp_config_name,
            lp_name=selected_lp_name,
            run_num=run_num,
        )
        hparams_json = create_hyperparams_json(pipeline_hyperparams, selected_lp_name)

        # Find the objective index by name
        objective_idx = -1
        for i, obj in enumerate(objectives_config):
            if obj.get('metric') == selected_objective:
                objective_idx = i
                break

        # Return the selected objective plot, parallel coordinates plot, Pareto plot, Ternary plot, Hypervolume plot, Parameter importance plot, the updated table, and the hyperparams JSON as a separate output
        if 0 <= objective_idx < len(plots):
            return plots[objective_idx], parallel_coordinates_plot, pareto_plot, ternary_plot, hypervolume_plot, parameter_importance_plot, table_markdown, hparams_json
        else:
            return alt.Chart().mark_text(text=f"No data for {selected_objective}"), parallel_coordinates_plot, pareto_plot, ternary_plot, hypervolume_plot, parameter_importance_plot, table_markdown, hparams_json

    # New function: only update the Min Objective Value plot
    def update_objective_plot(selected_lp_name, selected_objective):
        optimization_data = load_optimization_data(
            exp_config_name=exp_config_name,
            lp_name=selected_lp_name,
            run_num=run_num,
            max_trials=max_trials,
            ref_point=ref_point
        )
        visualizer = build_visualizer(
            task_info=optimization_data['task_info'],
            option='advanced',
            history=optimization_data['history'],
            auto_open_html=False,
        )
        draw_data = generate_draw_data(visualizer, update_importance=True, verify_surrogate=True)
        plots = create_optimization_plot(draw_data, objectives_config)
        # Find the objective index by name
        objective_idx = -1
        for i, obj in enumerate(objectives_config):
            if obj.get('metric') == selected_objective:
                objective_idx = i
                break
        if 0 <= objective_idx < len(plots):
            return plots[objective_idx]
        else:
            return alt.Chart().mark_text(text=f"No data for {selected_objective}")

    # Create Gradio components with the plots
    with gr.Blocks() as pipeline_page:
        gr.Markdown("# Pipeline Optimization Progress")
        
        # Dropdown for lp_name selection
        lp_name_dropdown = gr.Dropdown(
            choices=lp_name_combinations,
            value=default_lp_name,
            label="Select Learning Pipeline",
            info="Choose the learning pipeline configuration to visualize"
        )
        
        # Add refresh button above the plot
        refresh_btn = gr.Button("🔄 Refresh Plot", variant="secondary")
        
        # Additional components can be added here
        gr.Markdown("")
        gr.Markdown("## Optimization Details")
        details_table = gr.Markdown("""
        | Experiment Config | Acquisition Type | Surrogate Type | Total Trials | Reference Point |
        |------------------|------------------|----------------|--------------|-----------------|
        | {} | {} | {} | {} | {} |
        """.format(exp_config_name, "N/A", "N/A", "N/A", ref_point))

        # Objectives display (static from config)
        gr.Markdown("")
        gr.Markdown(objectives_display)

        # Hyperparameters Markdown List
        pipeline_hyperparams = get_tuned_lp_for_exp_config(
            secrets_path=str(secrets_path),
            exp_config_name=exp_config_name,
            lp_name=default_lp_name,
            run_num=run_num,
        )
        hparams_md = create_hyperparams_markdown(pipeline_hyperparams, default_lp_name)
        gr.Markdown("")
        gr.Markdown("## Hyperparameters of the Tuned Logical Pipeline")
        # Use gr.Code for scrollable JSON
        hparams_json = create_hyperparams_json(pipeline_hyperparams, default_lp_name)
        hyperparams_code = gr.Code(value=hparams_json, language="json", label="Pipeline Hyperparameters (JSON)")

        # Min Objective Value Plot
        gr.Markdown("")
        gr.Markdown("## Min Objective Value Plot")
        with gr.Row():
            objective_dropdown = gr.Dropdown(
                choices=objective_choices,
                value=default_objective,
                label="Select Objective",
                info="Choose which objective to visualize",
            )
            gr.Markdown("")
        plot_component = gr.Plot()
        
        # Parallel Coordinate Plot
        gr.Markdown("## Parallel Coordinate Plot")
        parallel_coordinates_plot_component = gr.Plot()
        
        # Pareto Frontier Plot
        gr.Markdown("## Pareto Frontier Plot")
        pareto_plot_component = gr.Plot()

        # Ternary Plot (only for 3 objectives)
        ternary_plot_component = gr.Plot(visible=(len(objectives_config) == 3))

        # Hypervolume Plot
        gr.Markdown("## Hypervolume Plot")
        hypervolume_plot_component = gr.Plot()

        # Parameter Importance Plot
        gr.Markdown("## Parameter Importance Plot")
        parameter_importance_plot_component = gr.Plot()
        
        # Update all plots and info when pipeline changes
        lp_name_dropdown.change(
            fn=update_plots,
            inputs=[lp_name_dropdown, objective_dropdown],
            outputs=[plot_component, parallel_coordinates_plot_component, pareto_plot_component, ternary_plot_component, hypervolume_plot_component, parameter_importance_plot_component, details_table, hyperparams_code]
        )
        
        # Only update the Min Objective Value plot when objective changes
        objective_dropdown.change(
            fn=update_objective_plot,
            inputs=[lp_name_dropdown, objective_dropdown],
            outputs=plot_component
        )

        # Refresh button updates the plot using current dropdown values
        refresh_btn.click(
            fn=update_plots,
            inputs=[lp_name_dropdown, objective_dropdown],
            outputs=[plot_component, parallel_coordinates_plot_component, pareto_plot_component, ternary_plot_component, hypervolume_plot_component, parameter_importance_plot_component, details_table, hyperparams_code]
        )
    
    return pipeline_page


# For standalone testing
if __name__ == "__main__":
    # Load configuration from YAML file
    config = load_exp_config()
    
    # Extract parameters from config
    exp_config_name = config['common_args']['exp_config_name']
    run_num = config['common_args']['run_nums'][0]  # Take first run number
    max_trials = config['optimisation_args']['max_total_pipelines_num']
    ref_point = config['optimisation_args']['ref_point']
    
    with gr.Blocks() as demo:
        create_pipeline_optimization_page(
            exp_config_name=exp_config_name,
            run_num=run_num,
            max_trials=max_trials,
            ref_point=ref_point
        )
    demo.launch()
