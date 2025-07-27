import gradio as gr
import pandas as pd
import datetime
import altair as alt
from typing import Optional, Dict, Any

from src.database import MongoDBClient
from src.utils.common_utils import load_yaml_content


def create_pipeline_quality_charts(db_client: Optional[MongoDBClient] = None, yaml_config: Optional[Dict[str, Any]] = None):
    """Create pipeline quality bar charts from MongoDB data"""
    if db_client is None or yaml_config is None:
        empty_df = pd.DataFrame({'x': [], 'y': []})
        return alt.Chart(empty_df).mark_text(text="Database client or config not available")

    # Get data from MongoDB
    df = db_client.get_logical_pipeline_scores(yaml_config['common_args']['exp_config_name'])

    if df.empty or 'Error' in df.columns or 'Message' in df.columns:
        empty_df = pd.DataFrame({'x': [], 'y': []})
        return alt.Chart(empty_df).mark_text(text="No data available for pipeline quality charts")

    # Check if required columns exist
    required_columns = ['pipeline_quality_mean', 'pipeline_quality_std', 'pipeline_execution_cost']
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        empty_df = pd.DataFrame({'x': [], 'y': []})
        return alt.Chart(empty_df).mark_text(text=f"Missing required columns: {missing_columns}")

    # Get metric names from config
    objectives_config = yaml_config.get('optimisation_args', {}).get('objectives', [])
    metric_names = {}
    for i, obj in enumerate(objectives_config, 1):
        metric_names[obj['name']] = obj.get('metric', f'Objective {i}')

    # Parse JSON columns
    mean_values = df['pipeline_quality_mean'].apply(pd.Series)
    std_values = df['pipeline_quality_std'].apply(pd.Series)

    for col in mean_values.columns:
        df[f'mean_{col}'] = mean_values[col]
        df[f'std_{col}'] = std_values[col]

    plot_df = df[['logical_pipeline_name', 'pipeline_execution_cost'] +
                 [col for col in df.columns if col.startswith('mean_')] +
                 [col for col in df.columns if col.startswith('std_')]]

    # Prepare data in long format (dynamic for 2 or 3 objectives)
    # Find all mean_ and std_ columns for objectives
    mean_cols = [col for col in plot_df.columns if col.startswith('mean_objective_')]
    std_cols = [col for col in plot_df.columns if col.startswith('std_objective_')]
    value_vars = mean_cols + std_cols

    long_df = pd.melt(
        plot_df.drop(columns=['pipeline_execution_cost']),
        id_vars=['logical_pipeline_name'],
        value_vars=value_vars,
        var_name='metric',
        value_name='value'
    )

    # stat: 'mean' or 'std' based on column name
    long_df['stat'] = long_df['metric'].apply(lambda x: 'mean' if x.startswith('mean_') else 'std')
    # objective: extract objective name (e.g., 'objective_1', 'objective_2', ...)
    long_df['objective'] = long_df['metric'].apply(lambda x: x.split('_', 1)[1])
    # metric_number: extract the number from 'objective_1', 'objective_2', ... for sorting
    long_df['metric_number'] = long_df['objective'].apply(lambda x: int(x.split('_')[1]) if '_' in x else 0)
    # Replace objective names with metric names from config
    long_df['objective'] = long_df['objective'].map(lambda obj: metric_names.get(obj, obj))
    # Sort by metric number to maintain consistent order
    long_df = long_df.sort_values('metric_number')

    pivot_df = long_df.pivot_table(
        index=['logical_pipeline_name', 'objective'],
        columns='stat',
        values='value'
    ).reset_index()

    # Adjust std values to meet the constraints
    pivot_df['std'] = pivot_df.apply(
        lambda row: min(row['std'], 1.0 - row['mean']) if row['mean'] + row['std'] > 1.0
        else min(row['std'], row['mean']) if row['mean'] - row['std'] < 0.0
        else row['std'],
        axis=1
    )

    # Main quality plot
    base_font_size = 16
    # Get the sorted order of objectives for consistent display
    objective_order = long_df['objective'].unique().tolist()
    
    bars = alt.Chart(pivot_df).mark_bar().encode(
        y=alt.Y('objective:N', title=None,
               axis=alt.Axis(labelFontSize=base_font_size, titleFontSize=base_font_size + 2, labels=False),
               sort=objective_order),
        x=alt.X('mean:Q', title='Average Objectives (Maximized)',
               axis=alt.Axis(labelFontSize=base_font_size, titleFontSize=base_font_size + 2)),
        color=alt.Color('objective:N', legend=alt.Legend(title=None), sort=objective_order),
    )

    error_bars = alt.Chart(pivot_df).mark_errorbar(size=3).encode(
        y=alt.Y('objective:N', title=None, sort=objective_order),
        x=alt.X('mean:Q', title='Average Objectives (Maximized)'),
        xError='std:Q',
    )

    quality_chart = alt.layer(bars, error_bars).facet(
        row=alt.Row('logical_pipeline_name:N', title=None,
                    header=alt.Header(labelFontSize=base_font_size,
                                      titleFontSize=base_font_size + 2,
                                      labelOrient='left',
                                      labelAngle=0,
                                      labelAnchor='start',
                                      labelPadding=10,
                                      labelLimit=400))
    ).resolve_scale(color='shared')

    # Pipeline execution cost plot
    cost_df = df[['logical_pipeline_name', 'pipeline_execution_cost']].drop_duplicates()

    cost_chart = alt.Chart(cost_df).mark_bar().encode(
        y=alt.Y('logical_pipeline_name:N', title=None,
               axis=alt.Axis(labelFontSize=base_font_size, titleFontSize=base_font_size + 2)),
        x=alt.X('pipeline_execution_cost:Q', title='Pipeline Execution Cost, seconds',
               axis=alt.Axis(labelFontSize=base_font_size, titleFontSize=base_font_size + 2)),
        color=alt.value('steelblue')
    ).properties(height=150)

    # Concatenate vertically
    final_chart = alt.vconcat(
        quality_chart,
        cost_chart,
        spacing=30,
    ).configure_axis(
        labelLimit=400,
    ).configure_legend(
        titleFontSize=base_font_size + 2,
        labelFontSize=base_font_size,
    )

    return final_chart


def get_execution_time(db_client: Optional[MongoDBClient] = None, yaml_config: Optional[Dict[str, Any]] = None):
    """Get current execution time in minutes"""
    try:
        if db_client is None:
            return "Database client not available"
        
        # First check if progress is 100%
        current, max_pipelines, _ = get_pipeline_progress(db_client, yaml_config)
        if 0 < max_pipelines <= current:
            # Progress is 100%, read from exp_config_history table
            try:
                exp_history_df = db_client.db['exp_config_history'].find({'exp_config_name': yaml_config['common_args']['exp_config_name']})
                exp_history_list = list(exp_history_df)
                
                if exp_history_list:
                    # Get the execution time from the first record (assuming there's only one)
                    exp_config_execution_time = exp_history_list[0].get('exp_config_execution_time')
                    if exp_config_execution_time is not None:
                        # Convert from seconds to minutes and seconds
                        total_seconds = int(exp_config_execution_time)
                        minutes = total_seconds // 60
                        seconds = total_seconds % 60
                        
                        # Format the display
                        return f"{minutes}m {seconds}s"
                    else:
                        return "No execution time data in history"
                else:
                    return "No history data available"
                    
            except Exception as e:
                return f"Error reading history: {str(e)}"
        
        # Progress is not 100%, calculate from create_datetime
        lp_scores_df = db_client.get_logical_pipeline_scores(yaml_config['common_args']['exp_config_name'])
        if lp_scores_df.empty or 'Error' in lp_scores_df.columns:
            return "No data available"
        
        # Get the earliest create_datetime
        if 'create_datetime' in lp_scores_df.columns:
            # Convert to datetime if it's not already
            lp_scores_df['create_datetime'] = pd.to_datetime(lp_scores_df['create_datetime'])
            earliest_time = lp_scores_df['create_datetime'].min()
            
            # Make sure earliest_time is timezone-aware (assume UTC if naive)
            if earliest_time.tzinfo is None:
                earliest_time = earliest_time.replace(tzinfo=datetime.timezone.utc)
            
            # Calculate time difference using UTC time
            current_time = datetime.datetime.now(datetime.timezone.utc)
            time_diff = current_time - earliest_time
            
            # Convert to minutes and seconds
            total_seconds = int(time_diff.total_seconds())
            minutes = total_seconds // 60
            seconds = total_seconds % 60
            
            # Format the display
            return f"{minutes}m {seconds}s (in progress)"
        else:
            return "No execution time data available"
            
    except Exception as e:
        return f"Error calculating execution time: {str(e)}"


def get_pipeline_progress(db_client: Optional[MongoDBClient] = None, yaml_config: Optional[Dict[str, Any]] = None):
    """Get current pipeline execution progress"""
    try:
        if db_client is None:
            return 0, 0, "Database client not available"
        
        lp_scores_df = db_client.get_logical_pipeline_scores(yaml_config['common_args']['exp_config_name'])
        if lp_scores_df.empty or 'Error' in lp_scores_df.columns:
            return 0, 0, "No data available"
        
        # Filter to get only the relevant columns
        if "num_completed_pps" in lp_scores_df.columns:
            lp_scores_df = lp_scores_df[["logical_pipeline_name", "num_completed_pps", "score", "acq_type", "surrogate_type"]]
            current_pipelines = lp_scores_df['num_completed_pps'].sum()
        else:
            current_pipelines = 0
        
        # Get max pipelines from the loaded YAML config
        max_pipelines = 0
        if yaml_config:
            max_pipelines = yaml_config['optimisation_args']['max_total_pipelines_num']
        
        progress_percentage = min((current_pipelines / max_pipelines) * 100, 100.0) if max_pipelines > 0 else 0
        progress_text = f"{current_pipelines}/{max_pipelines} ({progress_percentage:.1f}%)"
        
        return current_pipelines, max_pipelines, progress_text
        
    except Exception as e:
        return 0, 0, f"Error: {str(e)}"


def create_progress_display(db_client: Optional[MongoDBClient] = None, yaml_config: Optional[Dict[str, Any]] = None):
    """Create a progress display using HTML for better visibility"""
    current, max_pipelines, progress_text = get_pipeline_progress(db_client, yaml_config)
    
    if max_pipelines > 0:
        progress_percentage = min((current / max_pipelines) * 100, 100.0)
        progress_html = f"""
        <div style="margin: 15px 0; padding: 15px; border: 1px solid #ddd; border-radius: 8px; background-color: #f9f9f9;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 10px; font-weight: bold;">
                <span>Pipeline Execution Progress</span>
                <span>{progress_text}</span>
            </div>
            <div style="width: 100%; background-color: #e0e0e0; border-radius: 10px; overflow: hidden; height: 25px;">
                <div style="width: {progress_percentage}%; height: 100%; background: linear-gradient(90deg, #4CAF50, #45a049); transition: width 0.5s ease; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 12px;">
                    {progress_percentage:.1f}%
                </div>
            </div>
            <div style="margin-top: 10px; font-size: 14px; color: #666;">
                {current} of {max_pipelines} pipelines completed
            </div>
        </div>
        """
    else:
        progress_html = f"""
        <div style="margin: 15px 0; padding: 15px; border: 1px solid #ddd; border-radius: 8px; background-color: #f9f9f9;">
            <div style="font-weight: bold; margin-bottom: 10px;">Pipeline Execution Progress</div>
            <div style="color: #666;">{current} pipelines completed</div>
        </div>
        """
    
    return progress_html


def create_execution_progress_page(db_client: Optional[MongoDBClient] = None, yaml_config: Optional[Dict[str, Any]] = None):
    """Create the execution progress page components"""
    gr.Markdown("# Execution Progress")
    
    # Add description for the refresh button
    gr.Markdown("💡 **Click the button below to refresh the progress bar with the latest data from the database**")
    
    # Create a button to manually refresh progress
    refresh_btn = gr.Button("🔄 Refresh Progress", variant="secondary")
    
    # Display the progress bar
    progress_display = gr.HTML(value=create_progress_display(db_client, yaml_config))
    
    # Display current execution time
    execution_time_display = gr.Markdown(f"**⏱️ Execution Time:** {get_execution_time(db_client, yaml_config)}")

    gr.Markdown("") # Add vertical spacing
    gr.Markdown("## Logical Pipeline Statistics")

    # Load and display MongoDB data
    if db_client is not None:
        try:
            lp_scores_df = db_client.get_logical_pipeline_scores(yaml_config['common_args']['exp_config_name'])
            lp_scores_df = lp_scores_df[["logical_pipeline_name", "num_completed_pps", "score", "acq_type", "surrogate_type"]]
        except Exception as e:
            lp_scores_df = pd.DataFrame({'Error': [f'Failed to load data: {str(e)}']})
    else:
        lp_scores_df = pd.DataFrame({'Error': ['Database client not available']})

    # Rename columns to be more human-readable
    lp_scores_df_display = lp_scores_df.copy()
    lp_column_name = 'Pipeline: null imputer -> fairness mitigation -> model'
    lp_scores_df_display.columns = [
        lp_column_name,
        'Completed Pipelines',
        'Cost Model Score',
        'Acquisition Type',
        'Surrogate Type'
    ]
    lp_scores_df_display[lp_column_name] =  lp_scores_df_display[lp_column_name].str.replace('&', ' -> ')

    # Sort the dataframe by score in descending order before displaying
    lp_scores_df_display_sorted = lp_scores_df_display.sort_values('Cost Model Score', ascending=False)
    lp_scores_df_display_sorted['Cost Model Score'] = lp_scores_df_display_sorted['Cost Model Score'].round(4)
    
    # Create the dataframe component
    pipeline_scores_df = gr.Dataframe(
        value=lp_scores_df_display_sorted,
        interactive=False,
        wrap=True,
    )
    
    # Update progress display and table when refresh is clicked
    def refresh_progress():
        try:
            # Update progress bar
            new_progress_html = create_progress_display(db_client, yaml_config)
            
            # Update execution time
            new_execution_time = get_execution_time(db_client, yaml_config)
            
            # Update pipeline scores table
            if db_client is not None:
                try:
                    new_lp_scores_df = db_client.get_logical_pipeline_scores(yaml_config['common_args']['exp_config_name'])
                    new_lp_scores_df = new_lp_scores_df[["logical_pipeline_name", "num_completed_pps", "score", "acq_type", "surrogate_type"]]
                    
                    # Rename columns to be more human-readable
                    new_lp_scores_df_display = new_lp_scores_df.copy()
                    new_lp_scores_df_display.columns = [
                        lp_column_name,
                        'Completed Pipelines',
                        'Cost Model Score',
                        'Acquisition Type',
                        'Surrogate Type'
                    ]
                    new_lp_scores_df_display[lp_column_name] = new_lp_scores_df_display[lp_column_name].str.replace('&', ' -> ')
                    
                    # Sort the dataframe by score in descending order
                    new_lp_scores_df_display_sorted = new_lp_scores_df_display.sort_values('Cost Model Score', ascending=False)
                    new_lp_scores_df_display_sorted['Cost Model Score'] = new_lp_scores_df_display_sorted['Cost Model Score'].round(4)
                    
                except Exception as e:
                    new_lp_scores_df_display_sorted = pd.DataFrame({'Error': [f'Failed to load data: {str(e)}']})
            else:
                new_lp_scores_df_display_sorted = pd.DataFrame({'Error': ['Database client not available']})
            
            # Update pipeline quality charts
            new_pipeline_quality_charts = create_pipeline_quality_charts(db_client, yaml_config)
            
            return new_progress_html, f"**⏱️ Execution Time:** {new_execution_time}", new_lp_scores_df_display_sorted, new_pipeline_quality_charts
            
        except Exception as e:
            return create_progress_display(db_client, yaml_config), f"**⏱️ Execution Time:** Error updating", lp_scores_df_display_sorted, create_pipeline_quality_charts(db_client, yaml_config)

    gr.Markdown("")

    # Create the pipeline quality charts
    pipeline_quality_charts = gr.Plot(
        create_pipeline_quality_charts(db_client, yaml_config),
    )

    refresh_btn.click(
        fn=refresh_progress,
        outputs=[progress_display, execution_time_display, pipeline_scores_df, pipeline_quality_charts]
    )
    
    gr.Markdown("")
    gr.Markdown("## Experiment Configuration")
    
    gr.Code(
        value=load_yaml_content(),
        language="yaml",
        label="Configuration File",
        lines=20,
        interactive=False
    )


# For standalone testing
if __name__ == "__main__":
    with gr.Blocks() as demo:
        create_execution_progress_page()
    demo.launch()
