import pandas as pd
import altair as alt
from duckdb import query as sqldf
from virny_flow.visualizations.use_case_queries import get_models_disparity_metric_df


VIRNY_FLOW = 'virny_flow'
ALPINE = 'alpine_meadow'
AUTOSKLEARN = 'autosklearn'


def display_table_with_results(system_metrics_df, system_name: str, disparity_metric_name: str, group_name: str):
    if system_name == VIRNY_FLOW:
        system_metrics_df['system_name'] = system_name
        common_cols = ['system_name', 'dataset_name', 'num_workers', 'run_num', 'exp_config_execution_time']
    else:
        # common_cols = ['system_name', 'dataset_name', 'num_workers', 'run_num', 'optimization_time', 'total_execution_time']
        common_cols = ['system_name', 'dataset_name', 'num_workers', 'run_num', 'optimization_time']

    f1_metrics_df = system_metrics_df[system_metrics_df['metric'] == 'F1']
    f1_metrics_df['F1'] = f1_metrics_df['overall']
    f1_metrics_df = f1_metrics_df[common_cols + ['F1']]

    disparity_metric_df = get_models_disparity_metric_df(system_metrics_df, disparity_metric_name, group_name)
    disparity_metric_df[disparity_metric_name] = disparity_metric_df['disparity_metric_value']
    disparity_metric_df = disparity_metric_df[common_cols + [disparity_metric_name]]

    final_metrics_df = sqldf(f"""
        SELECT t1.*, t2.{disparity_metric_name}
        FROM f1_metrics_df AS t1
        JOIN disparity_metric_df AS t2
          ON t1.run_num = t2.run_num
         AND t1.num_workers = t2.num_workers
    """).to_df()

    if system_name == VIRNY_FLOW:
        final_metrics_df = final_metrics_df[~final_metrics_df['exp_config_execution_time'].isna()]
        final_metrics_df = final_metrics_df.rename(columns={'exp_config_execution_time': 'optimization_time'})

    return final_metrics_df


def display_table_with_results_heart(system_metrics_df, system_name: str, disparity_metric_name: str, group_name: str):
    if system_name == VIRNY_FLOW:
        system_metrics_df['system_name'] = system_name
        common_cols = ['system_name', 'dataset_name', 'num_workers', 'run_num', 'exp_config_execution_time']
    else:
        # common_cols = ['system_name', 'dataset_name', 'num_workers', 'run_num', 'optimization_time', 'total_execution_time']
        common_cols = ['system_name', 'dataset_name', 'num_workers', 'run_num', 'optimization_time']

    f1_metrics_df = system_metrics_df[system_metrics_df['metric'] == 'F1']
    f1_metrics_df['F1'] = f1_metrics_df['overall']
    f1_metrics_df = f1_metrics_df[common_cols + ['F1']]

    disparity_metric_df = get_models_disparity_metric_df(system_metrics_df, disparity_metric_name, group_name)
    disparity_metric_df[disparity_metric_name] = disparity_metric_df['disparity_metric_value']
    disparity_metric_df = disparity_metric_df[common_cols + [disparity_metric_name]]

    final_metrics_df = sqldf(f"""
        SELECT t1.*, t2.{disparity_metric_name}
        FROM f1_metrics_df AS t1
        JOIN disparity_metric_df AS t2
          ON t1.run_num = t2.run_num
         AND t1.num_workers = t2.num_workers
    """).to_df()

    if system_name == VIRNY_FLOW:
        final_metrics_df = final_metrics_df[~final_metrics_df['exp_config_execution_time'].isna()]
        final_metrics_df = final_metrics_df.rename(columns={'exp_config_execution_time': 'optimization_time'})

    return final_metrics_df


def create_performance_plot(final_metrics_df, metric_name: str, base_font_size: int = 22):
    system_order = [VIRNY_FLOW, ALPINE, AUTOSKLEARN]

    box_chart = alt.Chart(
        final_metrics_df
    ).mark_boxplot(
        ticks=True,
        median={'stroke': 'white', 'strokeWidth': 0.7},
    ).encode(
        x=alt.X('num_workers:O', title='# of Workers', axis=alt.Axis(labelAngle=-45)),
        y=alt.Y(f'{metric_name}:Q', scale=alt.Scale(zero=False)),
        color=alt.Color('system_name:N', legend=None, sort=system_order),
        column=alt.Column('system_name:N', title=None, sort=system_order)
    ).resolve_scale(
        x='independent'
    ).properties(
        width=240,
        height=300,
    )

    final_chart = (
        box_chart.configure_axis(
            labelFontSize=base_font_size + 4,
            titleFontSize=base_font_size + 6,
            labelFontWeight='normal',
            titleFontWeight='normal',
        ).configure_title(
            fontSize=base_font_size + 2
        ).configure_legend(
            titleFontSize=base_font_size + 4,
            labelFontSize=base_font_size + 2,
            symbolStrokeWidth=10,
            labelLimit=400,
            titleLimit=300,
            columns=4,
            orient='top',
            direction='horizontal',
            titleAnchor='middle',
            symbolOffset=120,
        ).configure_facet(
            spacing=10
        ).configure_view(
            stroke=None
        ).configure_header(
            labelOrient='top',
            labelPadding=5,
            labelFontSize=base_font_size + 2,
            titleFontSize=base_font_size + 2,
        ).configure_axis(
            labelFontSize=base_font_size + 4,
            titleFontSize=base_font_size + 6,
            labelFontWeight='normal',
            titleFontWeight='normal',
        )
    )
    return final_chart


def create_performance_plot_v3(final_metrics_df, metric_name: str, base_font_size: int = 22):
    system_order = [VIRNY_FLOW, ALPINE, AUTOSKLEARN]

    box_chart = alt.Chart(
        final_metrics_df
    ).mark_boxplot(
        ticks=True,
        median={'stroke': 'white', 'strokeWidth': 0.7},
    ).encode(
        x=alt.X('num_pp_candidates:O', title='# of PP Candidates', axis=alt.Axis(labelAngle=-45)),
        y=alt.Y(f'{metric_name}:Q', scale=alt.Scale(zero=False)),
        color=alt.Color('system_name:N', legend=None, sort=system_order),
        column=alt.Column('system_name:N', title=None, sort=system_order)
    ).resolve_scale(
        x='independent'
    ).properties(
        width=240,
        height=300,
    )

    final_chart = (
        box_chart.configure_axis(
            labelFontSize=base_font_size + 4,
            titleFontSize=base_font_size + 6,
            labelFontWeight='normal',
            titleFontWeight='normal',
        ).configure_title(
            fontSize=base_font_size + 2
        ).configure_legend(
            titleFontSize=base_font_size + 4,
            labelFontSize=base_font_size + 2,
            symbolStrokeWidth=10,
            labelLimit=400,
            titleLimit=300,
            columns=4,
            orient='top',
            direction='horizontal',
            titleAnchor='middle',
            symbolOffset=120,
        ).configure_facet(
            spacing=10
        ).configure_view(
            stroke=None
        ).configure_header(
            labelOrient='top',
            labelPadding=5,
            labelFontSize=base_font_size + 2,
            titleFontSize=base_font_size + 2,
        ).configure_axis(
            labelFontSize=base_font_size + 4,
            titleFontSize=base_font_size + 6,
            labelFontWeight='normal',
            titleFontWeight='normal',
        )
    )
    return final_chart


def create_performance_plot_v2(final_metrics_df, metric_name: str, base_font_size: int = 22):
    system_order = [VIRNY_FLOW, ALPINE, AUTOSKLEARN]

    metric_title = (metric_name.lower().replace('equalized_odds_', '') + 'D').upper() if 'equalized_odds' in metric_name.lower() else metric_name

    # Add a dashed line for an ideal value
    if 'equalized_odds' in metric_name.lower():
        box_chart = alt.Chart().mark_boxplot(
            ticks=True,
            median={'stroke': 'white', 'strokeWidth': 0.7},
        ).encode(
            x=alt.X('system_name:N', title=None, sort=system_order, axis=alt.Axis(labels=False, labelAngle=-45)),
            y=alt.Y(f'{metric_name}:Q', title=metric_title, scale=alt.Scale(zero=False)),
            color=alt.Color('system_name:N', sort=system_order, legend=alt.Legend(title=None)),
        )

        zero_line = alt.Chart().mark_rule(
            strokeDash=[4, 4], strokeWidth=1.5, color='red',
        ).encode(
            y=alt.datum(0)
        )

        final_chart = alt.layer(
            box_chart, zero_line,
            data=final_metrics_df,
        ).facet(
            column=alt.Column('num_workers:O', title='# of Workers'),
        )
    else:
        box_chart = alt.Chart(
            final_metrics_df
        ).mark_boxplot(
            ticks=True,
            median={'stroke': 'white', 'strokeWidth': 0.7},
        ).encode(
            x=alt.X('system_name:N', title=None, sort=system_order, axis=alt.Axis(labels=False, labelAngle=-45)),
            y=alt.Y(f'{metric_name}:Q', title=metric_title, scale=alt.Scale(zero=False)),
            color=alt.Color('system_name:N', sort=system_order, legend=alt.Legend(title=None)),
            column=alt.Column('num_workers:O', title='# of Workers'),
        ).resolve_scale(
            x='shared'
        )
        final_chart = box_chart

    final_chart = (
        final_chart.configure_axis(
            labelFontSize=base_font_size + 4,
            titleFontSize=base_font_size + 6,
            labelFontWeight='normal',
            titleFontWeight='normal',
        ).configure_title(
            fontSize=base_font_size + 2
        ).configure_legend(
            titleFontSize=base_font_size + 4,
            labelFontSize=base_font_size + 2,
            symbolStrokeWidth=10,
            labelLimit=400,
            titleLimit=300,
            columns=3,
            orient='top',
            direction='horizontal',
            titleAnchor='middle',
            symbolOffset=10,
        ).configure_facet(
            spacing=20
        ).configure_view(
            stroke=None
        ).configure_header(
            labelOrient='bottom',
            titleOrient='bottom',
            labelPadding=5,
            titlePadding=5,
            labelFontSize=base_font_size + 2,
            titleFontSize=base_font_size + 2,
            labelFontWeight='normal',
            titleFontWeight='normal',
        ).configure_axis(
            labelFontSize=base_font_size + 4,
            titleFontSize=base_font_size + 6,
            labelFontWeight='normal',
            titleFontWeight='normal',
        )
    )
    return final_chart


def create_speedup_plot(one_worker_metrics_df, final_metrics_df, metric_name: str = 'optimization_time',
                        dataset: str = 'folk_emp', base_font_size: int = 22):
    # Compute mean runtime per system_name and num_workers
    avg_one_worker_metrics_df = one_worker_metrics_df.groupby(['system_name', 'dataset_name', 'num_workers'])[metric_name].mean().reset_index()
    avg_final_metrics_df = final_metrics_df.groupby(['system_name', 'dataset_name', 'num_workers'])[metric_name].mean().reset_index()

    # Add an average runtime for one worker to each system_name
    avg_one_worker_metrics_am_df = avg_one_worker_metrics_df.copy()
    avg_one_worker_metrics_am_df['system_name'] = ALPINE
    avg_one_worker_metrics_askl_df = avg_one_worker_metrics_df.copy()
    avg_one_worker_metrics_askl_df['system_name'] = AUTOSKLEARN
    to_plot = pd.concat([avg_final_metrics_df, avg_one_worker_metrics_df, avg_one_worker_metrics_am_df, avg_one_worker_metrics_askl_df])

    # Compute speedup
    baseline_time = avg_one_worker_metrics_df[metric_name].values[0]
    to_plot['baseline_time'] = baseline_time
    to_plot['speedup'] = to_plot['baseline_time'] / to_plot[metric_name]

    # Create an altair line plot with colouring
    system_order = [VIRNY_FLOW, ALPINE, AUTOSKLEARN]
    chart = alt.Chart(to_plot).mark_line(point=alt.OverlayMarkDef(filled=True, size=100)).encode(
        x=alt.X('num_workers', title='# of Workers', scale=alt.Scale(type='linear')),
        y=alt.Y('speedup', title='Speedup'),
        color=alt.Color('system_name', title=None, sort=system_order),
    )

    # Define the horizontal rules
    if dataset == 'heart':
        speedup_am_workers_32 = to_plot[(to_plot['system_name'] == ALPINE) & (to_plot['num_workers'] == 32)]['speedup'].values[0]
        speedup_askl_workers_32 = to_plot[(to_plot['system_name'] == AUTOSKLEARN) & (to_plot['num_workers'] == 16)]['speedup'].values[0]
    else:
        speedup_am_workers_32 = to_plot[(to_plot['system_name'] == ALPINE) & (to_plot['num_workers'] == 32)]['speedup'].values[0]
        speedup_askl_workers_32 = to_plot[(to_plot['system_name'] == AUTOSKLEARN) & (to_plot['num_workers'] == 32)]['speedup'].values[0]

    rule_data = pd.DataFrame({
        'speedup': [speedup_am_workers_32, speedup_askl_workers_32, None],
        'system_name': [ALPINE, AUTOSKLEARN, VIRNY_FLOW]
    })
    rules = alt.Chart(rule_data).mark_rule(strokeDash=[8, 4], strokeWidth=3).encode(
        y='speedup',
        color=alt.Color('system_name', title=None, sort=system_order)
    )

    # Define an arrow to compare virny_flow and alpine_meadow
    speedup_vf_workers_128 = to_plot[(to_plot['system_name'] == VIRNY_FLOW) & (to_plot['num_workers'] == 128)]['speedup'].values[0]
    arrow_head_y_padding = 1.0 if dataset == 'heart' else 1.5
    arrow_tail_y_padding = 0.5 if dataset == 'heart' else 1
    arrow = alt.Chart(pd.DataFrame({
        'x': [128],
        'y_start': [speedup_am_workers_32],
        'y_end': [speedup_vf_workers_128 - arrow_head_y_padding]
    })).mark_rule(
        color='black',
        strokeWidth=2,
        opacity=0.5,
    ).encode(
        x='x:Q',
        y='y_start:Q',
        y2='y_end:Q'
    )
    arrow_head = alt.Chart(pd.DataFrame({
        'x': [128],
        'y': [speedup_vf_workers_128 - arrow_head_y_padding]
    })).mark_point(
        shape='triangle-up',
        color='black',
        point=alt.OverlayMarkDef(filled=True, size=100)
    ).encode(
        x='x:Q',
        y='y:Q'
    )
    arrow_tail = alt.Chart(pd.DataFrame({
        'x': [128],
        'y': [speedup_am_workers_32 + arrow_tail_y_padding]
    })).mark_point(
        shape='triangle-down',
        color='black',
        point=alt.OverlayMarkDef(filled=True, size=100)
    ).encode(
        x='x:Q',
        y='y:Q'
    )
    # Define text to compare virny_flow and alpine_meadow
    comparison_value = round(speedup_vf_workers_128 / speedup_am_workers_32, 2)
    comparison_text = alt.Chart(pd.DataFrame({
        'x': [128],
        'y': [(speedup_vf_workers_128 - 1.5 - speedup_am_workers_32) / 2 + speedup_am_workers_32],
        'label': [f'{comparison_value}x']
    })).mark_text(
        align='right',
        baseline='middle',
        dx=-5,
        dy=5,
        fontSize=base_font_size - 2,
        fontWeight='bold',
        color='black'
    ).encode(
        x='x:Q',
        y='y:Q',
        text='label'
    )

    final_chart = (chart + rules + arrow + arrow_head + arrow_tail + comparison_text).properties(width=500, height=300)
    final_chart = (
        final_chart.configure_legend(
            titleFontSize=base_font_size,
            labelFontSize=base_font_size,
            symbolStrokeWidth=10,
            labelLimit=500,
            orient='top-left',
            offset=20,
        ).configure_view(
            stroke=None,
        ).configure_axis(
            labelFontSize=base_font_size + 2,
            titleFontSize=base_font_size + 6,
            labelFontWeight='normal',
            titleFontWeight='normal',
        ).configure_title(
            fontSize=base_font_size + 6,
        )
    )

    return final_chart
