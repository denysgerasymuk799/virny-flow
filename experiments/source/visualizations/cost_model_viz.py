import pandas as pd
import altair as alt
from altair.utils.schemapi import Undefined

from virny_flow.visualizations.use_case_queries import get_models_disparity_metric_df, DISPARITY_METRIC_METADATA


def create_box_plot_per_cost_model(to_plot: pd.DataFrame, metric_name: str, group: str = 'overall',
                                   base_font_size: int = 22, ylim=Undefined):
    """
    Creates an Altair box plot for the specified metric from the given DataFrame.

    """
    to_plot['exp_config_name'] = to_plot['exp_config_name'].str.replace('cost_model_exp1_folk_emp_', '', regex=False)
    to_plot['exp_config_name'] = to_plot['exp_config_name'].str.replace('w_', '', regex=False)

    if group == 'overall':
        # Filter the DataFrame for the specified metric
        if metric_name in ('runtime_in_mins', 'compound_pp_quality'):
            filtered_df = to_plot[to_plot["metric"] == "F1"]
            y_column = metric_name
            title = "Runtime in Mins" if metric_name == "runtime_in_mins" else "Compound PP Quality"
        else:
            filtered_df = to_plot[to_plot["metric"] == metric_name]
            y_column = 'overall'
            title = metric_name
    else:
        filtered_df = get_models_disparity_metric_df(subgroup_metrics_df=to_plot,
                                                     disparity_metric_name=metric_name,
                                                     group=group)
        y_column = 'disparity_metric_value'

    # Create the box plot
    box_plot = alt.Chart(filtered_df).mark_boxplot(
        ticks=True,
        median={'stroke': 'black', 'strokeWidth': 0.7},
    ).encode(
        x=alt.X('exp_config_name:N', title=None, axis=alt.Axis(labelAngle=-45, labelLimit=300)),
        y=alt.Y(f'{y_column}:Q', title=title, scale=alt.Scale(zero=False, domain=ylim)),
    ).properties(
        width=400,
        height=400
    )

    final_chart = (
        box_plot.configure_view(
            stroke=None
        ).configure_axis(
            labelFontSize=base_font_size,
            titleFontSize=base_font_size + 6,
            labelFontWeight='normal',
            titleFontWeight='normal',
        ).configure_title(
            fontSize=base_font_size + 6,
        )
    )

    return final_chart


def create_box_plot_per_dataset_and_case_study(to_plot: pd.DataFrame, exp_name: str, dataset_name: str,
                                               case_study_name: str, metric_name: str, group: str = 'overall',
                                               base_font_size: int = 22, ylim=Undefined):
    """
    Creates an Altair box plot for the specified metric from the given DataFrame.
    """
    to_plot = to_plot[(to_plot['dataset_name'] == dataset_name) & (to_plot['exp_config_name'].str.contains(case_study_name, case=False, na=False))]

    to_plot['exp_config_name'] = to_plot['exp_config_name'].str.replace(exp_name + '_', '', regex=False)
    to_plot['exp_config_name'] = to_plot['exp_config_name'].str.replace(dataset_name + '_', '', regex=False)
    to_plot['exp_config_name'] = to_plot['exp_config_name'].str.replace(case_study_name + '_', '', regex=False)
    to_plot['exp_config_name'] = to_plot['exp_config_name'].str.replace('w_', '', regex=False)

    if group == 'overall':
        # Filter the DataFrame for the specified metric
        if metric_name in ('runtime_in_mins', 'compound_pp_quality'):
            filtered_df = to_plot[to_plot["metric"] == metric_name]
            y_column = metric_name
            title = "Runtime in Mins" if metric_name == "runtime_in_mins" else "Compound PP Quality"
        else:
            filtered_df = to_plot[to_plot["metric"] == metric_name]
            y_column = 'overall'
            title = metric_name
    else:
        filtered_df = get_models_disparity_metric_df(subgroup_metrics_df=to_plot,
                                                     disparity_metric_name=metric_name,
                                                     group=group)
        y_column = 'disparity_metric_value'
        if metric_name.lower().startswith('equalized_odds_'):
            title = metric_name[-3:] + 'D'
        elif metric_name.lower() == 'selection_rate_difference':
            title = 'SRD'
        else:
            title = metric_name

    # Create the box plot
    box_plot = alt.Chart(filtered_df).mark_boxplot(
        ticks=True,
        median={'stroke': 'black', 'strokeWidth': 0.7},
        color='orange',
    ).encode(
        x=alt.X('exp_config_name:N', title=None, axis=alt.Axis(labelAngle=-45, labelLimit=500)),
        y=alt.Y(f'{y_column}:Q', title=title, scale=alt.Scale(zero=False, domain=ylim)),
    ).properties(
        width=400,
        height=400
    )

    final_chart = (
        box_plot.configure_view(
            stroke=None
        ).configure_axis(
            labelFontSize=base_font_size,
            titleFontSize=base_font_size + 6,
            labelFontWeight='normal',
            titleFontWeight='normal',
        ).configure_title(
            fontSize=base_font_size + 6,
        )
    )

    return final_chart


def create_box_plot_per_dataset_and_all_case_studies(to_plot: pd.DataFrame, exp_name: str, dataset_name: str,
                                                     group_x: str, group_y: str, metric_x: str, metric_y: str,
                                                     rename_legend_dct: dict, with_configs: bool = True,
                                                     base_font_size: int = 22):
    """
    Creates an Altair box plot for the specified metric from the given DataFrame.
    """
    to_plot = to_plot[to_plot['dataset_name'] == dataset_name]

    to_plot['exp_config_name'] = to_plot['exp_config_name'].str.replace(exp_name + '_', '', regex=False)
    to_plot['exp_config_name'] = to_plot['exp_config_name'].str.replace(dataset_name + '_', '', regex=False)
    to_plot['exp_config_name'] = to_plot['exp_config_name'].str.replace('w_', '', regex=False)
    to_plot['exp_config_name'] = to_plot['exp_config_name'].str.replace('0_', '0.', regex=False)
    to_plot['exp_config_name'] = to_plot['exp_config_name'].replace(rename_legend_dct)

    titles = list()
    filtered_dfs = list()
    new_metrics = list()
    for idx, metric in enumerate([metric_x, metric_y]):
        metric_group = group_x if idx == 0 else group_y
        if metric_group == 'overall':
            # Filter the DataFrame for the specified metric
            filtered_df_for_metric = to_plot[to_plot["metric"] == metric]
            y_column = 'overall'
            metric_title = metric
        else:
            filtered_df_for_metric = get_models_disparity_metric_df(subgroup_metrics_df=to_plot,
                                                                    disparity_metric_name=metric,
                                                                    group=metric_group)
            y_column = 'disparity_metric_value'
            if metric.lower().startswith('equalized_odds_'):
                metric_title = metric[-3:] + 'D'
            elif metric.lower() == 'selection_rate_difference':
                metric_title = 'SRD'
            else:
                metric_title = metric

            metric_title += f' ({metric_group.lower()})'
            metric_title = metric_title.replace('rac1p', 'race').replace('gender', 'sex')

        metric_title = metric_title.replace('_', ' ')
        filtered_df_for_metric.rename(columns={y_column: metric_title}, inplace=True)
        filtered_dfs.append(filtered_df_for_metric)
        titles.append(metric_title)
        new_metrics.append(metric_title)

    key_columns = ["exp_config_name", "run_num", "logical_pipeline_uuid", "physical_pipeline_uuid",
                   "model_name", "dataset_name", "logical_pipeline_name"]
    filtered_df = filtered_dfs[0].merge(filtered_dfs[1], on=key_columns, how='inner')

    # Create the box plot
    sorted_legend = list(sorted(rename_legend_dct.values()))
    main_plot = alt.Chart(filtered_df).mark_point(
        filled=True,
    ).encode(
        x=alt.X(f'mean({new_metrics[0]}):Q', title=titles[0], scale=alt.Scale(zero=False)),
        y=alt.Y(f'mean({new_metrics[1]}):Q', title=titles[1], scale=alt.Scale(zero=False)),
        color=alt.Color('exp_config_name:N', title=None, sort=sorted_legend),
        size=alt.value(200),
        tooltip=["exp_config_name:N", f"mean({new_metrics[0]}):Q", f"mean({new_metrics[1]}):Q"],
    )

    # Error bars for x-axis
    error_bars_x = alt.Chart(filtered_df).mark_errorbar(
        extent='stdev',
        orient='vertical',
        ticks=True,
        thickness=3,
    ).encode(
        x=alt.X(f'mean({new_metrics[0]}):Q', title=titles[0], scale=alt.Scale(zero=False)),
        y=alt.Y(f'{new_metrics[1]}:Q', title=titles[1], scale=alt.Scale(zero=False)),
        color=alt.Color('exp_config_name:N', title=None, sort=sorted_legend),
    )

    # Error bars for y-axis
    error_bars_y = alt.Chart(filtered_df).mark_errorbar(
        extent='stdev',
        orient='horizontal',
        ticks=True,
        thickness=3,
    ).encode(
        x=alt.X(f'{new_metrics[0]}:Q', title=titles[0], scale=alt.Scale(zero=False)),
        y=alt.Y(f'mean({new_metrics[1]}):Q', title=titles[1], scale=alt.Scale(zero=False)),
        color=alt.Color('exp_config_name:N', title=None, sort=sorted_legend),
    )

    final_chart = (main_plot + error_bars_x + error_bars_y).properties(
        width=400,
        height=400
    )
    if with_configs:
        final_chart = (
            final_chart.configure_legend(
                titleFontSize=base_font_size + 4,
                labelFontSize=base_font_size + 2,
                symbolStrokeWidth=10,
                labelLimit=500,
                # columns=3,
                # orient='top',
                # direction='horizontal',
                # titleAnchor='middle',
            ).configure_view(
                stroke=None,
            ).configure_axis(
                labelFontSize=base_font_size + 4,
                titleFontSize=base_font_size + 6,
                labelFontWeight='normal',
                titleFontWeight='normal',
            ).configure_title(
                fontSize=base_font_size + 6,
            )
        )

    return final_chart


def create_concat_box_plot_per_dataset_and_all_case_studies(to_plot: pd.DataFrame, exp_name: str, dataset_name: str,
                                                            metric1: str, metric2: str, metric3: str,
                                                            group1: str, group2: str, group3: str,
                                                            rename_legend_dct: dict, base_font_size: int = 25):
    plot_pair1 = create_box_plot_per_dataset_and_all_case_studies(to_plot=to_plot,
                                                                  exp_name=exp_name,
                                                                  dataset_name=dataset_name,
                                                                  metric_x=metric1,
                                                                  group_x=group1,
                                                                  metric_y=metric2,
                                                                  group_y=group2,
                                                                  rename_legend_dct=rename_legend_dct,
                                                                  with_configs=False,
                                                                  base_font_size=base_font_size)
    plot_pair2 = create_box_plot_per_dataset_and_all_case_studies(to_plot=to_plot,
                                                                  exp_name=exp_name,
                                                                  dataset_name=dataset_name,
                                                                  metric_x=metric1,
                                                                  group_x=group1,
                                                                  metric_y=metric3,
                                                                  group_y=group3,
                                                                  rename_legend_dct=rename_legend_dct,
                                                                  with_configs=False,
                                                                  base_font_size=base_font_size)
    plot_pair3 = create_box_plot_per_dataset_and_all_case_studies(to_plot=to_plot,
                                                                  exp_name=exp_name,
                                                                  dataset_name=dataset_name,
                                                                  metric_x=metric2,
                                                                  group_x=group2,
                                                                  metric_y=metric3,
                                                                  group_y=group3,
                                                                  rename_legend_dct=rename_legend_dct,
                                                                  with_configs=False,
                                                                  base_font_size=base_font_size)

    # Concatenate box plots
    main_box_plot = alt.hconcat()
    main_box_plot |= plot_pair1
    main_box_plot |= plot_pair2
    main_box_plot |= plot_pair3

    final_chart = (
        main_box_plot.configure_legend(
            titleFontSize=base_font_size,
            # labelFontSize=base_font_size - 2, # folk-pubcov
            labelFontSize=base_font_size,
            symbolStrokeWidth=10,
            labelLimit=600,
            titleLimit=600,
            columns=3,
            orient='top',
            direction='horizontal',
            titleAnchor='middle',
            # symbolOffset=-120, # folk-pubcov
            # symbolOffset=20, # folk-emp
            symbolOffset=30, # diabetes
        ).configure_facet(
            spacing=10
        ).configure_view(
            stroke=None
        ).configure_header(
            labelOrient='bottom',
            labelPadding=5,
            labelFontSize=base_font_size + 2,
            titleFontSize=base_font_size + 2,
        ).configure_axis(
            labelFontSize=base_font_size + 4,
            titleFontSize=base_font_size + 6,
            labelLimit=400,
            titleLimit=300,
            labelFontWeight='normal',
            titleFontWeight='normal',
        )
    )
    return final_chart
