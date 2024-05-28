import pandas as pd
import altair as alt
import seaborn as sns
from altair.utils.schemapi import Undefined
from virny.custom_classes.metrics_composer import MetricsComposer
from virny.configs.constants import *

from configs.constants import EXP_COLLECTION_NAME
from configs.scenarios_config import EVALUATION_SCENARIOS_CONFIG
from source.custom_classes.database_client import DatabaseClient


DB_CLIENT_2 = DatabaseClient()
DB_CLIENT_2.connect()


def get_overall_metric_from_disparity_metric(disparity_metric):
    overall_to_disparity_metric_dct = {
        # Error disparity metrics
        TPR: [EQUALIZED_ODDS_TPR],
        TNR: [EQUALIZED_ODDS_TNR],
        FPR: [EQUALIZED_ODDS_FPR],
        FNR: [EQUALIZED_ODDS_FNR],
        ACCURACY: [ACCURACY_DIFFERENCE],
        SELECTION_RATE: [STATISTICAL_PARITY_DIFFERENCE, DISPARATE_IMPACT],
        # Stability disparity metrics
        LABEL_STABILITY: [LABEL_STABILITY_RATIO, LABEL_STABILITY_DIFFERENCE],
        JITTER: [JITTER_DIFFERENCE],
        IQR: [IQR_DIFFERENCE],
        # Uncertainty disparity metrics
        STD: [STD_DIFFERENCE, STD_RATIO],
        OVERALL_UNCERTAINTY: [OVERALL_UNCERTAINTY_DIFFERENCE, OVERALL_UNCERTAINTY_RATIO],
        ALEATORIC_UNCERTAINTY: [ALEATORIC_UNCERTAINTY_DIFFERENCE, ALEATORIC_UNCERTAINTY_RATIO],
        EPISTEMIC_UNCERTAINTY: [EPISTEMIC_UNCERTAINTY_DIFFERENCE, EPISTEMIC_UNCERTAINTY_RATIO],
    }
    for overall_metric in overall_to_disparity_metric_dct.keys():
        if disparity_metric in overall_to_disparity_metric_dct[overall_metric]:
            return overall_metric


def get_baseline_models_metric_df(db_client, dataset_name: str, metric_name: str, group: str):
    query = {
        'dataset_name': dataset_name,
        'null_imputer_name': 'baseline',
        'subgroup': group,
        'metric': metric_name,
        'tag': 'OK',
    }
    metric_df = db_client.read_metric_df_from_db(collection_name=EXP_COLLECTION_NAME,
                                                 query=query)

    # Check uniqueness
    duplicates_mask = metric_df.duplicated(subset=['Exp_Pipeline_Guid', 'Model_Name', 'Subgroup', 'Metric'], keep=False)
    assert len(metric_df[duplicates_mask]) == 0, 'Metric df contains duplicates'

    columns_subset = ['Dataset_Name', 'Null_Imputer_Name', 'Virny_Random_State',
                      'Model_Name', 'Subgroup', 'Metric', 'Metric_Value']
    metric_df = metric_df[columns_subset]

    return metric_df


def get_baseline_model_metrics(dataset_name: str, model_name: str, metric_name: str,
                               db_client, group: str = 'overall'):
    if group == 'overall':
        models_metric_df = get_baseline_models_metric_df(db_client=db_client,
                                                         dataset_name=dataset_name,
                                                         metric_name=metric_name,
                                                         group=group)
    else:
        overall_metric = get_overall_metric_from_disparity_metric(disparity_metric=metric_name)
        models_metric_df = get_baseline_models_disparity_metric_df(db_client=db_client,
                                                                   dataset_name=dataset_name,
                                                                   metric_name=overall_metric,
                                                                   group=group)
        models_metric_df = models_metric_df[models_metric_df['Metric'] == metric_name]
        models_metric_df = models_metric_df.rename(columns={group: 'Metric_Value'})

    models_metric_df = models_metric_df[models_metric_df['Model_Name'] == model_name]
    return models_metric_df


def get_baseline_model_median(dataset_name: str, model_name: str, metric_name: str,
                              db_client, group: str = 'overall'):
    models_metric_df = get_baseline_model_metrics(dataset_name=dataset_name,
                                                  model_name=model_name,
                                                  metric_name=metric_name,
                                                  db_client=db_client,
                                                  group=group)
    return models_metric_df['Metric_Value'].median()


def get_models_metric_df(db_client, dataset_name: str, evaluation_scenario: str,
                         metric_name: str, group: str):
    query = {
        'dataset_name': dataset_name,
        'evaluation_scenario': evaluation_scenario,
        'metric': metric_name,
        'subgroup': group,
        'tag': 'OK',
    }
    metric_df = db_client.read_metric_df_from_db(collection_name=EXP_COLLECTION_NAME,
                                                 query=query)
    if db_client.db_name == 'data_cleaning_stability_2':
        metric_df2 = DB_CLIENT_2.read_metric_df_from_db(collection_name=EXP_COLLECTION_NAME,
                                                        query=query)
        metric_df = pd.concat([metric_df, metric_df2])

    # Check uniqueness
    duplicates_mask = metric_df.duplicated(subset=['Exp_Pipeline_Guid', 'Model_Name', 'Subgroup', 'Metric', 'Test_Set_Index'], keep=False)
    assert len(metric_df[duplicates_mask]) == 0, 'Metric df contains duplicates'

    columns_subset = ['Dataset_Name', 'Null_Imputer_Name', 'Evaluation_Scenario', 'Virny_Random_State',
                      'Model_Name', 'Subgroup', 'Metric', 'Metric_Value', 'Test_Set_Index']
    metric_df = metric_df[columns_subset]

    return metric_df


def get_baseline_models_disparity_metric_df(db_client, dataset_name: str, metric_name: str, group: str):
    dis_grp_models_metric_df = get_baseline_models_metric_df(db_client=db_client,
                                                             dataset_name=dataset_name,
                                                             metric_name=metric_name,
                                                             group=group + '_dis')
    priv_grp_models_metric_df = get_baseline_models_metric_df(db_client=db_client,
                                                              dataset_name=dataset_name,
                                                              metric_name=metric_name,
                                                              group=group + '_priv')
    grp_models_metric_df = pd.concat([dis_grp_models_metric_df, priv_grp_models_metric_df])

    # Compose group metrics
    disparity_metric_df = pd.DataFrame()
    for null_imputer_name in grp_models_metric_df['Null_Imputer_Name'].unique():
        for exp_seed in grp_models_metric_df['Virny_Random_State'].unique():
            for model_name in grp_models_metric_df['Model_Name'].unique():
                model_subgroup_metrics_df = grp_models_metric_df[
                    (grp_models_metric_df['Null_Imputer_Name'] == null_imputer_name) &
                    (grp_models_metric_df['Virny_Random_State'] == exp_seed) &
                    (grp_models_metric_df['Model_Name'] == model_name)
                    ]

                # Create columns based on values in the Subgroup column
                pivoted_model_subgroup_metrics_df = model_subgroup_metrics_df.pivot(columns='Subgroup', values='Metric_Value',
                                                                                    index=[col for col in model_subgroup_metrics_df.columns
                                                                                           if col not in ('Subgroup', 'Metric_Value')]).reset_index()
                pivoted_model_subgroup_metrics_df = pivoted_model_subgroup_metrics_df.rename_axis(None, axis=1)

                metrics_composer = MetricsComposer(
                    {model_name: pivoted_model_subgroup_metrics_df},
                    sensitive_attributes_dct={group: None}
                )
                model_group_metrics_df = metrics_composer.compose_metrics()
                model_group_metrics_df['Null_Imputer_Name'] = null_imputer_name
                model_group_metrics_df['Virny_Random_State'] = exp_seed
                model_group_metrics_df['Model_Name'] = model_name

                disparity_metric_df = pd.concat([disparity_metric_df, model_group_metrics_df])

    return disparity_metric_df


def get_models_disparity_metric_df(db_client, dataset_name: str, evaluation_scenario: str, metric_name: str, group: str):
    dis_grp_models_metric_df = get_models_metric_df(db_client=db_client,
                                                    dataset_name=dataset_name,
                                                    evaluation_scenario=evaluation_scenario,
                                                    metric_name=metric_name,
                                                    group=group + '_dis')
    priv_grp_models_metric_df = get_models_metric_df(db_client=db_client,
                                                     dataset_name=dataset_name,
                                                     evaluation_scenario=evaluation_scenario,
                                                     metric_name=metric_name,
                                                     group=group + '_priv')
    grp_models_metric_df = pd.concat([dis_grp_models_metric_df, priv_grp_models_metric_df])

    # Compose group metrics
    disparity_metric_df = pd.DataFrame()
    for null_imputer_name in grp_models_metric_df['Null_Imputer_Name'].unique():
        for exp_seed in grp_models_metric_df['Virny_Random_State'].unique():
            for model_name in grp_models_metric_df['Model_Name'].unique():
                for test_set_index in grp_models_metric_df['Test_Set_Index'].unique():
                    model_subgroup_metrics_df = grp_models_metric_df[
                        (grp_models_metric_df['Null_Imputer_Name'] == null_imputer_name) &
                        (grp_models_metric_df['Virny_Random_State'] == exp_seed) &
                        (grp_models_metric_df['Model_Name'] == model_name) &
                        (grp_models_metric_df['Test_Set_Index'] == test_set_index)
                    ]

                    # Create columns based on values in the Subgroup column
                    pivoted_model_subgroup_metrics_df = model_subgroup_metrics_df.pivot(columns='Subgroup', values='Metric_Value',
                                                                                        index=[col for col in model_subgroup_metrics_df.columns
                                                                                               if col not in ('Subgroup', 'Metric_Value')]).reset_index()
                    pivoted_model_subgroup_metrics_df = pivoted_model_subgroup_metrics_df.rename_axis(None, axis=1)

                    metrics_composer = MetricsComposer(
                        {model_name: pivoted_model_subgroup_metrics_df},
                        sensitive_attributes_dct={group: None}
                    )
                    model_group_metrics_df = metrics_composer.compose_metrics()
                    model_group_metrics_df['Null_Imputer_Name'] = null_imputer_name
                    model_group_metrics_df['Evaluation_Scenario'] = evaluation_scenario
                    model_group_metrics_df['Virny_Random_State'] = exp_seed
                    model_group_metrics_df['Model_Name'] = model_name
                    model_group_metrics_df['Test_Set_Index'] = test_set_index

                    disparity_metric_df = pd.concat([disparity_metric_df, model_group_metrics_df])

    return disparity_metric_df


def create_scatter_plots_for_diff_models_and_single_eval_scenario(dataset_name: str, evaluation_scenario: str,
                                                                  metric_name: str, db_client, title: str,
                                                                  group: str = 'overall', base_font_size: int = 18,
                                                                  ylim=Undefined):
    sns.set_style("whitegrid")
    imputers_order = ['deletion', 'median-mode', 'median-dummy', 'miss_forest',
                      'k_means_clustering', 'datawig', 'automl', 'boost_clean']

    metric_name = '_'.join([c.capitalize() for c in metric_name.split('_')])
    models_metric_df = get_models_metric_df(db_client=db_client,
                                            dataset_name=dataset_name,
                                            evaluation_scenario=evaluation_scenario,
                                            metric_name=metric_name,
                                            group=group)
    # Group metric values by seed
    models_metric_df = models_metric_df.groupby(['Dataset_Name', 'Null_Imputer_Name', 'Evaluation_Scenario',
                                                 'Model_Name', 'Subgroup', 'Metric', 'Test_Set_Index']).mean().reset_index()

    models_metric_df['Test_Injection_Strategy'] = models_metric_df.apply(
        lambda row: EVALUATION_SCENARIOS_CONFIG[row['Evaluation_Scenario']]['test_injection_scenarios'][row['Test_Set_Index']][:-1],
        axis=1
    )
    metric_title = metric_name.replace('_', ' ')
    chart = (
        alt.Chart(models_metric_df).mark_circle(
            size=60
        ).encode(
            x=alt.X("Null_Imputer_Name:N",
                    title=None,
                    sort=imputers_order,
                    axis=alt.Axis(labels=False)),
            y=alt.Y(f"Metric_Value:Q",
                    title=metric_title,
                    scale=alt.Scale(zero=False, domain=ylim)),
            color=alt.Color("Null_Imputer_Name:N", title=None, sort=imputers_order),
            column=alt.Column('Test_Injection_Strategy:N',
                              title=None,
                              sort=['MCAR', 'MAR', 'MNAR'])
        ).properties(
            width=120,
            title=alt.TitleParams(text=title, fontSize=base_font_size + 6, anchor='middle', align='center', dx=40),
        )
    )

    return chart


def create_scatter_plots_for_diff_models(dataset_name: str, metric_name: str, db_client,
                                         group: str = 'overall', ylim=Undefined):
    base_font_size = 20
    base_chart1 = create_scatter_plots_for_diff_models_and_single_eval_scenario(dataset_name=dataset_name,
                                                                                evaluation_scenario='exp1_mcar3',
                                                                                title='MCAR train set',
                                                                                metric_name=metric_name,
                                                                                db_client=db_client,
                                                                                group=group,
                                                                                base_font_size=base_font_size,
                                                                                ylim=ylim)
    base_chart2 = create_scatter_plots_for_diff_models_and_single_eval_scenario(dataset_name=dataset_name,
                                                                                evaluation_scenario='exp1_mar3',
                                                                                title='MAR train set',
                                                                                metric_name=metric_name,
                                                                                db_client=db_client,
                                                                                group=group,
                                                                                base_font_size=base_font_size,
                                                                                ylim=ylim)
    base_chart3 = create_scatter_plots_for_diff_models_and_single_eval_scenario(dataset_name=dataset_name,
                                                                                evaluation_scenario='exp1_mnar3',
                                                                                title='MNAR train set',
                                                                                metric_name=metric_name,
                                                                                db_client=db_client,
                                                                                group=group,
                                                                                base_font_size=base_font_size,
                                                                                ylim=ylim)

    # Concatenate two base charts
    main_base_chart = alt.hconcat()
    main_base_chart |= base_chart1
    main_base_chart |= base_chart2
    main_base_chart |= base_chart3

    final_grid_chart = (
        main_base_chart.configure_title(
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
            symbolOffset=110,
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
            labelFontWeight='normal',
            titleFontWeight='normal',
        )
    )

    # Set a shared scale for the y-axis
    final_grid_chart = final_grid_chart.resolve_scale(y='shared')

    return final_grid_chart


def create_box_plots_for_diff_baseline_models(dataset_name: str, metric_name: str, db_client,
                                              group: str = 'overall', base_font_size: int = 18, ylim=Undefined):
    sns.set_style("whitegrid")
    null_imputer_name = 'baseline'
    models_order = ['dt_clf', 'lr_clf', 'lgbm_clf', 'rf_clf', 'mlp_clf']

    metric_name = '_'.join([c.capitalize() for c in metric_name.split('_')]) if 'equalized_odds' not in metric_name.lower() else metric_name
    if group == 'overall':
        models_metric_df = get_baseline_models_metric_df(db_client=db_client,
                                                         dataset_name=dataset_name,
                                                         metric_name=metric_name,
                                                         group=group)
    else:
        overall_metric = get_overall_metric_from_disparity_metric(disparity_metric=metric_name)
        models_metric_df = get_baseline_models_disparity_metric_df(db_client=db_client,
                                                                   dataset_name=dataset_name,
                                                                   metric_name=overall_metric,
                                                                   group=group)
        models_metric_df = models_metric_df[models_metric_df['Metric'] == metric_name]
        models_metric_df = models_metric_df.rename(columns={group: 'Metric_Value'})

    models_metric_df = models_metric_df[models_metric_df['Null_Imputer_Name'] == null_imputer_name]
    metric_title = metric_name.replace('_', ' ')
    chart = (
        alt.Chart(models_metric_df).mark_boxplot(
            ticks=True,
            median={'stroke': 'black', 'strokeWidth': 0.7},
        ).encode(
            x=alt.X("Model_Name:N",
                    title=None,
                    sort=models_order,
                    axis=alt.Axis(labels=False)),
            y=alt.Y(f"Metric_Value:Q",
                    title=metric_title,
                    scale=alt.Scale(zero=False, domain=ylim)),
            color=alt.Color("Model_Name:N", title=None, sort=models_order),
        ).properties(
            width=360,
        ).configure_title(
            fontSize=base_font_size + 2
        ).configure_legend(
            titleFontSize=base_font_size + 4,
            labelFontSize=base_font_size + 2,
            symbolStrokeWidth=10,
            labelLimit=400,
            titleLimit=300,
            columns=5,
            orient='top',
            direction='horizontal',
            titleAnchor='middle',
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
            labelFontWeight='normal',
            titleFontWeight='normal',
        )
    )

    return chart


def create_box_plots_for_diff_models_and_single_eval_scenario(dataset_name: str, evaluation_scenario: str,
                                                              null_imputer_name: str, metric_name: str, db_client,
                                                              title: str, group: str = 'overall',
                                                              base_font_size: int = 18, ylim=Undefined):
    sns.set_style("whitegrid")
    models_order = ['dt_clf', 'lr_clf', 'lgbm_clf', 'rf_clf', 'mlp_clf']

    metric_name = '_'.join([c.capitalize() for c in metric_name.split('_')]) if 'equalized_odds' not in metric_name.lower() else metric_name
    if group == 'overall':
        models_metric_df = get_models_metric_df(db_client=db_client,
                                                dataset_name=dataset_name,
                                                evaluation_scenario=evaluation_scenario,
                                                metric_name=metric_name,
                                                group=group)
    else:
        overall_metric = get_overall_metric_from_disparity_metric(disparity_metric=metric_name)
        models_metric_df = get_models_disparity_metric_df(db_client=db_client,
                                                          dataset_name=dataset_name,
                                                          evaluation_scenario=evaluation_scenario,
                                                          metric_name=overall_metric,
                                                          group=group)
        models_metric_df = models_metric_df[models_metric_df['Metric'] == metric_name]
        models_metric_df = models_metric_df.rename(columns={group: 'Metric_Value'})

    models_metric_df = models_metric_df[models_metric_df['Null_Imputer_Name'] == null_imputer_name]
    models_metric_df['Test_Injection_Strategy'] = models_metric_df.apply(
        lambda row: EVALUATION_SCENARIOS_CONFIG[row['Evaluation_Scenario']]['test_injection_scenarios'][row['Test_Set_Index']][:-1],
        axis=1
    )

    metric_title = metric_name.replace('_', ' ')
    chart = (
        alt.Chart(models_metric_df).mark_boxplot(
            ticks=True,
            median={'stroke': 'black', 'strokeWidth': 0.7},
        ).encode(
            x=alt.X("Model_Name:N",
                    title=None,
                    sort=models_order,
                    axis=alt.Axis(labels=False)),
            y=alt.Y(f"Metric_Value:Q",
                    title=metric_title,
                    scale=alt.Scale(zero=False, domain=ylim)),
            color=alt.Color("Model_Name:N", title=None, sort=models_order),
            column=alt.Column('Test_Injection_Strategy:N',
                              title=None,
                              sort=['MCAR', 'MAR', 'MNAR'])
        ).properties(
            width=120,
            title=alt.TitleParams(text=title, fontSize=base_font_size + 6, anchor='middle', align='center', dx=40),
        )
    )

    return chart


def create_box_plots_for_diff_models(dataset_name: str, null_imputer_name: str, metric_name: str, db_client,
                                     group: str = 'overall', ylim=Undefined):
    base_font_size = 20
    base_chart1 = create_box_plots_for_diff_models_and_single_eval_scenario(dataset_name=dataset_name,
                                                                            evaluation_scenario='exp1_mcar3',
                                                                            title='MCAR train set',
                                                                            null_imputer_name=null_imputer_name,
                                                                            metric_name=metric_name,
                                                                            db_client=db_client,
                                                                            group=group,
                                                                            base_font_size=base_font_size,
                                                                            ylim=ylim)
    print('Prepared a plot for an MCAR train set')
    base_chart2 = create_box_plots_for_diff_models_and_single_eval_scenario(dataset_name=dataset_name,
                                                                            evaluation_scenario='exp1_mar3',
                                                                            title='MAR train set',
                                                                            null_imputer_name=null_imputer_name,
                                                                            metric_name=metric_name,
                                                                            db_client=db_client,
                                                                            group=group,
                                                                            base_font_size=base_font_size,
                                                                            ylim=ylim)
    print('Prepared a plot for an MAR train set')
    base_chart3 = create_box_plots_for_diff_models_and_single_eval_scenario(dataset_name=dataset_name,
                                                                            evaluation_scenario='exp1_mnar3',
                                                                            title='MNAR train set',
                                                                            null_imputer_name=null_imputer_name,
                                                                            metric_name=metric_name,
                                                                            db_client=db_client,
                                                                            group=group,
                                                                            base_font_size=base_font_size,
                                                                            ylim=ylim)
    print('Prepared a plot for an MNAR train set')

    # Concatenate two base charts
    main_base_chart = alt.hconcat()
    main_base_chart |= base_chart1
    main_base_chart |= base_chart2
    main_base_chart |= base_chart3

    final_grid_chart = (
        main_base_chart.configure_title(
            fontSize=base_font_size + 2
        ).configure_legend(
            titleFontSize=base_font_size + 4,
            labelFontSize=base_font_size + 2,
            symbolStrokeWidth=10,
            labelLimit=400,
            titleLimit=300,
            columns=5,
            orient='top',
            direction='horizontal',
            titleAnchor='middle',
            symbolOffset=150,
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
            labelFontWeight='normal',
            titleFontWeight='normal',
        )
    )

    # Set a shared scale for the y-axis
    final_grid_chart = final_grid_chart.resolve_scale(y='shared')

    return final_grid_chart


def create_box_plots_for_diff_imputers_and_single_eval_scenario_v2(dataset_name: str, evaluation_scenario: str,
                                                                   model_name: str, metric_name: str, db_client,
                                                                   title: str, group: str = 'overall',
                                                                   base_font_size: int = 18, ylim=Undefined):
    sns.set_style("whitegrid")
    imputers_order = ['deletion', 'median-mode', 'median-dummy', 'miss_forest',
                      'k_means_clustering', 'datawig', 'automl', 'boost_clean']

    metric_name = '_'.join([c.capitalize() for c in metric_name.split('_')]) if 'equalized_odds' not in metric_name.lower() else metric_name
    if group == 'overall':
        models_metric_df = get_models_metric_df(db_client=db_client,
                                                dataset_name=dataset_name,
                                                evaluation_scenario=evaluation_scenario,
                                                metric_name=metric_name,
                                                group=group)
    else:
        overall_metric = get_overall_metric_from_disparity_metric(disparity_metric=metric_name)
        models_metric_df = get_models_disparity_metric_df(db_client=db_client,
                                                          dataset_name=dataset_name,
                                                          evaluation_scenario=evaluation_scenario,
                                                          metric_name=overall_metric,
                                                          group=group)
        models_metric_df = models_metric_df[models_metric_df['Metric'] == metric_name]
        models_metric_df = models_metric_df.rename(columns={group: 'Metric_Value'})

    models_metric_df = models_metric_df[models_metric_df['Model_Name'].isin([model_name, 'boost_clean'])]
    models_metric_df['Test_Injection_Strategy'] = models_metric_df.apply(
        lambda row: EVALUATION_SCENARIOS_CONFIG[row['Evaluation_Scenario']]['test_injection_scenarios'][row['Test_Set_Index']][:-1],
        axis=1
    )

    # Add a baseline median to models_metric_df to display it as a horizontal line
    baseline_median = get_baseline_model_median(dataset_name=dataset_name,
                                                model_name=model_name,
                                                metric_name=metric_name,
                                                db_client=db_client,
                                                group=group)
    models_metric_df['Baseline_Median'] = baseline_median

    metric_title = metric_name.replace('_', ' ')
    chart = (
        alt.Chart().mark_boxplot(
            ticks=True,
            median={'stroke': 'black', 'strokeWidth': 0.7},
        ).encode(
            x=alt.X("Null_Imputer_Name:N",
                    title=None,
                    sort=imputers_order,
                    axis=alt.Axis(labels=False)),
            y=alt.Y("Metric_Value:Q",
                    title=metric_title,
                    scale=alt.Scale(zero=False, domain=ylim)),
            color=alt.Color("Null_Imputer_Name:N", title=None, sort=imputers_order),
        )
    )

    horizontal_line = (
        alt.Chart().mark_rule().encode(
            y="Baseline_Median:Q",
            color=alt.value("red"),
            size=alt.value(2)
        )
    )

    final_chart = (
        alt.layer(
            chart, horizontal_line,
            data=models_metric_df,
        ).properties(
            width=120,
        ).facet(
            column=alt.Column('Test_Injection_Strategy:N',
                              title=title,
                              sort=['MCAR', 'MAR', 'MNAR']),
        )
    )

    return final_chart


def create_box_plots_for_diff_imputers_v2(dataset_name: str, model_name: str, metric_name: str, db_client,
                                          group: str = 'overall', ylim=Undefined):
    base_font_size = 20
    base_chart1 = create_box_plots_for_diff_imputers_and_single_eval_scenario_v2(dataset_name=dataset_name,
                                                                                 evaluation_scenario='exp1_mcar3',
                                                                                 title='MCAR train set',
                                                                                 model_name=model_name,
                                                                                 metric_name=metric_name,
                                                                                 db_client=db_client,
                                                                                 group=group,
                                                                                 base_font_size=base_font_size,
                                                                                 ylim=ylim)
    print('Prepared a plot for an MCAR train set')
    base_chart2 = create_box_plots_for_diff_imputers_and_single_eval_scenario_v2(dataset_name=dataset_name,
                                                                                 evaluation_scenario='exp1_mar3',
                                                                                 title='MAR train set',
                                                                                 model_name=model_name,
                                                                                 metric_name=metric_name,
                                                                                 db_client=db_client,
                                                                                 group=group,
                                                                                 base_font_size=base_font_size,
                                                                                 ylim=ylim)
    print('Prepared a plot for an MAR train set')
    base_chart3 = create_box_plots_for_diff_imputers_and_single_eval_scenario_v2(dataset_name=dataset_name,
                                                                                 evaluation_scenario='exp1_mnar3',
                                                                                 title='MNAR train set',
                                                                                 model_name=model_name,
                                                                                 metric_name=metric_name,
                                                                                 db_client=db_client,
                                                                                 group=group,
                                                                                 base_font_size=base_font_size,
                                                                                 ylim=ylim)
    print('Prepared a plot for an MNAR train set')

    # Concatenate two base charts
    main_base_chart = alt.hconcat()
    main_base_chart |= base_chart1
    main_base_chart |= base_chart2
    main_base_chart |= base_chart3

    final_grid_chart = (
        main_base_chart.configure_legend(
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
            spacing=5,
        ).configure_view(
            stroke=None
        ).configure_header(
            labelOrient='bottom',
            labelPadding=5,
            labelFontSize=base_font_size + 2,
            titleFontSize=base_font_size + 6,
        ).configure_axis(
            labelFontSize=base_font_size + 4,
            titleFontSize=base_font_size + 6,
            labelFontWeight='normal',
            titleFontWeight='normal',
        ).configure_title(
            fontSize=base_font_size + 6,
        )
    )

    # Set a shared scale for the y-axis
    final_grid_chart = final_grid_chart.resolve_scale(y='shared')

    return final_grid_chart


def get_line_bands_for_diff_imputers_and_single_test_set(models_metric_df, test_set: str, metric_name: str,
                                                         baseline_metrics_mean_df: pd.DataFrame, title: str,
                                                         base_font_size: int = 18, ylim=Undefined, with_band=True):
    imputers_order = ['deletion', 'median-mode', 'median-dummy', 'miss_forest',
                      'k_means_clustering', 'datawig', 'automl', 'boost_clean']

    title = f'{title} & {test_set} test'
    models_metric_df_for_test_set = models_metric_df[models_metric_df['Test_Injection_Strategy'] == test_set]

    line_chart = alt.Chart(models_metric_df_for_test_set).mark_line().encode(
        x=alt.X(field='Test_Error_Rate', type='quantitative', title='Test Error Rate',
                scale=alt.Scale(nice=False), axis=alt.Axis(labelExpr=f"(datum.value == 0.1) || (datum.value == 0.3) || (datum.value == 0.5) ? datum.label : ''")),
        y=alt.Y('mean(Metric_Value)', type='quantitative', title=metric_name.replace('_', ' '), scale=alt.Scale(zero=False, domain=ylim)),
        color=alt.Color('Null_Imputer_Name:N', title=None, sort=imputers_order),
    )
    if with_band:
        band_chart = alt.Chart(models_metric_df_for_test_set).mark_errorband(extent="stdev").encode(
            x=alt.X(field='Test_Error_Rate', type='quantitative', title='Test Error Rate',
                    scale=alt.Scale(nice=False), axis=alt.Axis(labelExpr=f"(datum.value == 0.1) || (datum.value == 0.3) || (datum.value == 0.5) ? datum.label : ''")),
            y=alt.Y(field='Metric_Value', type='quantitative', title=metric_name.replace('_', ' '), scale=alt.Scale(zero=False, domain=ylim)),
            color=alt.Color('Null_Imputer_Name:N', title=None, sort=imputers_order),
        )
        base_chart = (band_chart + line_chart)
    else:
        base_chart = line_chart

    # Add baseline to a base chart
    horizontal_line = (
        alt.Chart(baseline_metrics_mean_df).mark_rule(strokeDash=[10, 10]).encode(
            y="Baseline_Mean:Q",
            color=alt.value("grey"),
            size=alt.value(3)
        )
    )
    base_chart = base_chart + horizontal_line

    base_chart = base_chart.properties(
        width=200, height=200,
        title=alt.TitleParams(text=title, fontSize=base_font_size + 2),
    )

    return base_chart


def get_line_bands_for_diff_imputers_and_single_eval_scenario(dataset_name: str, evaluation_scenario: str,
                                                              model_name: str, metric_name: str, db_client,
                                                              title: str, group: str = 'overall',
                                                              base_font_size: int = 18, ylim=Undefined, with_band=True):
    sns.set_style("whitegrid")
    metric_name = '_'.join([c.capitalize() for c in metric_name.split('_')]) \
        if 'equalized_odds' not in metric_name.lower() else metric_name

    if group == 'overall':
        models_metric_df = get_models_metric_df(db_client=db_client,
                                                dataset_name=dataset_name,
                                                evaluation_scenario=evaluation_scenario,
                                                metric_name=metric_name,
                                                group=group)
    else:
        overall_metric = get_overall_metric_from_disparity_metric(disparity_metric=metric_name)
        models_metric_df = get_models_disparity_metric_df(db_client=db_client,
                                                          dataset_name=dataset_name,
                                                          evaluation_scenario=evaluation_scenario,
                                                          metric_name=overall_metric,
                                                          group=group)
        models_metric_df = models_metric_df[models_metric_df['Metric'] == metric_name]
        models_metric_df = models_metric_df.rename(columns={group: 'Metric_Value'})

    models_metric_df = models_metric_df[models_metric_df['Model_Name'].isin([model_name, 'boost_clean'])]
    models_metric_df['Test_Injection_Strategy'] = models_metric_df.apply(
        lambda row: EVALUATION_SCENARIOS_CONFIG[row['Evaluation_Scenario']]['test_injection_scenarios'][row['Test_Set_Index']][:-1],
        axis=1
    )
    models_metric_df['Test_Error_Rate'] = models_metric_df.apply(
        lambda row: 0.1 * int(EVALUATION_SCENARIOS_CONFIG[row['Evaluation_Scenario']]['test_injection_scenarios'][row['Test_Set_Index']][-1]),
        axis=1
    )

    # Add a baseline median to models_metric_df to display it as a horizontal line
    baseline_metrics_df = get_baseline_model_metrics(dataset_name=dataset_name,
                                                     model_name=model_name,
                                                     metric_name=metric_name,
                                                     db_client=db_client,
                                                     group=group)
    baseline_metrics_mean_df = pd.DataFrame({
        'Baseline_Mean': [baseline_metrics_df['Metric_Value'].mean()]
    })

    mcar_base_chart = get_line_bands_for_diff_imputers_and_single_test_set(models_metric_df=models_metric_df,
                                                                           baseline_metrics_mean_df=baseline_metrics_mean_df,
                                                                           test_set='MCAR',
                                                                           metric_name=metric_name,
                                                                           title=title,
                                                                           base_font_size=base_font_size,
                                                                           ylim=ylim,
                                                                           with_band=with_band)

    mar_base_chart = get_line_bands_for_diff_imputers_and_single_test_set(models_metric_df=models_metric_df,
                                                                          baseline_metrics_mean_df=baseline_metrics_mean_df,
                                                                          test_set='MAR',
                                                                          metric_name=metric_name,
                                                                          title=title,
                                                                          base_font_size=base_font_size,
                                                                          ylim=ylim,
                                                                          with_band=with_band)

    mnar_base_chart = get_line_bands_for_diff_imputers_and_single_test_set(models_metric_df=models_metric_df,
                                                                           baseline_metrics_mean_df=baseline_metrics_mean_df,
                                                                           test_set='MNAR',
                                                                           metric_name=metric_name,
                                                                           title=title,
                                                                           base_font_size=base_font_size,
                                                                           ylim=ylim,
                                                                           with_band=with_band)

    return mcar_base_chart, mar_base_chart, mnar_base_chart


def create_line_bands_for_diff_imputers(dataset_name: str, model_name: str, metric_name: str, db_client,
                                        group: str = 'overall', ylim=Undefined, with_band=True):
    base_font_size = 20
    mcar_base_chart1, mar_base_chart1, mnar_base_chart1 = (
        get_line_bands_for_diff_imputers_and_single_eval_scenario(dataset_name=dataset_name,
                                                                  evaluation_scenario='exp2_3_mcar3',
                                                                  title='MCAR train',
                                                                  model_name=model_name,
                                                                  metric_name=metric_name,
                                                                  db_client=db_client,
                                                                  group=group,
                                                                  base_font_size=base_font_size,
                                                                  ylim=ylim,
                                                                  with_band=with_band))
    print('Prepared a plot for an MCAR train set')
    mcar_base_chart2, mar_base_chart2, mnar_base_chart2 = (
        get_line_bands_for_diff_imputers_and_single_eval_scenario(dataset_name=dataset_name,
                                                                  evaluation_scenario='exp2_3_mar3',
                                                                  title='MAR train',
                                                                  model_name=model_name,
                                                                  metric_name=metric_name,
                                                                  db_client=db_client,
                                                                  group=group,
                                                                  base_font_size=base_font_size,
                                                                  ylim=ylim,
                                                                  with_band=with_band))
    print('Prepared a plot for an MAR train set')
    mcar_base_chart3, mar_base_chart3, mnar_base_chart3 = (
        get_line_bands_for_diff_imputers_and_single_eval_scenario(dataset_name=dataset_name,
                                                                  evaluation_scenario='exp2_3_mnar3',
                                                                  title='MNAR train',
                                                                  model_name=model_name,
                                                                  metric_name=metric_name,
                                                                  db_client=db_client,
                                                                  group=group,
                                                                  base_font_size=base_font_size,
                                                                  ylim=ylim,
                                                                  with_band=with_band))
    print('Prepared a plot for an MNAR train set')

    # Concatenate two base charts
    main_base_chart = alt.vconcat()

    row1 = alt.hconcat()
    row1 |= mcar_base_chart1
    row1 |= mar_base_chart1
    row1 |= mnar_base_chart1

    row2 = alt.hconcat()
    row2 |= mcar_base_chart2
    row2 |= mar_base_chart2
    row2 |= mnar_base_chart2

    row3 = alt.hconcat()
    row3 |= mcar_base_chart3
    row3 |= mar_base_chart3
    row3 |= mnar_base_chart3

    main_base_chart &= row1.resolve_scale(y='shared')
    main_base_chart &= row2.resolve_scale(y='shared')
    main_base_chart &= row3.resolve_scale(y='shared')

    final_grid_chart = (
        main_base_chart.configure_axis(
            labelFontSize=base_font_size + 2,
            titleFontSize=base_font_size + 2,
            labelFontWeight='normal',
            titleFontWeight='normal',
        ).configure_title(
            fontSize=base_font_size + 2
        ).configure_legend(
            titleFontSize=base_font_size + 2,
            labelFontSize=base_font_size + 2,
            symbolStrokeWidth=10,
            labelLimit=400,
            titleLimit=300,
            columns=3,
            orient='top',
            direction='horizontal',
            titleAnchor='middle',
            symbolOffset=80,
        )
    )

    # Set a shared scale for the y-axis
    final_grid_chart = final_grid_chart.resolve_scale(y='shared')

    return final_grid_chart


def get_exp2_line_bands_for_diff_imputers_and_single_test_set(models_metric_df, test_set: str, metric_name: str, train_set: str,
                                                              base_font_size: int = 18, ylim=Undefined, with_band=True):
    imputers_order = ['deletion', 'median-mode', 'median-dummy', 'miss_forest',
                      'k_means_clustering', 'datawig', 'automl', 'boost_clean']

    title = f'{train_set} train & {test_set} test'
    models_metric_df_for_test_set = models_metric_df[models_metric_df['Test_Injection_Strategy'] == test_set]

    line_chart = alt.Chart(models_metric_df_for_test_set).mark_line().encode(
        x=alt.X(field='Train_Error_Rate', type='quantitative', title='Train Error Rate',
                scale=alt.Scale(nice=False), axis=alt.Axis(labelExpr=f"(datum.value == 0.1) || (datum.value == 0.3) || (datum.value == 0.5) ? datum.label : ''")),
        y=alt.Y('mean(Metric_Value)', type='quantitative', title=metric_name.replace('_', ' '), scale=alt.Scale(zero=False, domain=ylim)),
        color=alt.Color('Null_Imputer_Name:N', title=None, sort=imputers_order),
    )
    if with_band:
        band_chart = alt.Chart(models_metric_df_for_test_set).mark_errorband(extent="stdev").encode(
            x=alt.X(field='Train_Error_Rate', type='quantitative', title='Train Error Rate',
                    scale=alt.Scale(nice=False), axis=alt.Axis(labelExpr=f"(datum.value == 0.1) || (datum.value == 0.3) || (datum.value == 0.5) ? datum.label : ''")),
            y=alt.Y(field='Metric_Value', type='quantitative', title=metric_name.replace('_', ' '), scale=alt.Scale(zero=False, domain=ylim)),
            color=alt.Color('Null_Imputer_Name:N', title=None, sort=imputers_order),
        )
        base_chart = (band_chart + line_chart)
    else:
        base_chart = line_chart

    base_chart = base_chart.properties(
        width=200, height=200,
        title=alt.TitleParams(text=title, fontSize=base_font_size + 2),
    )

    return base_chart


def get_exp2_line_bands_for_diff_imputers_and_single_eval_scenario(dataset_name: str, evaluation_scenarios: list,
                                                                   model_name: str, metric_name: str, db_client,
                                                                   train_set: str, group: str = 'overall',
                                                                   base_font_size: int = 18, ylim=Undefined, with_band=True):
    sns.set_style("whitegrid")
    metric_name = '_'.join([c.capitalize() for c in metric_name.split('_')]) \
        if 'equalized_odds' not in metric_name.lower() else metric_name

    models_metric_df = pd.DataFrame()
    for evaluation_scenario in evaluation_scenarios:
        if group == 'overall':
            scenario_models_metric_df = get_models_metric_df(db_client=db_client,
                                                             dataset_name=dataset_name,
                                                             evaluation_scenario=evaluation_scenario,
                                                             metric_name=metric_name,
                                                             group=group)
        else:
            overall_metric = get_overall_metric_from_disparity_metric(disparity_metric=metric_name)
            scenario_models_metric_df = get_models_disparity_metric_df(db_client=db_client,
                                                                       dataset_name=dataset_name,
                                                                       evaluation_scenario=evaluation_scenario,
                                                                       metric_name=overall_metric,
                                                                       group=group)
            scenario_models_metric_df = scenario_models_metric_df[scenario_models_metric_df['Metric'] == metric_name]
            scenario_models_metric_df = scenario_models_metric_df.rename(columns={group: 'Metric_Value'})

        models_metric_df = pd.concat([models_metric_df, scenario_models_metric_df])

    models_metric_df = models_metric_df[models_metric_df['Model_Name'].isin([model_name, 'boost_clean'])]
    models_metric_df['Train_Injection_Strategy'] = models_metric_df.apply(
        lambda row: EVALUATION_SCENARIOS_CONFIG[row['Evaluation_Scenario']]['train_injection_scenario'][:-1],
        axis=1
    )
    models_metric_df = models_metric_df[models_metric_df['Train_Injection_Strategy'] == train_set]

    models_metric_df['Train_Error_Rate'] = models_metric_df.apply(
        lambda row: 0.1 * int(EVALUATION_SCENARIOS_CONFIG[row['Evaluation_Scenario']]['train_injection_scenario'][-1]),
        axis=1
    )
    models_metric_df['Test_Injection_Strategy'] = models_metric_df.apply(
        lambda row: EVALUATION_SCENARIOS_CONFIG[row['Evaluation_Scenario']]['test_injection_scenarios'][row['Test_Set_Index']][:-1],
        axis=1
    )
    models_metric_df['Test_Injection_Scenario'] = models_metric_df.apply(
        lambda row: EVALUATION_SCENARIOS_CONFIG[row['Evaluation_Scenario']]['test_injection_scenarios'][row['Test_Set_Index']],
        axis=1
    )
    models_metric_df = models_metric_df[models_metric_df['Test_Injection_Scenario'].isin(['MCAR3', 'MAR3', 'MNAR3'])]

    mcar_base_chart = get_exp2_line_bands_for_diff_imputers_and_single_test_set(models_metric_df=models_metric_df,
                                                                                test_set='MCAR',
                                                                                metric_name=metric_name,
                                                                                train_set=train_set,
                                                                                base_font_size=base_font_size,
                                                                                ylim=ylim,
                                                                                with_band=with_band)

    mar_base_chart = get_exp2_line_bands_for_diff_imputers_and_single_test_set(models_metric_df=models_metric_df,
                                                                               test_set='MAR',
                                                                               metric_name=metric_name,
                                                                               train_set=train_set,
                                                                               base_font_size=base_font_size,
                                                                               ylim=ylim,
                                                                               with_band=with_band)

    mnar_base_chart = get_exp2_line_bands_for_diff_imputers_and_single_test_set(models_metric_df=models_metric_df,
                                                                                test_set='MNAR',
                                                                                metric_name=metric_name,
                                                                                train_set=train_set,
                                                                                base_font_size=base_font_size,
                                                                                ylim=ylim,
                                                                                with_band=with_band)

    return mcar_base_chart, mar_base_chart, mnar_base_chart


def create_exp2_line_bands_for_diff_imputers(dataset_name: str, model_name: str, metric_name: str, db_client,
                                             group: str = 'overall', ylim=Undefined, with_band=True):
    base_font_size = 20
    mcar_base_chart1, mar_base_chart1, mnar_base_chart1 = (
        get_exp2_line_bands_for_diff_imputers_and_single_eval_scenario(dataset_name=dataset_name,
                                                                       evaluation_scenarios=['exp2_3_mcar1', 'exp2_3_mcar3', 'exp2_3_mcar5'],
                                                                       train_set='MCAR',
                                                                       model_name=model_name,
                                                                       metric_name=metric_name,
                                                                       db_client=db_client,
                                                                       group=group,
                                                                       base_font_size=base_font_size,
                                                                       ylim=ylim,
                                                                       with_band=with_band))
    print('Prepared a plot for an MCAR train set')
    mcar_base_chart2, mar_base_chart2, mnar_base_chart2 = (
        get_exp2_line_bands_for_diff_imputers_and_single_eval_scenario(dataset_name=dataset_name,
                                                                       evaluation_scenarios=['exp2_3_mar1', 'exp2_3_mar3', 'exp2_3_mar5'],
                                                                       train_set='MAR',
                                                                       model_name=model_name,
                                                                       metric_name=metric_name,
                                                                       db_client=db_client,
                                                                       group=group,
                                                                       base_font_size=base_font_size,
                                                                       ylim=ylim,
                                                                       with_band=with_band))
    print('Prepared a plot for an MAR train set')
    mcar_base_chart3, mar_base_chart3, mnar_base_chart3 = (
        get_exp2_line_bands_for_diff_imputers_and_single_eval_scenario(dataset_name=dataset_name,
                                                                       evaluation_scenarios=['exp2_3_mnar1', 'exp2_3_mnar3', 'exp2_3_mnar5'],
                                                                       train_set='MNAR',
                                                                       model_name=model_name,
                                                                       metric_name=metric_name,
                                                                       db_client=db_client,
                                                                       group=group,
                                                                       base_font_size=base_font_size,
                                                                       ylim=ylim,
                                                                       with_band=with_band))
    print('Prepared a plot for an MNAR train set')

    # Concatenate two base charts
    main_base_chart = alt.vconcat()

    row1 = alt.hconcat()
    row1 |= mcar_base_chart1
    row1 |= mar_base_chart1
    row1 |= mnar_base_chart1

    row2 = alt.hconcat()
    row2 |= mcar_base_chart2
    row2 |= mar_base_chart2
    row2 |= mnar_base_chart2

    row3 = alt.hconcat()
    row3 |= mcar_base_chart3
    row3 |= mar_base_chart3
    row3 |= mnar_base_chart3

    main_base_chart &= row1.resolve_scale(y='shared')
    main_base_chart &= row2.resolve_scale(y='shared')
    main_base_chart &= row3.resolve_scale(y='shared')

    final_grid_chart = (
        main_base_chart.configure_axis(
            labelFontSize=base_font_size + 2,
            titleFontSize=base_font_size + 2,
            labelFontWeight='normal',
            titleFontWeight='normal',
        ).configure_title(
            fontSize=base_font_size + 2
        ).configure_legend(
            titleFontSize=base_font_size + 2,
            labelFontSize=base_font_size + 2,
            symbolStrokeWidth=10,
            labelLimit=400,
            titleLimit=300,
            columns=3,
            orient='top',
            direction='horizontal',
            titleAnchor='middle',
            symbolOffset=80,
        )
    )

    # Set a shared scale for the y-axis
    final_grid_chart = final_grid_chart.resolve_scale(y='shared')

    return final_grid_chart
