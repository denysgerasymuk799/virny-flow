"""Model explainability."""

import shap

from alpine_meadow.backend.base import BasePipelineExecutor


def explain_by_shap(task, pipeline: BasePipelineExecutor, df):
    """
    Explain the pipeline using tree model with SHAP
    :param task:
    :param pipeline:
    :param df:
    :return:
    """

    # create explainer
    model = pipeline.primitives[-1].primitive
    explainer = shap.TreeExplainer(model)

    # compute shap values
    preprocessing_pipeline = pipeline.copy()
    preprocessing_pipeline.primitives = preprocessing_pipeline.primitives[:-1]
    new_df = preprocessing_pipeline.test([task.dataset.from_data_frame(df)]).outputs
    shap_values = explainer.shap_values(new_df, check_additivity=False)

    return shap_values
