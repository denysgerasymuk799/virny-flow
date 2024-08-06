from enum import Enum


EXPERIMENT_RUN_SEEDS = [100 * i for i in range(1, 11)]
NUM_FOLDS_FOR_TUNING = 3
EXP_COLLECTION_NAME = 'exp_nulls_data_cleaning'
MODEL_HYPER_PARAMS_COLLECTION_NAME = 'tuned_model_hyper_params'
IMPUTATION_PERFORMANCE_METRICS_COLLECTION_NAME = 'imputation_performance_metrics'
NO_TASKS = 'NO_TASKS'
STAGE_SEPARATOR = '&'


# ====================================================================
# Error Repair Methods
# ====================================================================
class ErrorRepairMethod(Enum):
    deletion = 'deletion'
    median_mode = 'median-mode'
    median_dummy = 'median-dummy'
    miss_forest = 'miss_forest'
    k_means_clustering = 'k_means_clustering'
    datawig = 'datawig'
    automl = 'automl'
    boost_clean = 'boost_clean'
    cp_clean = 'cp_clean'

    def __str__(self):
        return self.value


# ====================================================================
# Fairness Interventions
# ====================================================================
class FairnessIntervention(Enum):
    dir = 'DIR'  # Disparate Impact Remover
    lfr = 'LFR'  # Learning Fair Representations
    ad = 'AD'    # Adversarial Debiasing
    egr = 'EGR'  # Exponentiated Gradient Reduction
    eop = 'EOP'  # Equalized Odds Postprocessing
    roc = 'ROC'  # Reject Option Classification

    def __str__(self):
        return self.value


# ====================================================================
# ML Models
# ====================================================================
class MLModels(Enum):
    dt_clf = 'dt_clf'
    lr_clf = 'lr_clf'
    lgbm_clf = 'lgbm_clf'
    rf_clf = 'rf_clf'
    mlp_clf = 'mlp_clf'

    def __str__(self):
        return self.value
