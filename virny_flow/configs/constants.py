from enum import Enum


EXPERIMENT_RUN_SEEDS = [100 * i for i in range(1, 11)]
NUM_FOLDS_FOR_TUNING = 3
NO_READY_TASK = 'NO_READY_TASK'
FINISH_EXECUTION = 'FINISH_EXECUTION'
STAGE_SEPARATOR = '&'
NO_FAIRNESS_INTERVENTION = 'NO_FAIRNESS_INTERVENTION'

# Table names
EXP_PROGRESS_TRACKING_COLLECTION_NAME = 'exp_progress_tracking'
EXP_COLLECTION_NAME = 'exp_pipeline_metrics'
MODEL_HYPER_PARAMS_COLLECTION_NAME = 'tuned_model_hyper_params'
IMPUTATION_PERFORMANCE_METRICS_COLLECTION_NAME = 'imputation_performance_metrics'


# ====================================================================
# S3 Folders
# ====================================================================
class S3Folder(Enum):
    virny_flow = 'virny_flow'  # root dir
    experiments = 'experiments'  # virny_flow/experiments/
    intermediate_state = 'intermediate_state'  # virny_flow/experiments/<EXP_CONFIG_NAME>/intermediate_state/
    artifacts = 'artifacts'  # virny_flow/experiments/<EXP_CONFIG_NAME>/artifacts/
    evaluation_scenarios = 'evaluation_scenarios'  # virny_flow/experiments/<EXP_CONFIG_NAME>/evaluation_scenarios/
    ml_pipeline_registry = 'ml_pipeline_registry'  # virny_flow/ml_pipeline_registry/

    def __str__(self):
        return self.value


# ====================================================================
# Error Types
# ====================================================================
class ErrorType(Enum):
    missing_value = 'missing_value'

    def __str__(self):
        return self.value

    @classmethod
    def has_value(cls, item):
        return item in [v.value for v in cls]


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
    DIR = 'DIR'  # Disparate Impact Remover
    LFR = 'LFR'  # Learning Fair Representations
    AD = 'AD'    # Adversarial Debiasing
    EGR = 'EGR'  # Exponentiated Gradient Reduction
    EOP = 'EOP'  # Equalized Odds Postprocessing
    ROC = 'ROC'  # Reject Option Classification

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
