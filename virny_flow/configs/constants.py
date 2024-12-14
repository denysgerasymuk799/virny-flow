from enum import Enum
from virny.configs.constants import (STD, IQR, JITTER, LABEL_STABILITY,
                                     ALEATORIC_UNCERTAINTY, EPISTEMIC_UNCERTAINTY, OVERALL_UNCERTAINTY,
                                     LABEL_STABILITY_RATIO, LABEL_STABILITY_DIFFERENCE, IQR_DIFFERENCE,
                                     STD_DIFFERENCE, STD_RATIO, JITTER_DIFFERENCE,
                                     OVERALL_UNCERTAINTY_DIFFERENCE, OVERALL_UNCERTAINTY_RATIO,
                                     EPISTEMIC_UNCERTAINTY_DIFFERENCE, EPISTEMIC_UNCERTAINTY_RATIO,
                                     ALEATORIC_UNCERTAINTY_DIFFERENCE, ALEATORIC_UNCERTAINTY_RATIO)


DEBUG_MODE = False
INIT_RANDOM_STATE = 100
NUM_FOLDS_FOR_TUNING = 3
NO_TASKS = 'NO_TASKS'
STAGE_SEPARATOR = '&'
NO_FAIRNESS_INTERVENTION = 'NO_FAIRNESS_INTERVENTION'
STABILITY_AND_UNCERTAINTY_METRICS = [
    STD, IQR, JITTER, LABEL_STABILITY,
    ALEATORIC_UNCERTAINTY, EPISTEMIC_UNCERTAINTY, OVERALL_UNCERTAINTY,
    LABEL_STABILITY_RATIO, LABEL_STABILITY_DIFFERENCE, IQR_DIFFERENCE, STD_DIFFERENCE, STD_RATIO, JITTER_DIFFERENCE,
    OVERALL_UNCERTAINTY_DIFFERENCE, OVERALL_UNCERTAINTY_RATIO, EPISTEMIC_UNCERTAINTY_DIFFERENCE, EPISTEMIC_UNCERTAINTY_RATIO,
    ALEATORIC_UNCERTAINTY_DIFFERENCE, ALEATORIC_UNCERTAINTY_RATIO
]

# ====================================================================
# MongoDB Constants
# ====================================================================
LOGICAL_PIPELINE_SCORES_TABLE = 'logical_pipeline_scores'
PHYSICAL_PIPELINE_OBSERVATIONS_TABLE = 'physical_pipeline_observations'
ALL_EXPERIMENT_METRICS_TABLE = 'all_experiment_metrics'
TASK_QUEUE_TABLE = 'task_queue'
EXP_CONFIG_HISTORY_TABLE = 'exp_config_history'


# ====================================================================
# Kafka Constants
# ====================================================================
NEW_TASKS_QUEUE_TOPIC = 'NewTasksQueue'
COMPLETED_TASKS_QUEUE_TOPIC = 'CompletedTasksQueue'
TASK_MANAGER_CONSUMER_GROUP = "task_manager_consumer_group"
WORKER_CONSUMER_GROUP = "worker_consumer_group"


# ====================================================================
# Stage Names
# ====================================================================
class StageName(Enum):
    null_imputation = "null_imputation"
    fairness_intervention = "fairness_intervention"
    model_evaluation = "model_evaluation"

    def __str__(self):
        return self.value


# ====================================================================
# Task Statuses
# ====================================================================
class TaskStatus(Enum):
    WAITING = "WAITING"
    ASSIGNED = "ASSIGNED"
    DONE = "DONE"

    def __str__(self):
        return self.value


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
