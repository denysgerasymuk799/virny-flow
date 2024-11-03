from enum import Enum


NO_READY_TASK = 'NO_READY_TASK'
FINISH_EXECUTION = 'FINISH_EXECUTION'
STAGE_SEPARATOR = '&'
NO_FAIRNESS_INTERVENTION = 'NO_FAIRNESS_INTERVENTION'

# ====================================================================
# MongoDB Constants
# ====================================================================
TASK_QUEUE_TABLE = 'task_queue'
LOGICAL_PIPELINE_SCORES_TABLE = 'logical_pipeline_scores'


# ====================================================================
# Stage Names
# ====================================================================
class StageName(Enum):
    null_imputation = "null_imputation"
    fairness_intervention = "fairness_intervention"
    model_evaluation = "model_evaluation"

    def __str__(self):
        return self.value


STAGE_NAME_TO_STAGE_ID = {
    StageName.null_imputation.value: 1,
    StageName.fairness_intervention.value: 2,
    StageName.model_evaluation.value: 3,
}


# ====================================================================
# Task Statuses
# ====================================================================
class TaskStatus(Enum):
    BLOCKED = "BLOCKED"
    READY = "READY"
    ASSIGNED = "ASSIGNED"
    DONE = "DONE"

    def __str__(self):
        return self.value


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
