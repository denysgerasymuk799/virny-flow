"""Alpine Meadow config class."""

from alpine_meadow.utils import setup_logger, get_logger


class Config:
    """
    Config class.
    """

    def __init__(self, **kwargs):
        # budget
        self.timeout_seconds = 30

        # seed
        self.seed = 100

        # workers
        from multiprocessing import cpu_count

        self.generation_threads_num = 1
        self.evaluation_workers_num = cpu_count()
        self.evaluation_workers_reservation_rate = 0.5

        # validation
        self.train_data_size = 0.7
        self.compute_all_metrics = False

        # rule-based search space
        self.random_random_threshold = 0.3
        self.random_cost_model_threshold = 0.6
        self.enable_mutation_rules = True
        self.mutation_rules_threshold = 0.5

        # including or excluding primitives
        self.including_primitives = []
        self.excluding_primitives = []

        # search space: process feature
        self.enable_feature_processing = True
        self.explainable_feature_processing = False
        self.predict_proba = False
        self.only_tree_models = False

        # meta learning
        self.enable_meta_learning = True
        self.meta_learning_similarity_threshold = 0.6
        self.meta_learning_similar_datasets_num = 10

        # cost model (multi-armed bandit)
        self.ucb_delta = 1.0
        self.enable_learn_from_history = True
        self.history_weight = 0.2
        self.enable_cost_model = True
        self.score_threshold_candidates_num = 10

        # feature engineering
        self.enable_feature_engineering = False
        self.fe_budget_proportion = 0.5
        self.fe_hyperparams = {"verbose": False,
                               "test_weights_file_name": "weights_11_30_rf.pkl"}
        self.fe_num_engineered_datasets = 5

        # starting pipelines
        self.starting_pipelines_num = 5

        # hyper-parameter tuning (bayesian optimization)
        self.enable_bayesian_optimization = True
        self.starting_configurations_from_history_num = 1
        self.configurations_per_arm_num = 5

        # evaluation method
        self.evaluation_method = 'adaptive'

        # adaptive pipeline selection
        self.aps_minimum_slice_size = 1 << 20
        self.enable_aps_pruning = True
        self.enable_aps_curve_fitting = False

        # cross validation strategy
        self.enable_cross_validation = False
        self.cross_validation_instances_num_threshold = 50000
        self.cross_validation_strategy = 'kfold'
        self.cross_validation_k_folds_num = 5

        # ensembling
        self.enable_ensembling = False
        self.ensembling_input_pipelines_num = 3
        self.ensembling_folds_num = 3

        # threshold tuning
        self.enable_threshold_tuning = True

        # deterministic
        self.enable_deterministic = False

        # log and debug
        self.log_trace = False
        self.debug = True
        self.setup_logger = True
        self.enable_api_client = False
        # self.log_to_file = True
        self.log_to_file = False

        # initialize other parameters
        for key, value in kwargs.items():
            self.__setattr__(key, value)

        # set up logger
        if self.setup_logger:
            if self.log_to_file:
                from time import gmtime, strftime

                output_file = f'am-{strftime("%Y-%m-%d %H:%M:%S", gmtime())}.log'
                setup_logger(output_file)
            else:
                setup_logger()
        self.logger = get_logger('alpine_meadow', self.debug)
