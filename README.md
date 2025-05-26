# VirnyFlow: A Design Space for Responsible Model Development

<p align="center">
    <img src="./docs/virnyflow_architecture.png" alt="System Design" width="75%">
</p>

This repository contains the source code for **VirnyFlow**, a flexible and scalable framework for responsible machine learning pipeline development. VirnyFlow enables multi-stage, multi-objective optimization of ML pipelines with support for fairness, stability, and uncertainty as optimization criteria. The system is designed to facilitate human-in-the-loop workflows through a modular architecture that integrates evaluation protocol definition, Bayesian optimization and distributed execution. The repository also includes experiment configurations, execution scripts, and benchmarking pipelines used in the paper *“VirnyFlow: A Design Space for Responsible Model Development.”*

## Repository Structure

**`virny_flow/`**: Core library containing the main functionality
  - `configs/`: Configuration files, constants, and data structures
  - `core/`: Core components of the framework
    - `custom_classes/`: Custom implementations for the framework
    - `error_injectors/`: Components for injecting various types of errors into datasets
    - `fairness_interventions/`: Implementations of fairness intervention techniques
    - `null_imputers/`: Methods for imputing null values in datasets
    - `utils/`: Utility functions used throughout the framework
  - `task_manager/`: Components for distributed task management
    - `database/`: Database interaction layer
    - `domain_logic/`: Business logic for task management
  - `user_interfaces/`: Interfaces for interacting with the system
  - `visualizations/`: Components for visualization of results
  - `external_dependencies/`: External libraries and dependencies

**`virny_flow_demo/`**: Demo implementation of VirnyFlow
  - `configs/`: Configuration files for the demo
  - `docker-compose.yaml`: Docker Compose file for running the demo
  - `run_*.py`: Scripts for running different components of the system

**`experiments/`**: Scripts and configurations for experiments
  - `cluster/`: Configuration for distributed computing
  - `scripts/`: Scripts for running experiments
  - `notebooks/`: Jupyter notebooks for analysis and visualization
  - `external/`: External dependencies for experiments

**`tests/`**: Test suite for VirnyFlow
  - `custom_classes/`: Tests for custom class implementations
  - `error_injectors/`: Tests for error injection components
  - `logs/`: Test execution logs

**`docs/`**: Documentation files, including architecture diagrams

## Setup

Create a virtual environment and install requirements:
```shell
python -m venv venv 
source venv/bin/activate
pip3 install --upgrade pip3
pip3 install -r requiremnents.txt
```

Install datawig:
```shell
pip3 install mxnet-cu110
pip3 install datawig --no-deps

# In case of an import error for libcuda.so, use the command below recommended in
# https://stackoverflow.com/questions/54249577/importerror-libcuda-so-1-cannot-open-shared-object-file
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.0/compat
```

Add MongoDB secrets (optional)
```dotenv
# Create configs/secrets.env file with database variables
DB_NAME=your_mongodb_name
CONNECTION_STRING=your_mongodb_connection_string
```

Start the system for local debugging:
```shell
# Start Kafka
docker-compose up --build

# Start TaskManager in the root dir
python3 -m virny_flow_demo.run_task_manager

# Start Worker in the root dir
python3 -m virny_flow_demo.run_worker
```

Shutdown the system:
```shell
docker-compose down --volumes
#docker-compose down --volumes && docker system prune -a --volumes -f
```

## How to start VirnyFlow

```shell
/virny-flow/virny_flow_demo $ docker-compose up --build

# To stop all container use "docker-compose down --volumes"

# KAFKA_BROKER env variable should be set to localhost:9093
/virny-flow $ python3 -m virny_flow_demo.run_task_manager

/virny-flow $ python3 -m virny_flow_demo.run_worker
```

## Experiment Configuration

VirnyFlow uses YAML configuration files to define experiment parameters. These files are typically located in the `virny_flow_demo/configs/` directory. Below is an explanation of the key configuration sections and parameters:

### Configuration Structure

The experiment configuration file is divided into several sections:

```yaml
common_args:
  # General experiment settings
  
pipeline_args:
  # Dataset and model pipeline configuration
  
optimisation_args:
  # Multi-objective optimization settings
  
virny_args:
  # Fairness evaluation settings
```

### Common Arguments

```yaml
common_args:
  exp_config_name: "test_folk_emp"  # Name of the experiment configuration
  run_nums: [1]                     # Run numbers for different random seeds
  secrets_path: "path/to/secrets.env"  # Path to secrets file with database credentials
```

### Pipeline Arguments

```yaml
pipeline_args:
  dataset: "folk_emp"                        # Dataset to use
  sensitive_attrs_for_intervention: ["SEX", "RAC1P"]  # Attributes to use for fairness interventions
  null_imputers: []                          # Null imputation techniques (if any)
  fairness_interventions: []                 # Fairness intervention methods (if any)
  models: ["lr_clf", "rf_clf", "lgbm_clf"]   # ML models to evaluate
```

### Optimization Arguments

```yaml
optimisation_args:
  ref_point: [0.30, 0.10]  # Reference point for hypervolume calculation
  
  objectives:  # Multi-objective optimization targets
    - { name: "objective_1", metric: "F1", group: "overall", weight: 0.25 }
    - { name: "objective_2", metric: "Equalized_Odds_FNR", group: "SEX&RAC1P", weight: 0.75 }
  
  max_trials: 3               # Maximum number of optimization trials
  num_workers: 2              # Number of parallel workers
  num_pp_candidates: 2        # Number of preprocessing candidates to consider
  
  # Progressive training fractions to evaluate model performance
  training_set_fractions_for_halting: [0.7, 0.8, 0.9, 1.0]
  
  exploration_factor: 0.5     # Controls exploration vs. exploitation trade-off
  risk_factor: 0.5            # Controls risk tolerance in optimization
```

### Virny Arguments

```yaml
virny_args:
  # Configuration for sensitive attributes and their values
  sensitive_attributes_dct: {
    'SEX': '2', 
    'RAC1P': ['2', '3', '4', '5', '6', '7', '8', '9'], 
    'SEX&RAC1P': None
  }
```

## Usage

### MVM technique evaluation

This console command evaluates single or multiple null imputation techniques on the selected dataset. The argument `evaluation_scenarios` defines which evaluation scenarios to use. Available scenarios are listed in `configs/scenarios_config.py`, but users have an option to create own evaluation scenarios. `tune_imputers` is a bool parameter whether to tune imputers or to reuse hyper-parameters from NULL_IMPUTERS_HYPERPARAMS in `configs/null_imputers_config.py`. `save_imputed_datasets` is a bool parameter whether to save imputed datasets locally for future use. `dataset` and `null_imputers` arguments should be chosen from supported datasets and MVM techniques. `run_nums` defines run numbers for different seeds, for example, the number 3 corresponds to 300 seed defined in EXPERIMENT_RUN_SEEDS in `configs/constants.py`.
```shell
python ./scripts/impute_nulls_with_predictor.py \
    --dataset folk \
    --null_imputers [\"miss_forest\",\"datawig\"] \
    --run_nums [1,2,3] \
    --tune_imputers true \
    --save_imputed_datasets true \
    --evaluation_scenarios [\"exp1_mcar3\"]
```

### Models evaluation

This console command evaluates single or multiple null imputation techniques along with ML models training on the selected dataset. Arguments `evaluation_scenarios`, `dataset`, `null_imputers`, `run_nums` are used for the same purpose as in `impute_nulls_with_predictor.py`. `models` defines which ML models to evaluate in the pipeline. `ml_impute` is a bool argument which decides whether to impute null dynamically or use precomputed saved datasets with imputed values (if they are available).
```shell
python ./scripts/evaluate_models.py \
    --dataset folk \
    --null_imputers [\"miss_forest\",\"datawig\"] \
    --models [\"lr_clf\",\"mlp_clf\"] \
    --run_nums [1,2,3] \
    --tune_imputers true \
    --save_imputed_datasets true \
    --ml_impute true \
    --evaluation_scenarios [\"exp1_mcar3\"]
```

### Baseline evaluation

This console command evaluates ML models on clean datasets (without injected nulls) for getting baseline metrics. Arguments follow same logic as in `evaluate_models.py`.
```shell
python ./scripts/evaluate_baseline.py \
    --dataset folk \
    --models [\"lr_clf\",\"mlp_clf\"] \
    --run_nums [1,2,3]
```


## Extending the benchmark

### Adding a new dataset

1. To add a new dataset, you need to use Virny wrapper BaseFlowDataset, where reading and basic preprocessing take place
   ([link to documentation](https://dataresponsibly.github.io/Virny/examples/Multiple_Models_Interface_Use_Case/#preprocess-the-dataset-and-create-a-baseflowdataset-class)).
2. Create a `config yaml` file in `configs/yaml_files` with settings for the number of estimators, bootstrap fraction and sensitive attributes dict like in example below.
```yaml
dataset_name: folk
bootstrap_fraction: 0.8
n_estimators: 50
computation_mode: error_analysis
sensitive_attributes_dct: {'SEX': '2', 'RAC1P': ['2', '3', '4', '5', '6', '7', '8', '9'], 'SEX & RAC1P': None}
```
3. In `configs/dataset_config.py`, add a newly created wrapper for your dataset specifing kwarg arguments, test set fraction and config yaml path in the `DATASET_CONFIG` dict.


### Adding a new ML model

1. To add a new model, add the model name to `MLModels` enum in `configs/constants.py`.
2. Set up a model instance and hyper-parameters grid for tuning inside the function `get_models_params_for_tuning` in `configs/models_config_for_tuning.py`. Model instance should inherit sklearn BaseEstimator from scikit-learn in order to support logic with tuning and fitting model ([link to documentation](https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html)).


### Adding a new null imputer

1. Create a new imputation method for your imputer in `source/null_imputers/imputation_methods.py` similar to:
```python
def new_imputation_method(X_train_with_nulls: pd.DataFrame, X_tests_with_nulls_lst: list,
                          numeric_columns_with_nulls: list, categorical_columns_with_nulls: list,
                          hyperparams: dict, **kwargs):
    """
    This method imputes nulls using the new null imputer method.
    
    Arguments:
        X_train_with_nulls -- a training features df with nulls in numeric_columns_with_nulls and categorical_columns_with_nulls columns
        X_tests_with_nulls_lst -- a list of different X test dfs with nulls in numeric_columns_with_nulls and categorical_columns_with_nulls columns
        numeric_columns_with_nulls -- a list of numerical column names with nulls
        categorical_columns_with_nulls -- a list of categorical column names with nulls
        hyperparams -- a dictionary of tuned hyperparams for the null imputer
        kwargs -- all other params needed for the null imputer
    
    Returns:
        X_train_imputed (pd.DataFrame) -- a training features df with imputed columns defined in numeric_columns_with_nulls
                                          and categorical_columns_with_nulls
        X_tests_imputed_lst (list) -- a list of test features df with imputed columns defined in numeric_columns_with_nulls 
                                         and categorical_columns_with_nulls
        null_imputer_params_dct (dict) -- a dictionary where a keys is a column name with nulls, and 
                                          a value is a dictionary of null imputer parameters used to impute this column
    """
    
    # Write here either a call to the algorithm or the algorithm itself
    ...
    
    return X_train_imputed, X_tests_imputed_lst, null_imputer_params_dct
```

2. Add the configuration of your new imputer to `configs/null_imputers_config.py` to the _NULL_IMPUTERS_CONFIG_ dictionary.
3. Add your imputer name to the _ErrorRepairMethod_ enum in `configs/constants.py`.
4. [Optional] If a standard imputation pipeline does not work for a new null imputer, add a new if-statement to `source/custom_classes/benchmark.py` to the _impute_nulls method.


### Adding a new evaluation scenario

1. Add a configuration for the new _missingness scenario_ and the desired dataset to the `ERROR_INJECTION_SCENARIOS_CONFIG` dict in `configs/scenarios_config.py`. Missingness scenario should follow the structure below: `missing_features` are columns for null injection, and `setting` is a dict, specifying error rates and conditions for error injection.
```python
ACS_INCOME_DATASET: {
    "MCAR": [
        {
            'missing_features': ['WKHP', 'AGEP', 'SCHL', 'MAR'],
            'setting': {'error_rates': [0.1, 0.2, 0.3, 0.4, 0.5]},
        },
    ],
    "MAR": [
        {
            'missing_features': ['WKHP', 'SCHL'],
            'setting': {'condition': ('SEX', '2'), 'error_rates': [0.08, 0.12, 0.20, 0.28, 0.35]}
        }
    ],
    ...
}
```
2. Create a new _evaluation scenario_ with the new _missingness scenario_ in the `EVALUATION_SCENARIOS_CONFIG` dict in `configs/scenarios_config.py`. A new _missingness scenario_ can be used alone or combined with others. `train_injection_scenario` and `test_injection_scenarios` define settings of error injection for train and test sets, respectively. `test_injection_scenarios` takes a list as an input since the benchmark has an optimisation for multiple test sets.
```python
EVALUATION_SCENARIOS_CONFIG = {
    'mixed_exp': {
        'train_injection_scenario': 'MCAR1 & MAR1 & MNAR1',
        'test_injection_scenarios': ['MCAR1 & MAR1 & MNAR1'],
    },
    'exp1_mcar3': {
        'train_injection_scenario': 'MCAR3',
        'test_injection_scenarios': ['MCAR3', 'MAR3', 'MNAR3'],
    },
    ...
}
```
