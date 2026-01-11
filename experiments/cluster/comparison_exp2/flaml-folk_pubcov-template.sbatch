#!/bin/bash

# ====================================================================================
# Define SLURM configs
# ====================================================================================
#SBATCH --nodes=1
#SBATCH --mem=<MEM>GB
#SBATCH --cpus-per-task=<CPUS>
#SBATCH --time=48:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=<EMAIL>@nyu.edu
#SBATCH --account=pr_152_general
#SBATCH --job-name=<EXP_NAME>_<DATASET>_w<NUM_WORKERS>_<EXP_CONFIG_SUFFIX>
#SBATCH --output=%j_<EXP_NAME>_<DATASET>_w<NUM_WORKERS>_<EXP_CONFIG_SUFFIX>.out

module purge


# ====================================================================================
# Default parameters
# ====================================================================================
EXP_CONFIG_NAME=<EXP_NAME>_<DATASET>_w<NUM_WORKERS>_<EXP_CONFIG_SUFFIX>
NUM_WORKERS=<NUM_WORKERS>
EMAIL=<EMAIL>
SESSION=${SLURM_JOB_ID}_${EXP_CONFIG_NAME}


# ====================================================================================
# Define exp_config.yaml
# ====================================================================================
mkdir -p ./flaml/$SESSION/

cat <<EOL > ./flaml/$SESSION/exp_config.yaml
common_args:
  exp_config_name: "$EXP_CONFIG_NAME"
  run_nums: [<RUN_NUM>]
  secrets_path: "/home/$EMAIL/projects/virny-flow-experiments/scripts/configs/secrets.env"

pipeline_args:
  dataset: "<DATASET>"
  sensitive_attrs_for_intervention: ["SEX", "RAC1P"]
  null_imputers: []
  fairness_interventions: []
  models: ["dt_clf", "lr_clf", "rf_clf", "xgb_clf", "lgbm_clf"]

optimisation_args:
  ref_point: [0.50, 0.15, 0.15]
  objectives:
    - { name: "objective_1", metric: "F1", group: "overall", weight: <W1> }
    - { name: "objective_2", metric: "Statistical_Parity_Difference", group: "SEX", weight: <W2> }
    - { name: "objective_3", metric: "Statistical_Parity_Difference", group: "RAC1P", weight: <W3> }
  max_time_budget: <MAX_TIME_BUDGET>
  num_workers: $NUM_WORKERS
  num_pp_candidates: 2
  training_set_fractions_for_halting: [0.5, 1.0]
  exploration_factor: 0.5
  risk_factor: 0.5

virny_args:
  sensitive_attributes_dct: {'SEX': '2', 'RAC1P': ['2', '3', '4', '5', '6', '7', '8', '9'], 'SEX&RAC1P': None}
EOL


# ====================================================================================
# Start FLAML
# ====================================================================================
EXECUTION_COMMAND="
python /home/$EMAIL/projects/virny-flow-experiments/external/flaml/multi_objective_flaml_exp.py --exp-config-yaml-path ./flaml/$SESSION/exp_config.yaml --storage-path ./flaml/$SESSION/
"

singularity exec \
	    --overlay /scratch/$EMAIL/virny_flow_project/vldb_sds_env.ext3:ro \
	    /scratch/work/public/singularity/ubuntu-22.04.1.sif \
	    /bin/bash -c "source /ext3/env.sh; ${EXECUTION_COMMAND}"
