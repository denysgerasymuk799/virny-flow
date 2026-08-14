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
#SBATCH --job-name=<EXP_NAME>_<DATASET>_w<NUM_WORKERS>_<EXP_CONFIG_SUFFIX>_<RUN_NUM>
#SBATCH --output=%j_<EXP_NAME>_<DATASET>_w<NUM_WORKERS>_<EXP_CONFIG_SUFFIX>_<RUN_NUM>.out

module purge


# ====================================================================================
# Default parameters
# ====================================================================================
EXP_CONFIG_NAME=<EXP_NAME>_<DATASET>_w<NUM_WORKERS>_<EXP_CONFIG_SUFFIX>
NUM_WORKERS=<NUM_WORKERS>
EMAIL=<EMAIL>
SESSION=${SLURM_JOB_ID}_${EXP_CONFIG_NAME}_<RUN_NUM>


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
  sensitive_attrs_for_intervention: ["gender"]
  null_imputers: []
  fairness_interventions: []
  models: ["dt_clf", "lr_clf", "rf_clf", "xgb_clf", "lgbm_clf"]

optimisation_args:
  ref_point: [0.33, 0.05]
  objectives:
    - { name: "objective_1", metric: "F1", group: "overall", weight: <W1> }
    - { name: "objective_2", metric: "Equalized_Odds_TNR", group: "gender", weight: <W2> }
  max_total_pipelines_num: <MAX_TOTAL_PIPELINES_NUM>
  num_workers: $NUM_WORKERS
  num_pp_candidates: 1
  training_set_fractions_for_halting: [0.25, 0.5, 1.0]
  exploration_factor: 0.5
  risk_factor: 0.5

virny_args:
  sensitive_attributes_dct: {'gender': '1'}
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
