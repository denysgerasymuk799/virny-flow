#!/bin/bash

# ====================================================================================
# Define SLURM configs
# ====================================================================================
#SBATCH --nodes=1
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=36
#SBATCH --time=48:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=dh3553@nyu.edu
#SBATCH --account=pr_152_general
#SBATCH --job-name=test_diabetes
#SBATCH --output=%j_test_diabetes.out

module purge


# ====================================================================================
# Default parameters
# ====================================================================================
EXP_CONFIG_NAME=test_diabetes
NUM_WORKERS=8
NUM_CPUS_PER_WORKER=4
CLUSTER_TYPE=one_node_configs  # one_node_configs or many_node_configs
EMAIL=dh3553
SESSION=${SLURM_JOB_ID}_${EXP_CONFIG_NAME}


# ====================================================================================
# Define exp_config.yaml
# ====================================================================================
mkdir ./$SESSION
mkdir -p ./$SESSION/tmp/zookeeper-data/
mkdir -p ./$SESSION/tmp/zookeeper-logs/
mkdir -p ./$SESSION/tmp/kafka-logs/

cat <<EOL > ./$SESSION/exp_config.yaml
common_args:
  exp_config_name: "$EXP_CONFIG_NAME"
  run_nums: [1]
  secrets_path: "/home/$EMAIL/projects/virny-flow-experiments/scripts/configs/secrets.env"

pipeline_args:
  dataset: "diabetes"
  sensitive_attrs_for_intervention: ["Gender"]
  null_imputers: []
  fairness_interventions: []
  models: ["lr_clf", "rf_clf", "lgbm_clf", "gandalf_clf"]

optimisation_args:
  ref_point: [0.20, 0.20]
  objectives:
    - { name: "objective_1", metric: "F1", group: "overall", weight: 0.25 }
    - { name: "objective_2", metric: "Equalized_Odds_TPR", group: "Gender", weight: 0.75 }
  max_trials: 20
  num_workers: $NUM_WORKERS
  num_pp_candidates: 2
  training_set_fractions_for_halting: [0.7, 0.8, 0.9, 1.0]
  exploration_factor: 0.5
  risk_factor: 0.5

virny_args:
  sensitive_attributes_dct: {'Gender': 'Female'}
EOL


# ====================================================================================
# Start VirnyFlow cluster
# ====================================================================================
# Start Kafka
bash /home/$EMAIL/projects/virny-flow-experiments/cluster/$CLUSTER_TYPE/run_singularity_kafka.sh $NUM_WORKERS $SESSION $EMAIL

# Start virny_flow_cluster
singularity exec \
	    --overlay /scratch/$EMAIL/virny_flow_project/vldb_sds_env.ext3:ro \
	    /scratch/work/public/singularity/ubuntu-20.04.1.sif \
	    /bin/bash -c "source /ext3/env.sh; bash /home/$EMAIL/projects/virny-flow-experiments/cluster/$CLUSTER_TYPE/run_virny_flow_cluster.sh $NUM_WORKERS $NUM_CPUS_PER_WORKER $SESSION $EMAIL"
