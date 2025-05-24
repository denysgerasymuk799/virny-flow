# ====================================================================================
# Default parameters
# ====================================================================================
EXP_CONFIG_NAME=test_many_node
NUM_NODES=2
NUM_WORKERS=4
NUM_CPUS_PER_WORKER=1
CLUSTER_TYPE=many_node_configs
EMAIL=dh3553
SESSION=${SLURM_JOB_ID}_${EXP_CONFIG_NAME}
KAFKA_BROKER_ADDRESS="$(hostname -s):9093"
NODES=($(scontrol show hostnames "$SLURM_NODELIST"))


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
  models: ["lr_clf", "rf_clf", "lgbm_clf"]

optimisation_args:
  ref_point: [0.20, 0.20]
  objectives:
    - { name: "objective_1", metric: "F1", group: "overall", weight: 0.5 }
    - { name: "objective_2", metric: "Equalized_Odds_TPR", group: "Gender", weight: 0.5 }
  max_trials: 100
  num_workers: $NUM_WORKERS
  num_pp_candidates: 2
  training_set_fractions_for_halting: [0.7, 0.8, 0.9, 1.0]
  exploration_factor: 0.5
  risk_factor: 0.5

virny_args:
  sensitive_attributes_dct: {'Gender': 'Female'}
EOL


# ====================================================================================
# NODE 1
# ====================================================================================
# Start Kafka
bash /home/$EMAIL/projects/virny-flow-experiments/cluster/$CLUSTER_TYPE/run_singularity_kafka.sh $NUM_WORKERS $NUM_NODES $SESSION $EMAIL

# Start virny_flow cluster on the first node
singularity exec \
	    --overlay /scratch/$EMAIL/virny_flow_project/vldb_sds_env.ext3:ro \
	    /scratch/work/public/singularity/ubuntu-20.04.1.sif \
	    /bin/bash -c "source /ext3/env.sh; bash /home/$EMAIL/projects/virny-flow-experiments/cluster/$CLUSTER_TYPE/run_virny_flow_cluster.sh $NUM_WORKERS $NUM_CPUS_PER_WORKER $SESSION $EMAIL $KAFKA_BROKER_ADDRESS"


# ====================================================================================
# NODE 2
# ====================================================================================
ssh dh3553@${NODES[1]}

# Start workers
singularity exec \
	    --overlay /scratch/$EMAIL/virny_flow_project/vldb_sds_env.ext3:ro \
	    /scratch/work/public/singularity/ubuntu-20.04.1.sif \
	    /bin/bash -c "source /ext3/env.sh; bash /home/$EMAIL/projects/virny-flow-experiments/cluster/$CLUSTER_TYPE/run_workers.sh $NUM_WORKERS $NUM_CPUS_PER_WORKER $SESSION $EMAIL $KAFKA_BROKER_ADDRESS"
