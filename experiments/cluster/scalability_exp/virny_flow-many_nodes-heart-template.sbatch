#!/bin/bash

# ====================================================================================
# Define SLURM configs
# ====================================================================================
#SBATCH --nodes=<NUM_NODES>
#SBATCH --mem=<MEM>GB
#SBATCH --cpus-per-task=<CPUS>
#SBATCH --time=48:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=<EMAIL>@nyu.edu
#SBATCH --account=pr_152_general
#SBATCH --job-name=<EXP_NAME>_<DATASET>_n<NUM_NODES>_w<NUM_WORKERS>_<EXP_CONFIG_SUFFIX>
#SBATCH --output=%j_<EXP_NAME>_<DATASET>_n<NUM_NODES>_w<NUM_WORKERS>_<EXP_CONFIG_SUFFIX>.out

module purge


# ====================================================================================
# Default parameters
# ====================================================================================
EXP_CONFIG_NAME=<EXP_NAME>_<DATASET>_n<NUM_NODES>_w<NUM_WORKERS>_<EXP_CONFIG_SUFFIX>
NUM_NODES=<NUM_NODES>
NUM_WORKERS=<NUM_WORKERS>
NUM_PARTITIONS=$((NUM_NODES * NUM_WORKERS))
NUM_CPUS_PER_WORKER=1
CLUSTER_TYPE=many_node_configs
EMAIL=<EMAIL>
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
  num_workers: $NUM_PARTITIONS
  num_pp_candidates: 5
  training_set_fractions_for_halting: [0.25, 0.5, 1.0]
  exploration_factor: 0.5
  risk_factor: 0.5

virny_args:
  sensitive_attributes_dct: {'gender': '1'}
EOL


# ====================================================================================
# Define bash files for workers
# ====================================================================================
for ((i = 1; i < NUM_NODES; i++)); do
  NODE_HOSTNAME=${NODES[$i]}

  cat <<EOL > ./$SESSION/start_worker_$i.sh
#!/bin/bash

ssh $EMAIL@$NODE_HOSTNAME <<'SSH_EOF'
singularity exec \\
  --overlay /scratch/$EMAIL/virny_flow_project/vldb_sds_env.ext3:ro \\
  /scratch/work/public/singularity/ubuntu-20.04.1.sif \\
  /bin/bash -c "source /ext3/env.sh; bash /home/$EMAIL/projects/virny-flow-experiments/cluster/$CLUSTER_TYPE/run_workers.sh $NUM_WORKERS $NUM_CPUS_PER_WORKER $SESSION $EMAIL $KAFKA_BROKER_ADDRESS $NODE_HOSTNAME"
SSH_EOF

EOL
done


# ====================================================================================
# Start virny_flow cluster
# ====================================================================================
# Start Kafka
bash /home/$EMAIL/projects/virny-flow-experiments/cluster/$CLUSTER_TYPE/run_singularity_kafka.sh $NUM_WORKERS $NUM_NODES $SESSION $EMAIL

# Start secondary workers
for ((i = 1; i < NUM_NODES; i++)); do
  bash ./$SESSION/start_worker_$i.sh &
  echo "Successfully started worker $i!"
done

# Start a primary worker
echo "Starting the primary worker..."
singularity exec \
	    --overlay /scratch/$EMAIL/virny_flow_project/vldb_sds_env.ext3:ro \
	    /scratch/work/public/singularity/ubuntu-20.04.1.sif \
	    /bin/bash -c "source /ext3/env.sh; bash /home/$EMAIL/projects/virny-flow-experiments/cluster/$CLUSTER_TYPE/run_virny_flow_cluster.sh $NUM_WORKERS $NUM_CPUS_PER_WORKER $SESSION $EMAIL $KAFKA_BROKER_ADDRESS"
