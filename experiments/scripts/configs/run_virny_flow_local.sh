#!/bin/bash

# Usage: bash run_virny_flow_local.sh <exp_name> <dataset> <num_workers> <cpus> <num_pp_candidates> <run_num> <w1> <w2> <w3> <max_total_pipelines_num>
# Example: bash run_virny_flow_local.sh age_cohorts_exp heart_cohort_a 16 16 4 1 0.5 0.5 0.0 200

if [ "$#" -ne 10 ]; then
    echo "Error: expected 10 arguments, got $#"
    echo "Usage: $0 <exp_name> <dataset> <num_workers> <cpus> <num_pp_candidates> <run_num> <w1> <w2> <w3> <max_total_pipelines_num>"
    exit 1
fi


# ====================================================================================
# Parse input arguments
# ====================================================================================
EXP_NAME=$1
DATASET=$2
NUM_WORKERS=$3
CPUS=$4
NUM_PP_CANDIDATES=$5
RUN_NUM=$6
W1=$7
W2=$8
W3=$9
MAX_TOTAL_PIPELINES_NUM=${10}


# ====================================================================================
# Default parameters
# ====================================================================================
PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
LOGS_DIR="$PROJECT_DIR/logs"
NUM_CPUS_PER_WORKER=1
EXP_CONFIG_NAME=${EXP_NAME}_${DATASET}_w${NUM_WORKERS}_w_acc_0_5_w_fair_0_5_v2
SESSION="${EXP_CONFIG_NAME}_run_${RUN_NUM}_$(date +%Y%m%d%H%M%S)"

cd "$LOGS_DIR"


# ====================================================================================
# Define exp_config.yaml
# ====================================================================================
mkdir -p ./$SESSION

cat <<EOL > ./$SESSION/exp_config.yaml
common_args:
  exp_config_name: "$EXP_CONFIG_NAME"
  run_nums: [$RUN_NUM]
  save_storage: false
  secrets_path: "$PROJECT_DIR/scripts/configs/secrets.env"

pipeline_args:
  dataset: "$DATASET"
  sensitive_attrs_for_intervention: ["gender"]
  null_imputers: []
  fairness_interventions: ["DIR", "AD"]
  models: ["dt_clf", "lr_clf", "rf_clf", "xgb_clf", "lgbm_clf"]

optimisation_args:
  ref_point: [0.4, 0.05]
  objectives:
    - { name: "objective_1", metric: "F1", group: "overall", weight: $W1 }
    - { name: "objective_2", metric: "Equalized_Odds_FNR", group: "gender", weight: $W2 }
  max_total_pipelines_num: $MAX_TOTAL_PIPELINES_NUM
  num_workers: $NUM_WORKERS
  num_pp_candidates: $NUM_PP_CANDIDATES
  training_set_fractions_for_halting: [0.5, 1.0]
  exploration_factor: 0.5
  risk_factor: 0.5

virny_args:
  sensitive_attributes_dct: {'gender': '1'}
EOL


# ====================================================================================
# Start VirnyFlow cluster
# ====================================================================================
EXP_CONFIG_YAML_PATH="$LOGS_DIR/$SESSION/exp_config.yaml"

cleanup() {
    echo -e '\nStopping all processes...'
    kill -- -$$ 2>/dev/null
    wait 2>/dev/null
    echo 'All processes stopped.'
    exit 1
}
trap cleanup SIGINT SIGTERM

echo -e 'Starting TaskManager...'
python "$PROJECT_DIR/scripts/run_task_manager.py" \
    --exp_config_yaml_path "$EXP_CONFIG_YAML_PATH" \
    --kafka_broker_address "localhost:9093" \
    > ./$SESSION/task_manager.log 2>&1 &

echo -e 'Starting Workers...'
for i in $(seq 1 $NUM_WORKERS); do
    (
        OMP_NUM_THREADS=$NUM_CPUS_PER_WORKER \
        MKL_NUM_THREADS=$NUM_CPUS_PER_WORKER \
        python "$PROJECT_DIR/scripts/run_worker.py" \
            --exp_config_yaml_path "$EXP_CONFIG_YAML_PATH" \
            --kafka_broker_address "localhost:9093"
    ) > ./$SESSION/worker_$i.log 2>&1 &
done

wait
echo "All processes finished. Logs are in: $LOGS_DIR/$SESSION"
