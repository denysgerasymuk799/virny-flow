#!/bin/bash

NUM_WORKERS=$1
NUM_CPUS_PER_WORKER=$2
SESSION=$3
EMAIL=$4
KAFKA_BROKER_ADDRESS=$5

echo "Current hostname is: $(hostname -s)"
echo "Current directory is: $(pwd)"

# Start worker processes
echo -e 'Starting Workers...'
for i in $(seq 1 $NUM_WORKERS); do
    (
      OMP_NUM_THREADS=$NUM_CPUS_PER_WORKER MKL_NUM_THREADS=$NUM_CPUS_PER_WORKER python /home/$EMAIL/projects/virny-flow-experiments/scripts/run_worker.py --kafka_broker_address $KAFKA_BROKER_ADDRESS --exp_config_yaml_path /scratch/$EMAIL/projects/virny-flow-experiments/logs/$SESSION/exp_config.yaml
    ) > /scratch/$EMAIL/projects/virny-flow-experiments/logs/$SESSION/worker_$(hostname -s)_$i.log 2>&1 &
done

# Wait for all background processes to finish
wait
