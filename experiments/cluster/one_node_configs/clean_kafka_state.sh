rm -rf ./$SESSION/tmp/zookeeper-data/*
rm -rf ./$SESSION/tmp/zookeeper-logs/*
rm -rf ./$SESSION/tmp/kafka-logs/*
rm -f ./$SESSION/kafka-broker.log
rm -f ./$SESSION/task_manager.log
rm -f ./$SESSION/worker_*
rm -f ./zookeeper_latest.sif

echo "Cleaning Singularity cache..."
singularity cache clean
