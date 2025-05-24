#!/bin/bash

# Define variables
NUM_WORKERS=$1
NUM_NODES=$2
SESSION=$3
EMAIL=$4

NUM_PARTITIONS=$((NUM_WORKERS * NUM_NODES))
ZOOKEEPER_IMAGE="docker://bitnami/zookeeper:latest"
KAFKA_IMAGE="docker://bitnami/kafka:latest"
INIT_IMAGE="docker://confluentinc/cp-kafka:6.1.1"

KAFKA_BROKER_HOST=$(hostname -s)
ZOOKEEPER_PORT=2181
KAFKA_PORT1=9093

# Step 1: Download all images
echo "Downloading Zookeeper image..."
singularity pull zookeeper.sif $ZOOKEEPER_IMAGE

echo "Downloading Kafka image..."
singularity pull kafka.sif $KAFKA_IMAGE

echo "Downloading Confluent Kafka image with utils..."
singularity pull confluent-kafka.sif $INIT_IMAGE


# Step 2: Start Zookeeper
(
  echo "Starting Zookeeper..."
  singularity exec \
      --bind ./$SESSION/tmp/zookeeper-data:/opt/bitnami/zookeeper/data \
      --bind ./$SESSION/tmp/zookeeper-logs:/opt/bitnami/zookeeper/logs \
      --bind /home/$EMAIL/projects/virny-flow-experiments/cluster/zoo.cfg:/opt/bitnami/zookeeper/conf/zoo.cfg \
      zookeeper.sif \
      sh -c "ALLOW_ANONYMOUS_LOGIN=yes && /opt/bitnami/zookeeper/bin/zkServer.sh start"
) > ./$SESSION/zookeeper.log 2>&1 &

# Wait for Zookeeper to start
echo "Waiting for Zookeeper to initialize..."
sleep 30


# Step 3: Start Kafka Broker
(
  echo "Starting Kafka Broker..."
  singularity exec \
      --bind ./$SESSION/tmp/kafka-logs:/tmp/kafka-logs \
      --bind /home/$EMAIL/projects/virny-flow-experiments/cluster/many_node_configs/server.properties:/opt/bitnami/kafka/config/server.properties \
      kafka.sif \
      sh -c "/opt/bitnami/kafka/bin/kafka-server-start.sh /opt/bitnami/kafka/config/server.properties --override listeners=PLAINTEXT_INTERNAL://localhost:9092,PLAINTEXT_EXTERNAL://$(hostname -s):9093 --override advertised.listeners=PLAINTEXT_INTERNAL://localhost:9092,PLAINTEXT_EXTERNAL://$(hostname -s):9093"
) > ./$SESSION/kafka-broker.log 2>&1 &

# Wait for Kafka to start
echo "Waiting for Kafka Broker to initialize..."
sleep 60


# Step 4: Initialize Kafka Topics
echo "Initializing Kafka Topics..."
singularity exec \
    confluent-kafka.sif \
    sh -c "
      kafka-topics --bootstrap-server $KAFKA_BROKER_HOST:$KAFKA_PORT1 --list && \
      kafka-topics --bootstrap-server $KAFKA_BROKER_HOST:$KAFKA_PORT1 --create --if-not-exists --topic NewTasksQueue --replication-factor 1 --partitions $NUM_PARTITIONS && \
      kafka-topics --bootstrap-server $KAFKA_BROKER_HOST:$KAFKA_PORT1 --create --if-not-exists --topic CompletedTasksQueue --replication-factor 1 --partitions $NUM_PARTITIONS && \
      echo -e '\nSuccessfully created the following topics:' && \
      kafka-topics --bootstrap-server $KAFKA_BROKER_HOST:$KAFKA_PORT1 --list
    "

echo "Kafka setup completed successfully!"
