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
#SBATCH --account=torch_pr_1201_tandon_advanced
#SBATCH --job-name=<EXP_NAME>_<DATASET>_w<NUM_WORKERS>_<EXP_CONFIG_SUFFIX>_<RUN_NUM>
#SBATCH --output=%j_<EXP_NAME>_<DATASET>_w<NUM_WORKERS>_<EXP_CONFIG_SUFFIX>_<RUN_NUM>.out

module purge

EXP_CONFIG_NAME=<EXP_NAME>_<DATASET>_w<NUM_WORKERS>_<EXP_CONFIG_SUFFIX>

EXECUTION_COMMAND="
python /home/<EMAIL>/projects/virny-flow-experiments/external/alpine-meadow/tools/benchmark/exps/run_exp.py ${EXP_CONFIG_NAME} <DATASET> <RUN_NUM> <NUM_WORKERS> none <MAX_TOTAL_PIPELINES_NUM>
"

singularity exec \
	    --overlay /scratch/<EMAIL>/virny_flow_project/vldb_sds_am_env.ext3:ro \
	    --bind /scratch/<EMAIL>/folktables_data/2018:/ext3/miniconda3/lib/python3.9/site-packages/virny/datasets/2018 \
	    /share/apps/images/ubuntu-20.04.4.sif \
	    /bin/bash -c "source /ext3/env.sh; ${EXECUTION_COMMAND}"
