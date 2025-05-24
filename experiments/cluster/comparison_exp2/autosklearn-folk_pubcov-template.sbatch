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

EXP_CONFIG_NAME=<EXP_NAME>_<DATASET>_w<NUM_WORKERS>_<EXP_CONFIG_SUFFIX>

EXECUTION_COMMAND="
python ../external/auto-sklearn/examples/exps/run_exp.py ${EXP_CONFIG_NAME} <DATASET> <RUN_NUM> /scratch/<EMAIL>/projects/virny-flow-experiments/tmp/exp3 <NUM_WORKERS> none <MAX_TIME_BUDGET> none
"

singularity exec \
	    --overlay /scratch/<EMAIL>/virny_flow_project/vldb_sds_askl_env.ext3:ro \
	    /scratch/work/public/singularity/ubuntu-20.04.1.sif \
	    /bin/bash -c "source /ext3/env.sh; ${EXECUTION_COMMAND}"
