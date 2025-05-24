# Define the list of tuples (exp_name, dataset, system_name, num_workers, cpus, mem, run_num, w1, w2, w3, exp_config_suffix, max_time_budget, email)
declare -a job_configs=(
    "comparison_exp2 folk_pubcov virny_flow 32 32 96 1 0.33 0.33 0.33 vf 3600 np2969"
    "comparison_exp2 folk_pubcov virny_flow 32 32 96 2 0.33 0.33 0.33 vf 3600 np2969"
    "comparison_exp2 folk_pubcov virny_flow 32 32 96 3 0.33 0.33 0.33 vf 3600 np2969"
    "comparison_exp2 folk_pubcov virny_flow 32 32 96 4 0.33 0.33 0.33 vf 3600 np2969"
    "comparison_exp2 folk_pubcov virny_flow 32 32 96 5 0.33 0.33 0.33 vf 3600 np2969"
    "comparison_exp2 folk_pubcov virny_flow 32 32 96 6 0.33 0.33 0.33 vf 3600 np2969"
    "comparison_exp2 folk_pubcov virny_flow 32 32 96 7 0.33 0.33 0.33 vf 3600 np2969"
    "comparison_exp2 folk_pubcov virny_flow 32 32 96 8 0.33 0.33 0.33 vf 3600 np2969"
    "comparison_exp2 folk_pubcov virny_flow 32 32 96 9 0.33 0.33 0.33 vf 3600 np2969"
    "comparison_exp2 folk_pubcov virny_flow 32 32 96 10 0.33 0.33 0.33 vf 3600 np2969"
)

# Initialize a counter
index=0

# Iterate through the array of job_configs
for job_config in "${job_configs[@]}"
do
    # Split the job_config into separate variables
    read -r exp_name dataset system_name num_workers cpus mem run_num w1 w2 w3 exp_config_suffix max_time_budget email <<< "$job_config"
    template_file="/home/${email}/projects/virny-flow-experiments/cluster/${exp_name}/${system_name}-${dataset}-template.sbatch"

    # Define the output file name
    output_file="/home/${email}/projects/virny-flow-experiments/cluster/${exp_name}/sbatch_files/${exp_name}_${dataset}_${num_workers}_${exp_config_suffix}_run_${run_num}_${index}_$(date +"%Y%m%d%H%M%S").sbatch"

    # Create an empty file
    touch $output_file

    # Use sed to replace placeholders with actual values
    sed -e "s/<EXP_NAME>/${exp_name}/g" -e "s/<DATASET>/${dataset}/g" -e "s/<NUM_WORKERS>/${num_workers}/g" -e "s/<CPUS>/${cpus}/g" -e "s/<MEM>/${mem}/g" -e "s/<RUN_NUM>/${run_num}/g" -e "s/<W1>/${w1}/g" -e "s/<W2>/${w2}/g" -e "s/<W3>/${w3}/g" -e "s/<MAX_TIME_BUDGET>/${max_time_budget}/g" -e "s/<EXP_CONFIG_SUFFIX>/${exp_config_suffix}/g" -e "s/<EMAIL>/${email}/g" $template_file > $output_file

    # Execute a SLURM job
    sbatch $output_file

    echo "Job was executed: $output_file"

    # Increment the index
    ((index++))
done
