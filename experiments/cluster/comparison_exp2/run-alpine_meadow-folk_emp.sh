# Define the list of tuples (exp_name, dataset, system_name, num_workers, cpus, mem, run_num, w1, w2, w3, exp_config_suffix, max_time_budget, email)
declare -a job_configs=(
    "comparison_exp2 folk_emp alpine_meadow 32 32 80 1 0.5 0.5 0.0 am 3600 dh3553"
    "comparison_exp2 folk_emp alpine_meadow 32 32 80 2 0.5 0.5 0.0 am 3600 dh3553"
    "comparison_exp2 folk_emp alpine_meadow 32 32 80 3 0.5 0.5 0.0 am 3600 dh3553"
    "comparison_exp2 folk_emp alpine_meadow 32 32 80 4 0.5 0.5 0.0 am 3600 dh3553"
    "comparison_exp2 folk_emp alpine_meadow 32 32 80 5 0.5 0.5 0.0 am 3600 dh3553"
    "comparison_exp2 folk_emp alpine_meadow 32 32 80 6 0.5 0.5 0.0 am 3600 dh3553"
    "comparison_exp2 folk_emp alpine_meadow 32 32 80 7 0.5 0.5 0.0 am 3600 dh3553"
    "comparison_exp2 folk_emp alpine_meadow 32 32 80 8 0.5 0.5 0.0 am 3600 dh3553"
    "comparison_exp2 folk_emp alpine_meadow 32 32 80 9 0.5 0.5 0.0 am 3600 dh3553"
    "comparison_exp2 folk_emp alpine_meadow 32 32 80 10 0.5 0.5 0.0 am 3600 dh3553"

    "comparison_exp2 folk_emp alpine_meadow 32 32 80 11 0.5 0.5 0.0 am 3600 dh3553"
    "comparison_exp2 folk_emp alpine_meadow 32 32 80 12 0.5 0.5 0.0 am 3600 dh3553"
    "comparison_exp2 folk_emp alpine_meadow 32 32 80 13 0.5 0.5 0.0 am 3600 dh3553"
    "comparison_exp2 folk_emp alpine_meadow 32 32 80 14 0.5 0.5 0.0 am 3600 dh3553"
    "comparison_exp2 folk_emp alpine_meadow 32 32 80 15 0.5 0.5 0.0 am 3600 dh3553"
    "comparison_exp2 folk_emp alpine_meadow 32 32 80 16 0.5 0.5 0.0 am 3600 dh3553"
    "comparison_exp2 folk_emp alpine_meadow 32 32 80 17 0.5 0.5 0.0 am 3600 dh3553"
    "comparison_exp2 folk_emp alpine_meadow 32 32 80 18 0.5 0.5 0.0 am 3600 dh3553"
    "comparison_exp2 folk_emp alpine_meadow 32 32 80 19 0.5 0.5 0.0 am 3600 dh3553"
    "comparison_exp2 folk_emp alpine_meadow 32 32 80 20 0.5 0.5 0.0 am 3600 dh3553"
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
