# Define the list of tuples (exp_name, dataset, system_name, num_workers, cpus, mem, run_num, w1, w2, w3, exp_config_suffix, max_total_pipelines_num, email)
declare -a job_configs=(
    "scalability_exp folk_emp_big alpine_meadow 1 1 150 1 0.5 0.5 0.0 am 200 dh3553"
    "scalability_exp folk_emp_big alpine_meadow 1 1 150 2 0.5 0.5 0.0 am 200 dh3553"
    "scalability_exp folk_emp_big alpine_meadow 1 1 150 3 0.5 0.5 0.0 am 200 dh3553"
    "scalability_exp folk_emp_big alpine_meadow 1 1 150 4 0.5 0.5 0.0 am 200 dh3553"
    "scalability_exp folk_emp_big alpine_meadow 1 1 150 5 0.5 0.5 0.0 am 200 dh3553"
    "scalability_exp folk_emp_big alpine_meadow 1 1 150 6 0.5 0.5 0.0 am 200 dh3553"
    "scalability_exp folk_emp_big alpine_meadow 1 1 150 7 0.5 0.5 0.0 am 200 dh3553"
    "scalability_exp folk_emp_big alpine_meadow 1 1 150 8 0.5 0.5 0.0 am 200 dh3553"
    "scalability_exp folk_emp_big alpine_meadow 1 1 150 9 0.5 0.5 0.0 am 200 dh3553"
    "scalability_exp folk_emp_big alpine_meadow 1 1 150 10 0.5 0.5 0.0 am 200 dh3553"
    "scalability_exp folk_emp_big alpine_meadow 1 1 150 11 0.5 0.5 0.0 am 200 dh3553"
    "scalability_exp folk_emp_big alpine_meadow 1 1 150 12 0.5 0.5 0.0 am 200 dh3553"
    "scalability_exp folk_emp_big alpine_meadow 1 1 150 13 0.5 0.5 0.0 am 200 dh3553"
    "scalability_exp folk_emp_big alpine_meadow 1 1 150 14 0.5 0.5 0.0 am 200 dh3553"
    "scalability_exp folk_emp_big alpine_meadow 1 1 150 15 0.5 0.5 0.0 am 200 dh3553"
    "scalability_exp folk_emp_big alpine_meadow 1 1 150 16 0.5 0.5 0.0 am 200 dh3553"
    "scalability_exp folk_emp_big alpine_meadow 1 1 150 17 0.5 0.5 0.0 am 200 dh3553"
    "scalability_exp folk_emp_big alpine_meadow 1 1 150 18 0.5 0.5 0.0 am 200 dh3553"
    "scalability_exp folk_emp_big alpine_meadow 1 1 150 19 0.5 0.5 0.0 am 200 dh3553"
    "scalability_exp folk_emp_big alpine_meadow 1 1 150 20 0.5 0.5 0.0 am 200 dh3553"

#    "scalability_exp folk_emp_big alpine_meadow 2 2 150 2 0.5 0.5 0.0 am 200 dh3553"
#    "scalability_exp folk_emp_big alpine_meadow 2 2 150 3 0.5 0.5 0.0 am 200 dh3553"
#    "scalability_exp folk_emp_big alpine_meadow 2 2 150 4 0.5 0.5 0.0 am 200 dh3553"
#    "scalability_exp folk_emp_big alpine_meadow 2 2 150 5 0.5 0.5 0.0 am 200 dh3553"
#    "scalability_exp folk_emp_big alpine_meadow 2 2 150 6 0.5 0.5 0.0 am 200 dh3553"
#    "scalability_exp folk_emp_big alpine_meadow 2 2 150 7 0.5 0.5 0.0 am 200 dh3553"
#    "scalability_exp folk_emp_big alpine_meadow 2 2 150 8 0.5 0.5 0.0 am 200 dh3553"
#    "scalability_exp folk_emp_big alpine_meadow 2 2 150 9 0.5 0.5 0.0 am 200 dh3553"
#    "scalability_exp folk_emp_big alpine_meadow 2 2 150 10 0.5 0.5 0.0 am 200 dh3553"
#    "scalability_exp folk_emp_big alpine_meadow 2 2 150 11 0.5 0.5 0.0 am 200 dh3553"
#    "scalability_exp folk_emp_big alpine_meadow 2 2 150 12 0.5 0.5 0.0 am 200 dh3553"
#    "scalability_exp folk_emp_big alpine_meadow 2 2 150 13 0.5 0.5 0.0 am 200 dh3553"
#    "scalability_exp folk_emp_big alpine_meadow 2 2 150 14 0.5 0.5 0.0 am 200 dh3553"
#    "scalability_exp folk_emp_big alpine_meadow 2 2 150 15 0.5 0.5 0.0 am 200 dh3553"
#    "scalability_exp folk_emp_big alpine_meadow 2 2 150 16 0.5 0.5 0.0 am 200 dh3553"
#    "scalability_exp folk_emp_big alpine_meadow 2 2 150 17 0.5 0.5 0.0 am 200 dh3553"
#    "scalability_exp folk_emp_big alpine_meadow 2 2 150 18 0.5 0.5 0.0 am 200 dh3553"
#    "scalability_exp folk_emp_big alpine_meadow 2 2 150 19 0.5 0.5 0.0 am 200 dh3553"
#    "scalability_exp folk_emp_big alpine_meadow 2 2 150 20 0.5 0.5 0.0 am 200 dh3553"
#    "scalability_exp folk_emp_big alpine_meadow 2 2 150 21 0.5 0.5 0.0 am 200 dh3553"
#    "scalability_exp folk_emp_big alpine_meadow 2 2 150 22 0.5 0.5 0.0 am 200 dh3553"
#    "scalability_exp folk_emp_big alpine_meadow 2 2 150 23 0.5 0.5 0.0 am 200 dh3553"
#    "scalability_exp folk_emp_big alpine_meadow 2 2 150 24 0.5 0.5 0.0 am 200 dh3553"
#    "scalability_exp folk_emp_big alpine_meadow 2 2 150 25 0.5 0.5 0.0 am 200 dh3553"
#    "scalability_exp folk_emp_big alpine_meadow 2 2 150 26 0.5 0.5 0.0 am 200 dh3553"
#    "scalability_exp folk_emp_big alpine_meadow 2 2 150 27 0.5 0.5 0.0 am 200 dh3553"

#    "scalability_exp folk_emp_big alpine_meadow 4 4 150 2 0.5 0.5 0.0 am 200 dh3553"
#    "scalability_exp folk_emp_big alpine_meadow 4 4 150 3 0.5 0.5 0.0 am 200 dh3553"
#    "scalability_exp folk_emp_big alpine_meadow 4 4 150 4 0.5 0.5 0.0 am 200 dh3553"
#    "scalability_exp folk_emp_big alpine_meadow 4 4 150 5 0.5 0.5 0.0 am 200 dh3553"
#    "scalability_exp folk_emp_big alpine_meadow 4 4 150 6 0.5 0.5 0.0 am 200 dh3553"
#    "scalability_exp folk_emp_big alpine_meadow 4 4 150 7 0.5 0.5 0.0 am 200 dh3553"
#    "scalability_exp folk_emp_big alpine_meadow 4 4 150 8 0.5 0.5 0.0 am 200 dh3553"
#    "scalability_exp folk_emp_big alpine_meadow 4 4 150 9 0.5 0.5 0.0 am 200 dh3553"
#    "scalability_exp folk_emp_big alpine_meadow 4 4 150 10 0.5 0.5 0.0 am 200 dh3553"
#    "scalability_exp folk_emp_big alpine_meadow 4 4 150 11 0.5 0.5 0.0 am 200 dh3553"
#    "scalability_exp folk_emp_big alpine_meadow 4 4 150 12 0.5 0.5 0.0 am 200 dh3553"
#    "scalability_exp folk_emp_big alpine_meadow 4 4 150 13 0.5 0.5 0.0 am 200 dh3553"
#    "scalability_exp folk_emp_big alpine_meadow 4 4 150 14 0.5 0.5 0.0 am 200 dh3553"
#    "scalability_exp folk_emp_big alpine_meadow 4 4 150 15 0.5 0.5 0.0 am 200 dh3553"
#    "scalability_exp folk_emp_big alpine_meadow 4 4 150 16 0.5 0.5 0.0 am 200 dh3553"
#    "scalability_exp folk_emp_big alpine_meadow 4 4 150 17 0.5 0.5 0.0 am 200 dh3553"

#    "scalability_exp folk_emp_big alpine_meadow 8 8 150 2 0.5 0.5 0.0 am 200 dh3553"
#    "scalability_exp folk_emp_big alpine_meadow 8 8 150 8 0.5 0.5 0.0 am 200 dh3553"
#    "scalability_exp folk_emp_big alpine_meadow 8 8 150 9 0.5 0.5 0.0 am 200 dh3553"
#    "scalability_exp folk_emp_big alpine_meadow 8 8 150 11 0.5 0.5 0.0 am 200 dh3553"
#    "scalability_exp folk_emp_big alpine_meadow 8 8 150 12 0.5 0.5 0.0 am 200 dh3553"
#    "scalability_exp folk_emp_big alpine_meadow 8 8 150 13 0.5 0.5 0.0 am 200 dh3553"
#    "scalability_exp folk_emp_big alpine_meadow 8 8 150 14 0.5 0.5 0.0 am 200 dh3553"
#    "scalability_exp folk_emp_big alpine_meadow 8 8 150 15 0.5 0.5 0.0 am 200 dh3553"
#    "scalability_exp folk_emp_big alpine_meadow 8 8 150 16 0.5 0.5 0.0 am 200 dh3553"
#    "scalability_exp folk_emp_big alpine_meadow 8 8 150 17 0.5 0.5 0.0 am 200 dh3553"
#    "scalability_exp folk_emp_big alpine_meadow 8 8 150 18 0.5 0.5 0.0 am 200 dh3553"
#    "scalability_exp folk_emp_big alpine_meadow 8 8 150 19 0.5 0.5 0.0 am 200 dh3553"
#    "scalability_exp folk_emp_big alpine_meadow 8 8 150 20 0.5 0.5 0.0 am 200 dh3553"
#    "scalability_exp folk_emp_big alpine_meadow 8 8 150 21 0.5 0.5 0.0 am 200 dh3553"

#    "scalability_exp folk_emp_big alpine_meadow 16 16 120 3 0.5 0.5 0.0 am 200 dh3553"
#    "scalability_exp folk_emp_big alpine_meadow 16 16 120 4 0.5 0.5 0.0 am 200 dh3553"
#    "scalability_exp folk_emp_big alpine_meadow 16 16 120 12 0.5 0.5 0.0 am 200 dh3553"
#    "scalability_exp folk_emp_big alpine_meadow 16 16 120 14 0.5 0.5 0.0 am 200 dh3553"
#    "scalability_exp folk_emp_big alpine_meadow 16 16 120 15 0.5 0.5 0.0 am 200 dh3553"
#    "scalability_exp folk_emp_big alpine_meadow 16 16 120 16 0.5 0.5 0.0 am 200 dh3553"
#    "scalability_exp folk_emp_big alpine_meadow 16 16 120 17 0.5 0.5 0.0 am 200 dh3553"
#    "scalability_exp folk_emp_big alpine_meadow 16 16 120 18 0.5 0.5 0.0 am 200 dh3553"
#    "scalability_exp folk_emp_big alpine_meadow 16 16 150 19 0.5 0.5 0.0 am 200 dh3553"
#    "scalability_exp folk_emp_big alpine_meadow 16 16 150 20 0.5 0.5 0.0 am 200 dh3553"
#    "scalability_exp folk_emp_big alpine_meadow 16 16 180 21 0.5 0.5 0.0 am 200 dh3553"
#    "scalability_exp folk_emp_big alpine_meadow 16 16 180 22 0.5 0.5 0.0 am 200 dh3553"
#    "scalability_exp folk_emp_big alpine_meadow 16 16 180 23 0.5 0.5 0.0 am 200 dh3553"
#    "scalability_exp folk_emp_big alpine_meadow 16 16 180 24 0.5 0.5 0.0 am 200 dh3553"
#    "scalability_exp folk_emp_big alpine_meadow 16 16 180 25 0.5 0.5 0.0 am 200 dh3553"
#    "scalability_exp folk_emp_big alpine_meadow 16 16 180 26 0.5 0.5 0.0 am 200 dh3553"
#    "scalability_exp folk_emp_big alpine_meadow 16 16 180 27 0.5 0.5 0.0 am 200 dh3553"
#    "scalability_exp folk_emp_big alpine_meadow 16 16 180 28 0.5 0.5 0.0 am 200 dh3553"
#    "scalability_exp folk_emp_big alpine_meadow 16 16 180 29 0.5 0.5 0.0 am 200 dh3553"
#    "scalability_exp folk_emp_big alpine_meadow 16 16 180 30 0.5 0.5 0.0 am 200 dh3553"
#    "scalability_exp folk_emp_big alpine_meadow 16 16 180 31 0.5 0.5 0.0 am 200 dh3553"
#    "scalability_exp folk_emp_big alpine_meadow 16 16 180 32 0.5 0.5 0.0 am 200 dh3553"

#    "scalability_exp folk_emp_big alpine_meadow 32 32 128 2 0.5 0.5 0.0 am_workers_32 200 dh3553"
#    "scalability_exp folk_emp_big alpine_meadow 32 32 128 3 0.5 0.5 0.0 am_workers_32 200 dh3553"
#    "scalability_exp folk_emp_big alpine_meadow 32 32 128 4 0.5 0.5 0.0 am_workers_32 200 dh3553"
#    "scalability_exp folk_emp_big alpine_meadow 32 32 128 5 0.5 0.5 0.0 am_workers_32 200 dh3553"
#    "scalability_exp folk_emp_big alpine_meadow 32 32 128 6 0.5 0.5 0.0 am_workers_32 200 dh3553"
#    "scalability_exp folk_emp_big alpine_meadow 32 32 150 7 0.5 0.5 0.0 am_workers_32 200 dh3553"
#    "scalability_exp folk_emp_big alpine_meadow 32 32 150 8 0.5 0.5 0.0 am_workers_32 200 dh3553"
#    "scalability_exp folk_emp_big alpine_meadow 32 32 150 9 0.5 0.5 0.0 am_workers_32 200 dh3553"
#    "scalability_exp folk_emp_big alpine_meadow 32 32 150 10 0.5 0.5 0.0 am_workers_32 200 dh3553"
#    "scalability_exp folk_emp_big alpine_meadow 32 32 150 11 0.5 0.5 0.0 am_workers_32 200 dh3553"
)

# Initialize a counter
index=0

# Iterate through the array of job_configs
for job_config in "${job_configs[@]}"
do
    # Split the job_config into separate variables
    read -r exp_name dataset system_name num_workers cpus mem run_num w1 w2 w3 exp_config_suffix max_total_pipelines_num email <<< "$job_config"
    template_file="/home/${email}/projects/virny-flow-experiments/cluster/${exp_name}/${system_name}-${dataset}-template.sbatch"

    # Define the output file name
    output_file="/home/${email}/projects/virny-flow-experiments/cluster/${exp_name}/sbatch_files/${exp_name}_${dataset}_${num_workers}_${exp_config_suffix}_run_${run_num}_${index}_$(date +"%Y%m%d%H%M%S").sbatch"

    # Create an empty file
    touch $output_file

    # Use sed to replace placeholders with actual values
    sed -e "s/<EXP_NAME>/${exp_name}/g" -e "s/<DATASET>/${dataset}/g" -e "s/<NUM_WORKERS>/${num_workers}/g" -e "s/<CPUS>/${cpus}/g" -e "s/<MEM>/${mem}/g" -e "s/<RUN_NUM>/${run_num}/g" -e "s/<W1>/${w1}/g" -e "s/<W2>/${w2}/g" -e "s/<W3>/${w3}/g" -e "s/<MAX_TOTAL_PIPELINES_NUM>/${max_total_pipelines_num}/g" -e "s/<EXP_CONFIG_SUFFIX>/${exp_config_suffix}/g" -e "s/<EMAIL>/${email}/g" $template_file > $output_file

    # Execute a SLURM job
    sbatch $output_file

    echo "Job was executed: $output_file"

    # Increment the index
    ((index++))
done
