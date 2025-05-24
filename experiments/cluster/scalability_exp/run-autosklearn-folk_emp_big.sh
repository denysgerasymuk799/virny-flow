# Define the list of tuples (exp_name, dataset, system_name, num_workers, cpus, mem, run_num, w1, w2, w3, exp_config_suffix, max_total_pipelines_num, email)
declare -a job_configs=(
    "scalability_exp folk_emp_big autosklearn 1 1 20 1 0.5 0.5 0.0 askl 200 dh3553"
    "scalability_exp folk_emp_big autosklearn 1 1 20 2 0.5 0.5 0.0 askl 200 dh3553"
    "scalability_exp folk_emp_big autosklearn 1 1 20 3 0.5 0.5 0.0 askl 200 dh3553"
    "scalability_exp folk_emp_big autosklearn 1 1 20 4 0.5 0.5 0.0 askl 200 dh3553"
    "scalability_exp folk_emp_big autosklearn 1 1 20 5 0.5 0.5 0.0 askl 200 dh3553"
    "scalability_exp folk_emp_big autosklearn 1 1 20 6 0.5 0.5 0.0 askl 200 dh3553"
    "scalability_exp folk_emp_big autosklearn 1 1 20 7 0.5 0.5 0.0 askl 200 dh3553"
    "scalability_exp folk_emp_big autosklearn 1 1 20 8 0.5 0.5 0.0 askl 200 dh3553"
    "scalability_exp folk_emp_big autosklearn 1 1 20 9 0.5 0.5 0.0 askl 200 dh3553"
    "scalability_exp folk_emp_big autosklearn 1 1 20 10 0.5 0.5 0.0 askl 200 dh3553"
    "scalability_exp folk_emp_big autosklearn 1 1 20 11 0.5 0.5 0.0 askl 200 dh3553"
    "scalability_exp folk_emp_big autosklearn 1 1 20 12 0.5 0.5 0.0 askl 200 dh3553"
    "scalability_exp folk_emp_big autosklearn 1 1 20 13 0.5 0.5 0.0 askl 200 dh3553"
    "scalability_exp folk_emp_big autosklearn 1 1 20 14 0.5 0.5 0.0 askl 200 dh3553"
    "scalability_exp folk_emp_big autosklearn 1 1 20 15 0.5 0.5 0.0 askl 200 dh3553"
    "scalability_exp folk_emp_big autosklearn 1 1 20 16 0.5 0.5 0.0 askl 200 dh3553"
    "scalability_exp folk_emp_big autosklearn 1 1 20 17 0.5 0.5 0.0 askl 200 dh3553"
    "scalability_exp folk_emp_big autosklearn 1 1 20 18 0.5 0.5 0.0 askl 200 dh3553"
    "scalability_exp folk_emp_big autosklearn 1 1 20 19 0.5 0.5 0.0 askl 200 dh3553"
    "scalability_exp folk_emp_big autosklearn 1 1 20 20 0.5 0.5 0.0 askl 200 dh3553"

#    "scalability_exp folk_emp_big autosklearn 2 2 20 1 0.5 0.5 0.0 askl 200 dh3553"
#    "scalability_exp folk_emp_big autosklearn 2 2 20 2 0.5 0.5 0.0 askl 200 dh3553"
#    "scalability_exp folk_emp_big autosklearn 2 2 20 3 0.5 0.5 0.0 askl 200 dh3553"
#    "scalability_exp folk_emp_big autosklearn 2 2 20 4 0.5 0.5 0.0 askl 200 dh3553"
#    "scalability_exp folk_emp_big autosklearn 2 2 20 5 0.5 0.5 0.0 askl 200 dh3553"
#    "scalability_exp folk_emp_big autosklearn 2 2 20 6 0.5 0.5 0.0 askl 200 dh3553"
#    "scalability_exp folk_emp_big autosklearn 2 2 20 7 0.5 0.5 0.0 askl 200 dh3553"
#    "scalability_exp folk_emp_big autosklearn 2 2 20 8 0.5 0.5 0.0 askl 200 dh3553"
#    "scalability_exp folk_emp_big autosklearn 2 2 20 9 0.5 0.5 0.0 askl 200 dh3553"
#    "scalability_exp folk_emp_big autosklearn 2 2 20 10 0.5 0.5 0.0 askl 200 dh3553"
#
#    "scalability_exp folk_emp_big autosklearn 4 4 40 1 0.5 0.5 0.0 askl 200 dh3553"
#    "scalability_exp folk_emp_big autosklearn 4 4 40 2 0.5 0.5 0.0 askl 200 dh3553"
#    "scalability_exp folk_emp_big autosklearn 4 4 40 3 0.5 0.5 0.0 askl 200 dh3553"
#    "scalability_exp folk_emp_big autosklearn 4 4 40 4 0.5 0.5 0.0 askl 200 dh3553"
#    "scalability_exp folk_emp_big autosklearn 4 4 40 5 0.5 0.5 0.0 askl 200 dh3553"
#    "scalability_exp folk_emp_big autosklearn 4 4 40 6 0.5 0.5 0.0 askl 200 dh3553"
#    "scalability_exp folk_emp_big autosklearn 4 4 40 7 0.5 0.5 0.0 askl 200 dh3553"
#    "scalability_exp folk_emp_big autosklearn 4 4 40 8 0.5 0.5 0.0 askl 200 dh3553"
#    "scalability_exp folk_emp_big autosklearn 4 4 40 9 0.5 0.5 0.0 askl 200 dh3553"
#    "scalability_exp folk_emp_big autosklearn 4 4 40 10 0.5 0.5 0.0 askl 200 dh3553"
#
#    "scalability_exp folk_emp_big autosklearn 8 8 60 1 0.5 0.5 0.0 askl 200 dh3553"
#    "scalability_exp folk_emp_big autosklearn 8 8 60 2 0.5 0.5 0.0 askl 200 dh3553"
#    "scalability_exp folk_emp_big autosklearn 8 8 60 3 0.5 0.5 0.0 askl 200 dh3553"
#    "scalability_exp folk_emp_big autosklearn 8 8 60 4 0.5 0.5 0.0 askl 200 dh3553"
#    "scalability_exp folk_emp_big autosklearn 8 8 60 5 0.5 0.5 0.0 askl 200 dh3553"
#    "scalability_exp folk_emp_big autosklearn 8 8 60 6 0.5 0.5 0.0 askl 200 dh3553"
#    "scalability_exp folk_emp_big autosklearn 8 8 60 7 0.5 0.5 0.0 askl 200 dh3553"
#    "scalability_exp folk_emp_big autosklearn 8 8 60 8 0.5 0.5 0.0 askl 200 dh3553"
#    "scalability_exp folk_emp_big autosklearn 8 8 60 9 0.5 0.5 0.0 askl 200 dh3553"
#    "scalability_exp folk_emp_big autosklearn 8 8 60 10 0.5 0.5 0.0 askl 200 dh3553"
#
#    "scalability_exp folk_emp_big autosklearn 16 16 80 1 0.5 0.5 0.0 askl 200 dh3553"
#    "scalability_exp folk_emp_big autosklearn 16 16 80 2 0.5 0.5 0.0 askl 200 dh3553"
#    "scalability_exp folk_emp_big autosklearn 16 16 80 3 0.5 0.5 0.0 askl 200 dh3553"
#    "scalability_exp folk_emp_big autosklearn 16 16 80 4 0.5 0.5 0.0 askl 200 dh3553"
#    "scalability_exp folk_emp_big autosklearn 16 16 80 5 0.5 0.5 0.0 askl 200 dh3553"
#    "scalability_exp folk_emp_big autosklearn 16 16 80 6 0.5 0.5 0.0 askl 200 dh3553"
#    "scalability_exp folk_emp_big autosklearn 16 16 80 7 0.5 0.5 0.0 askl 200 dh3553"
#    "scalability_exp folk_emp_big autosklearn 16 16 80 8 0.5 0.5 0.0 askl 200 dh3553"
#    "scalability_exp folk_emp_big autosklearn 16 16 80 9 0.5 0.5 0.0 askl 200 dh3553"
#    "scalability_exp folk_emp_big autosklearn 16 16 80 10 0.5 0.5 0.0 askl 200 dh3553"

#    "scalability_exp folk_emp_big autosklearn 32 32 150 1 0.5 0.5 0.0 askl_workers_32 200 dh3553"
#    "scalability_exp folk_emp_big autosklearn 32 32 150 2 0.5 0.5 0.0 askl_workers_32 200 dh3553"
#    "scalability_exp folk_emp_big autosklearn 32 32 150 3 0.5 0.5 0.0 askl_workers_32 200 dh3553"
#    "scalability_exp folk_emp_big autosklearn 32 32 150 4 0.5 0.5 0.0 askl_workers_32 200 dh3553"
#    "scalability_exp folk_emp_big autosklearn 32 32 150 5 0.5 0.5 0.0 askl_workers_32 200 dh3553"
#    "scalability_exp folk_emp_big autosklearn 32 32 150 6 0.5 0.5 0.0 askl_workers_32 200 dh3553"
#    "scalability_exp folk_emp_big autosklearn 32 32 150 7 0.5 0.5 0.0 askl_workers_32 200 dh3553"
#    "scalability_exp folk_emp_big autosklearn 32 32 150 8 0.5 0.5 0.0 askl_workers_32 200 dh3553"
#    "scalability_exp folk_emp_big autosklearn 32 32 150 9 0.5 0.5 0.0 askl_workers_32 200 dh3553"
#    "scalability_exp folk_emp_big autosklearn 32 32 150 10 0.5 0.5 0.0 askl_workers_32 200 dh3553"
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
