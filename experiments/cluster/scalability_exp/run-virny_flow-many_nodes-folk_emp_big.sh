# Define the list of tuples (exp_name, dataset, system_name, num_nodes, num_workers, cpus_per_node, mem, run_num, w1, w2, w3, exp_config_suffix, max_total_pipelines_num, email)
declare -a job_configs=(
#    "scalability_exp folk_emp_big virny_flow 2 32 32 150 1 0.5 0.5 0.0 vf 200 dh3553"
#    "scalability_exp folk_emp_big virny_flow 2 32 32 150 2 0.5 0.5 0.0 vf 200 dh3553"
#    "scalability_exp folk_emp_big virny_flow 2 32 32 150 3 0.5 0.5 0.0 vf 200 dh3553"
#    "scalability_exp folk_emp_big virny_flow 2 32 32 150 4 0.5 0.5 0.0 vf 200 dh3553"
#    "scalability_exp folk_emp_big virny_flow 2 32 32 150 5 0.5 0.5 0.0 vf 200 dh3553"
#    "scalability_exp folk_emp_big virny_flow 2 32 32 150 6 0.5 0.5 0.0 vf 200 dh3553"
#    "scalability_exp folk_emp_big virny_flow 2 32 32 150 7 0.5 0.5 0.0 vf 200 dh3553"
#    "scalability_exp folk_emp_big virny_flow 2 32 32 150 8 0.5 0.5 0.0 vf 200 dh3553"
#    "scalability_exp folk_emp_big virny_flow 2 32 32 150 9 0.5 0.5 0.0 vf 200 dh3553"
#    "scalability_exp folk_emp_big virny_flow 2 32 32 150 10 0.5 0.5 0.0 vf 200 dh3553"
    "scalability_exp folk_emp_big virny_flow 2 32 32 150 11 0.5 0.5 0.0 vf 200 dh3553"
    "scalability_exp folk_emp_big virny_flow 2 32 32 150 12 0.5 0.5 0.0 vf 200 dh3553"

#    "scalability_exp folk_emp_big virny_flow 4 32 32 150 1 0.5 0.5 0.0 vf 200 dh3553"
#    "scalability_exp folk_emp_big virny_flow 4 32 32 150 2 0.5 0.5 0.0 vf 200 dh3553"
#    "scalability_exp folk_emp_big virny_flow 4 32 32 150 3 0.5 0.5 0.0 vf 200 dh3553"
#    "scalability_exp folk_emp_big virny_flow 4 32 32 150 4 0.5 0.5 0.0 vf 200 dh3553"
#    "scalability_exp folk_emp_big virny_flow 4 32 32 150 5 0.5 0.5 0.0 vf 200 dh3553"
#    "scalability_exp folk_emp_big virny_flow 4 32 32 150 6 0.5 0.5 0.0 vf 200 dh3553"
#    "scalability_exp folk_emp_big virny_flow 4 32 32 150 7 0.5 0.5 0.0 vf 200 dh3553"
#    "scalability_exp folk_emp_big virny_flow 4 32 32 150 8 0.5 0.5 0.0 vf 200 dh3553"
#    "scalability_exp folk_emp_big virny_flow 4 32 32 150 9 0.5 0.5 0.0 vf 200 dh3553"
#    "scalability_exp folk_emp_big virny_flow 4 32 32 150 10 0.5 0.5 0.0 vf 200 dh3553"
    "scalability_exp folk_emp_big virny_flow 4 32 32 150 11 0.5 0.5 0.0 vf 200 dh3553"
    "scalability_exp folk_emp_big virny_flow 4 32 32 150 12 0.5 0.5 0.0 vf 200 dh3553"
)

# Initialize a counter
index=0

# Iterate through the array of job_configs
for job_config in "${job_configs[@]}"
do
    # Split the job_config into separate variables
    read -r exp_name dataset system_name num_nodes num_workers cpus mem run_num w1 w2 w3 exp_config_suffix max_total_pipelines_num email <<< "$job_config"
    template_file="/home/${email}/projects/virny-flow-experiments/cluster/${exp_name}/${system_name}-many_nodes-${dataset}-template.sbatch"

    # Define the output file name
    output_file="/home/${email}/projects/virny-flow-experiments/cluster/${exp_name}/sbatch_files/${exp_name}_${dataset}_${num_nodes}_${num_workers}_${exp_config_suffix}_run_${run_num}_${index}_$(date +"%Y%m%d%H%M%S").sbatch"

    # Create an empty file
    touch $output_file

    # Use sed to replace placeholders with actual values
    sed -e "s/<EXP_NAME>/${exp_name}/g" -e "s/<DATASET>/${dataset}/g" -e "s/<NUM_NODES>/${num_nodes}/g" -e "s/<NUM_WORKERS>/${num_workers}/g" -e "s/<CPUS>/${cpus}/g" -e "s/<MEM>/${mem}/g" -e "s/<RUN_NUM>/${run_num}/g" -e "s/<W1>/${w1}/g" -e "s/<W2>/${w2}/g" -e "s/<W3>/${w3}/g" -e "s/<MAX_TOTAL_PIPELINES_NUM>/${max_total_pipelines_num}/g" -e "s/<EXP_CONFIG_SUFFIX>/${exp_config_suffix}/g" -e "s/<EMAIL>/${email}/g" $template_file > $output_file

    # Execute a SLURM job
    sbatch $output_file

    echo "Job was executed: $output_file"

    # Increment the index
    ((index++))
done
