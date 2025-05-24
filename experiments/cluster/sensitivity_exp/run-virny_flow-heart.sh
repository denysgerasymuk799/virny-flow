# Define the list of tuples (exp_name, dataset, system_name, num_workers, cpus, mem, num_pp_candidates, run_num, w1, w2, w3, exp_config_suffix, max_total_pipelines_num, email)
declare -a job_configs=(
#    "sensitivity_exp heart virny_flow 32 32 120 1 1 0.5 0.5 0.0 vf_pp_cand_1 200 dh3553"
    "sensitivity_exp heart virny_flow 32 32 120 1 2 0.5 0.5 0.0 vf_pp_cand_1 200 dh3553"
    "sensitivity_exp heart virny_flow 32 32 120 1 3 0.5 0.5 0.0 vf_pp_cand_1 200 dh3553"
    "sensitivity_exp heart virny_flow 32 32 120 1 4 0.5 0.5 0.0 vf_pp_cand_1 200 dh3553"
    "sensitivity_exp heart virny_flow 32 32 120 1 5 0.5 0.5 0.0 vf_pp_cand_1 200 dh3553"
    "sensitivity_exp heart virny_flow 32 32 120 1 6 0.5 0.5 0.0 vf_pp_cand_1 200 dh3553"
    "sensitivity_exp heart virny_flow 32 32 120 1 7 0.5 0.5 0.0 vf_pp_cand_1 200 dh3553"
    "sensitivity_exp heart virny_flow 32 32 120 1 8 0.5 0.5 0.0 vf_pp_cand_1 200 dh3553"
    "sensitivity_exp heart virny_flow 32 32 120 1 9 0.5 0.5 0.0 vf_pp_cand_1 200 dh3553"
    "sensitivity_exp heart virny_flow 32 32 120 1 10 0.5 0.5 0.0 vf_pp_cand_1 200 dh3553"

#    "sensitivity_exp heart virny_flow 32 32 120 2 1 0.5 0.5 0.0 vf_pp_cand_2 200 dh3553"
    "sensitivity_exp heart virny_flow 32 32 120 2 2 0.5 0.5 0.0 vf_pp_cand_2 200 dh3553"
    "sensitivity_exp heart virny_flow 32 32 120 2 3 0.5 0.5 0.0 vf_pp_cand_2 200 dh3553"
    "sensitivity_exp heart virny_flow 32 32 120 2 4 0.5 0.5 0.0 vf_pp_cand_2 200 dh3553"
    "sensitivity_exp heart virny_flow 32 32 120 2 5 0.5 0.5 0.0 vf_pp_cand_2 200 dh3553"
    "sensitivity_exp heart virny_flow 32 32 120 2 6 0.5 0.5 0.0 vf_pp_cand_2 200 dh3553"
    "sensitivity_exp heart virny_flow 32 32 120 2 7 0.5 0.5 0.0 vf_pp_cand_2 200 dh3553"
    "sensitivity_exp heart virny_flow 32 32 120 2 8 0.5 0.5 0.0 vf_pp_cand_2 200 dh3553"
    "sensitivity_exp heart virny_flow 32 32 120 2 9 0.5 0.5 0.0 vf_pp_cand_2 200 dh3553"
    "sensitivity_exp heart virny_flow 32 32 120 2 10 0.5 0.5 0.0 vf_pp_cand_2 200 dh3553"

#    "sensitivity_exp heart virny_flow 32 32 120 4 1 0.5 0.5 0.0 vf_pp_cand_4 200 dh3553"
    "sensitivity_exp heart virny_flow 32 32 120 4 2 0.5 0.5 0.0 vf_pp_cand_4 200 dh3553"
    "sensitivity_exp heart virny_flow 32 32 120 4 3 0.5 0.5 0.0 vf_pp_cand_4 200 dh3553"
    "sensitivity_exp heart virny_flow 32 32 120 4 4 0.5 0.5 0.0 vf_pp_cand_4 200 dh3553"
    "sensitivity_exp heart virny_flow 32 32 120 4 5 0.5 0.5 0.0 vf_pp_cand_4 200 dh3553"
    "sensitivity_exp heart virny_flow 32 32 120 4 6 0.5 0.5 0.0 vf_pp_cand_4 200 dh3553"
    "sensitivity_exp heart virny_flow 32 32 120 4 7 0.5 0.5 0.0 vf_pp_cand_4 200 dh3553"
    "sensitivity_exp heart virny_flow 32 32 120 4 8 0.5 0.5 0.0 vf_pp_cand_4 200 dh3553"
    "sensitivity_exp heart virny_flow 32 32 120 4 9 0.5 0.5 0.0 vf_pp_cand_4 200 dh3553"
    "sensitivity_exp heart virny_flow 32 32 120 4 10 0.5 0.5 0.0 vf_pp_cand_4 200 dh3553"

#    "sensitivity_exp heart virny_flow 32 32 120 8 1 0.5 0.5 0.0 vf_pp_cand_8 200 dh3553"
    "sensitivity_exp heart virny_flow 32 32 120 8 2 0.5 0.5 0.0 vf_pp_cand_8 200 dh3553"
    "sensitivity_exp heart virny_flow 32 32 120 8 3 0.5 0.5 0.0 vf_pp_cand_8 200 dh3553"
    "sensitivity_exp heart virny_flow 32 32 120 8 4 0.5 0.5 0.0 vf_pp_cand_8 200 dh3553"
    "sensitivity_exp heart virny_flow 32 32 120 8 5 0.5 0.5 0.0 vf_pp_cand_8 200 dh3553"
    "sensitivity_exp heart virny_flow 32 32 120 8 6 0.5 0.5 0.0 vf_pp_cand_8 200 dh3553"
    "sensitivity_exp heart virny_flow 32 32 120 8 7 0.5 0.5 0.0 vf_pp_cand_8 200 dh3553"
    "sensitivity_exp heart virny_flow 32 32 120 8 8 0.5 0.5 0.0 vf_pp_cand_8 200 dh3553"
    "sensitivity_exp heart virny_flow 32 32 120 8 9 0.5 0.5 0.0 vf_pp_cand_8 200 dh3553"
    "sensitivity_exp heart virny_flow 32 32 120 8 10 0.5 0.5 0.0 vf_pp_cand_8 200 dh3553"

#    "sensitivity_exp heart virny_flow 32 32 120 16 1 0.5 0.5 0.0 vf_pp_cand_16 200 dh3553"
    "sensitivity_exp heart virny_flow 32 32 120 16 2 0.5 0.5 0.0 vf_pp_cand_16 200 dh3553"
    "sensitivity_exp heart virny_flow 32 32 120 16 3 0.5 0.5 0.0 vf_pp_cand_16 200 dh3553"
    "sensitivity_exp heart virny_flow 32 32 120 16 4 0.5 0.5 0.0 vf_pp_cand_16 200 dh3553"
    "sensitivity_exp heart virny_flow 32 32 120 16 5 0.5 0.5 0.0 vf_pp_cand_16 200 dh3553"
    "sensitivity_exp heart virny_flow 32 32 120 16 6 0.5 0.5 0.0 vf_pp_cand_16 200 dh3553"
    "sensitivity_exp heart virny_flow 32 32 120 16 7 0.5 0.5 0.0 vf_pp_cand_16 200 dh3553"
    "sensitivity_exp heart virny_flow 32 32 120 16 8 0.5 0.5 0.0 vf_pp_cand_16 200 dh3553"
    "sensitivity_exp heart virny_flow 32 32 120 16 9 0.5 0.5 0.0 vf_pp_cand_16 200 dh3553"
    "sensitivity_exp heart virny_flow 32 32 120 16 10 0.5 0.5 0.0 vf_pp_cand_16 200 dh3553"

#    "sensitivity_exp heart virny_flow 32 32 120 32 1 0.5 0.5 0.0 vf_pp_cand_32 200 dh3553"
    "sensitivity_exp heart virny_flow 32 32 120 32 2 0.5 0.5 0.0 vf_pp_cand_32 200 dh3553"
    "sensitivity_exp heart virny_flow 32 32 120 32 3 0.5 0.5 0.0 vf_pp_cand_32 200 dh3553"
    "sensitivity_exp heart virny_flow 32 32 120 32 4 0.5 0.5 0.0 vf_pp_cand_32 200 dh3553"
    "sensitivity_exp heart virny_flow 32 32 120 32 5 0.5 0.5 0.0 vf_pp_cand_32 200 dh3553"
    "sensitivity_exp heart virny_flow 32 32 120 32 6 0.5 0.5 0.0 vf_pp_cand_32 200 dh3553"
    "sensitivity_exp heart virny_flow 32 32 120 32 7 0.5 0.5 0.0 vf_pp_cand_32 200 dh3553"
    "sensitivity_exp heart virny_flow 32 32 120 32 8 0.5 0.5 0.0 vf_pp_cand_32 200 dh3553"
    "sensitivity_exp heart virny_flow 32 32 120 32 9 0.5 0.5 0.0 vf_pp_cand_32 200 dh3553"
    "sensitivity_exp heart virny_flow 32 32 120 32 10 0.5 0.5 0.0 vf_pp_cand_32 200 dh3553"
)

# Initialize a counter
index=0

# Iterate through the array of job_configs
for job_config in "${job_configs[@]}"
do
    # Split the job_config into separate variables
    read -r exp_name dataset system_name num_workers cpus mem num_pp_candidates run_num w1 w2 w3 exp_config_suffix max_total_pipelines_num email <<< "$job_config"
    template_file="/home/${email}/projects/virny-flow-experiments/cluster/${exp_name}/${system_name}-${dataset}-template.sbatch"

    # Define the output file name
    output_file="/home/${email}/projects/virny-flow-experiments/cluster/${exp_name}/sbatch_files/${exp_name}_${dataset}_${num_workers}_${exp_config_suffix}_run_${run_num}_${index}_$(date +"%Y%m%d%H%M%S").sbatch"

    # Create an empty file
    touch $output_file

    # Use sed to replace placeholders with actual values
    sed -e "s/<EXP_NAME>/${exp_name}/g" -e "s/<DATASET>/${dataset}/g" -e "s/<NUM_WORKERS>/${num_workers}/g" -e "s/<CPUS>/${cpus}/g" -e "s/<MEM>/${mem}/g" -e "s/<NUM_PP_CANDIDATES>/${num_pp_candidates}/g" -e "s/<RUN_NUM>/${run_num}/g" -e "s/<W1>/${w1}/g" -e "s/<W2>/${w2}/g" -e "s/<W3>/${w3}/g" -e "s/<MAX_TOTAL_PIPELINES_NUM>/${max_total_pipelines_num}/g" -e "s/<EXP_CONFIG_SUFFIX>/${exp_config_suffix}/g" -e "s/<EMAIL>/${email}/g" $template_file > $output_file

    # Execute a SLURM job
    sbatch $output_file

    echo "Job was executed: $output_file"

    # Increment the index
    ((index++))
done
