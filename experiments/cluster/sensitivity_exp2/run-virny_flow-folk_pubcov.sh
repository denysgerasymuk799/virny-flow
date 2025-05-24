# Define the list of tuples (exp_name, dataset, system_name, num_workers, cpus, mem, training_set_fractions_for_halting, run_num, w1, w2, w3, exp_config_suffix, max_total_pipelines_num, email)
declare -a job_configs=(
#    "sensitivity_exp2 folk_pubcov virny_flow 32 32 120 [1.0] 1 0.5 0.5 0.0 vf_halting_1 200 dh3553"
#    "sensitivity_exp2 folk_pubcov virny_flow 32 32 120 [1.0] 2 0.5 0.5 0.0 vf_halting_1 200 dh3553"
#    "sensitivity_exp2 folk_pubcov virny_flow 32 32 120 [1.0] 3 0.5 0.5 0.0 vf_halting_1 200 dh3553"
#    "sensitivity_exp2 folk_pubcov virny_flow 32 32 120 [1.0] 4 0.5 0.5 0.0 vf_halting_1 200 dh3553"
#    "sensitivity_exp2 folk_pubcov virny_flow 32 32 120 [1.0] 5 0.5 0.5 0.0 vf_halting_1 200 dh3553"
#    "sensitivity_exp2 folk_pubcov virny_flow 32 32 120 [1.0] 6 0.5 0.5 0.0 vf_halting_1 200 dh3553"
#    "sensitivity_exp2 folk_pubcov virny_flow 32 32 120 [1.0] 7 0.5 0.5 0.0 vf_halting_1 200 dh3553"
#    "sensitivity_exp2 folk_pubcov virny_flow 32 32 120 [1.0] 8 0.5 0.5 0.0 vf_halting_1 200 dh3553"
#    "sensitivity_exp2 folk_pubcov virny_flow 32 32 120 [1.0] 9 0.5 0.5 0.0 vf_halting_1 200 dh3553"
#    "sensitivity_exp2 folk_pubcov virny_flow 32 32 120 [1.0] 10 0.5 0.5 0.0 vf_halting_1 200 dh3553"
#    "sensitivity_exp2 folk_pubcov virny_flow 32 32 120 [1.0] 11 0.5 0.5 0.0 vf_halting_1 200 dh3553"
#    "sensitivity_exp2 folk_pubcov virny_flow 32 32 120 [1.0] 12 0.5 0.5 0.0 vf_halting_1 200 dh3553"
    "sensitivity_exp2 folk_pubcov virny_flow 32 32 120 [1.0] 13 0.5 0.5 0.0 vf_halting_1 200 dh3553"
    "sensitivity_exp2 folk_pubcov virny_flow 32 32 120 [1.0] 14 0.5 0.5 0.0 vf_halting_1 200 dh3553"
    "sensitivity_exp2 folk_pubcov virny_flow 32 32 120 [1.0] 15 0.5 0.5 0.0 vf_halting_1 200 dh3553"

#    "sensitivity_exp2 folk_pubcov virny_flow 32 32 120 [0.25,1.0] 1 0.5 0.5 0.0 vf_halting_2 200 dh3553"
#    "sensitivity_exp2 folk_pubcov virny_flow 32 32 120 [0.25,1.0] 2 0.5 0.5 0.0 vf_halting_2 200 dh3553"
#    "sensitivity_exp2 folk_pubcov virny_flow 32 32 120 [0.25,1.0] 3 0.5 0.5 0.0 vf_halting_2 200 dh3553"
#    "sensitivity_exp2 folk_pubcov virny_flow 32 32 120 [0.25,1.0] 4 0.5 0.5 0.0 vf_halting_2 200 dh3553"
#    "sensitivity_exp2 folk_pubcov virny_flow 32 32 120 [0.25,1.0] 5 0.5 0.5 0.0 vf_halting_2 200 dh3553"
#    "sensitivity_exp2 folk_pubcov virny_flow 32 32 120 [0.25,1.0] 6 0.5 0.5 0.0 vf_halting_2 200 dh3553"
#    "sensitivity_exp2 folk_pubcov virny_flow 32 32 120 [0.25,1.0] 7 0.5 0.5 0.0 vf_halting_2 200 dh3553"
#    "sensitivity_exp2 folk_pubcov virny_flow 32 32 120 [0.25,1.0] 8 0.5 0.5 0.0 vf_halting_2 200 dh3553"
#    "sensitivity_exp2 folk_pubcov virny_flow 32 32 120 [0.25,1.0] 9 0.5 0.5 0.0 vf_halting_2 200 dh3553"
#    "sensitivity_exp2 folk_pubcov virny_flow 32 32 120 [0.25,1.0] 10 0.5 0.5 0.0 vf_halting_2 200 dh3553"
#    "sensitivity_exp2 folk_pubcov virny_flow 32 32 120 [0.25,1.0] 11 0.5 0.5 0.0 vf_halting_2 200 dh3553"
#    "sensitivity_exp2 folk_pubcov virny_flow 32 32 120 [0.25,1.0] 12 0.5 0.5 0.0 vf_halting_2 200 dh3553"

#    "sensitivity_exp2 folk_pubcov virny_flow 32 32 120 [0.5,1.0] 1 0.5 0.5 0.0 vf_halting_3 200 dh3553"
#    "sensitivity_exp2 folk_pubcov virny_flow 32 32 120 [0.5,1.0] 2 0.5 0.5 0.0 vf_halting_3 200 dh3553"
#    "sensitivity_exp2 folk_pubcov virny_flow 32 32 120 [0.5,1.0] 3 0.5 0.5 0.0 vf_halting_3 200 dh3553"
#    "sensitivity_exp2 folk_pubcov virny_flow 32 32 120 [0.5,1.0] 4 0.5 0.5 0.0 vf_halting_3 200 dh3553"
#    "sensitivity_exp2 folk_pubcov virny_flow 32 32 120 [0.5,1.0] 5 0.5 0.5 0.0 vf_halting_3 200 dh3553"
#    "sensitivity_exp2 folk_pubcov virny_flow 32 32 120 [0.5,1.0] 6 0.5 0.5 0.0 vf_halting_3 200 dh3553"
#    "sensitivity_exp2 folk_pubcov virny_flow 32 32 120 [0.5,1.0] 7 0.5 0.5 0.0 vf_halting_3 200 dh3553"
#    "sensitivity_exp2 folk_pubcov virny_flow 32 32 120 [0.5,1.0] 8 0.5 0.5 0.0 vf_halting_3 200 dh3553"
#    "sensitivity_exp2 folk_pubcov virny_flow 32 32 120 [0.5,1.0] 9 0.5 0.5 0.0 vf_halting_3 200 dh3553"
#    "sensitivity_exp2 folk_pubcov virny_flow 32 32 120 [0.5,1.0] 10 0.5 0.5 0.0 vf_halting_3 200 dh3553"
#    "sensitivity_exp2 folk_pubcov virny_flow 32 32 120 [0.5,1.0] 11 0.5 0.5 0.0 vf_halting_3 200 dh3553"
#    "sensitivity_exp2 folk_pubcov virny_flow 32 32 120 [0.5,1.0] 12 0.5 0.5 0.0 vf_halting_3 200 dh3553"

#    "sensitivity_exp2 folk_pubcov virny_flow 32 32 120 [0.75,1.0] 1 0.5 0.5 0.0 vf_halting_4 200 dh3553"
#    "sensitivity_exp2 folk_pubcov virny_flow 32 32 120 [0.75,1.0] 2 0.5 0.5 0.0 vf_halting_4 200 dh3553"
#    "sensitivity_exp2 folk_pubcov virny_flow 32 32 120 [0.75,1.0] 3 0.5 0.5 0.0 vf_halting_4 200 dh3553"
#    "sensitivity_exp2 folk_pubcov virny_flow 32 32 120 [0.75,1.0] 4 0.5 0.5 0.0 vf_halting_4 200 dh3553"
#    "sensitivity_exp2 folk_pubcov virny_flow 32 32 120 [0.75,1.0] 5 0.5 0.5 0.0 vf_halting_4 200 dh3553"
#    "sensitivity_exp2 folk_pubcov virny_flow 32 32 120 [0.75,1.0] 6 0.5 0.5 0.0 vf_halting_4 200 dh3553"
#    "sensitivity_exp2 folk_pubcov virny_flow 32 32 120 [0.75,1.0] 7 0.5 0.5 0.0 vf_halting_4 200 dh3553"
#    "sensitivity_exp2 folk_pubcov virny_flow 32 32 120 [0.75,1.0] 8 0.5 0.5 0.0 vf_halting_4 200 dh3553"
#    "sensitivity_exp2 folk_pubcov virny_flow 32 32 120 [0.75,1.0] 9 0.5 0.5 0.0 vf_halting_4 200 dh3553"
#    "sensitivity_exp2 folk_pubcov virny_flow 32 32 120 [0.75,1.0] 10 0.5 0.5 0.0 vf_halting_4 200 dh3553"
#    "sensitivity_exp2 folk_pubcov virny_flow 32 32 120 [0.75,1.0] 11 0.5 0.5 0.0 vf_halting_4 200 dh3553"
#    "sensitivity_exp2 folk_pubcov virny_flow 32 32 120 [0.75,1.0] 12 0.5 0.5 0.0 vf_halting_4 200 dh3553"

#    "sensitivity_exp2 folk_pubcov virny_flow 32 32 120 [0.25,0.5,1.0] 1 0.5 0.5 0.0 vf_halting_5 200 dh3553"
#    "sensitivity_exp2 folk_pubcov virny_flow 32 32 120 [0.25,0.5,1.0] 2 0.5 0.5 0.0 vf_halting_5 200 dh3553"
#    "sensitivity_exp2 folk_pubcov virny_flow 32 32 120 [0.25,0.5,1.0] 3 0.5 0.5 0.0 vf_halting_5 200 dh3553"
#    "sensitivity_exp2 folk_pubcov virny_flow 32 32 120 [0.25,0.5,1.0] 4 0.5 0.5 0.0 vf_halting_5 200 dh3553"
#    "sensitivity_exp2 folk_pubcov virny_flow 32 32 120 [0.25,0.5,1.0] 5 0.5 0.5 0.0 vf_halting_5 200 dh3553"
#    "sensitivity_exp2 folk_pubcov virny_flow 32 32 120 [0.25,0.5,1.0] 6 0.5 0.5 0.0 vf_halting_5 200 dh3553"
#    "sensitivity_exp2 folk_pubcov virny_flow 32 32 120 [0.25,0.5,1.0] 7 0.5 0.5 0.0 vf_halting_5 200 dh3553"
#    "sensitivity_exp2 folk_pubcov virny_flow 32 32 120 [0.25,0.5,1.0] 8 0.5 0.5 0.0 vf_halting_5 200 dh3553"
#    "sensitivity_exp2 folk_pubcov virny_flow 32 32 120 [0.25,0.5,1.0] 9 0.5 0.5 0.0 vf_halting_5 200 dh3553"
#    "sensitivity_exp2 folk_pubcov virny_flow 32 32 120 [0.25,0.5,1.0] 10 0.5 0.5 0.0 vf_halting_5 200 dh3553"
#    "sensitivity_exp2 folk_pubcov virny_flow 32 32 120 [0.25,0.5,1.0] 11 0.5 0.5 0.0 vf_halting_5 200 dh3553"
#    "sensitivity_exp2 folk_pubcov virny_flow 32 32 120 [0.25,0.5,1.0] 12 0.5 0.5 0.0 vf_halting_5 200 dh3553"
#    "sensitivity_exp2 folk_pubcov virny_flow 32 32 120 [0.25,0.5,1.0] 13 0.5 0.5 0.0 vf_halting_5 200 dh3553"
#    "sensitivity_exp2 folk_pubcov virny_flow 32 32 120 [0.25,0.5,1.0] 14 0.5 0.5 0.0 vf_halting_5 200 dh3553"
#    "sensitivity_exp2 folk_pubcov virny_flow 32 32 120 [0.25,0.5,1.0] 15 0.5 0.5 0.0 vf_halting_5 200 dh3553"
#    "sensitivity_exp2 folk_pubcov virny_flow 32 32 120 [0.25,0.5,1.0] 16 0.5 0.5 0.0 vf_halting_5 200 dh3553"
#    "sensitivity_exp2 folk_pubcov virny_flow 32 32 120 [0.25,0.5,1.0] 17 0.5 0.5 0.0 vf_halting_5 200 dh3553"
#    "sensitivity_exp2 folk_pubcov virny_flow 32 32 120 [0.25,0.5,1.0] 18 0.5 0.5 0.0 vf_halting_5 200 dh3553"
#    "sensitivity_exp2 folk_pubcov virny_flow 32 32 120 [0.25,0.5,1.0] 19 0.5 0.5 0.0 vf_halting_5 200 dh3553"

#    "sensitivity_exp2 folk_pubcov virny_flow 32 32 120 [0.5,0.75,1.0] 1 0.5 0.5 0.0 vf_halting_6 200 dh3553"
#    "sensitivity_exp2 folk_pubcov virny_flow 32 32 120 [0.5,0.75,1.0] 2 0.5 0.5 0.0 vf_halting_6 200 dh3553"
#    "sensitivity_exp2 folk_pubcov virny_flow 32 32 120 [0.5,0.75,1.0] 3 0.5 0.5 0.0 vf_halting_6 200 dh3553"
#    "sensitivity_exp2 folk_pubcov virny_flow 32 32 120 [0.5,0.75,1.0] 4 0.5 0.5 0.0 vf_halting_6 200 dh3553"
#    "sensitivity_exp2 folk_pubcov virny_flow 32 32 120 [0.5,0.75,1.0] 5 0.5 0.5 0.0 vf_halting_6 200 dh3553"
#    "sensitivity_exp2 folk_pubcov virny_flow 32 32 120 [0.5,0.75,1.0] 6 0.5 0.5 0.0 vf_halting_6 200 dh3553"
#    "sensitivity_exp2 folk_pubcov virny_flow 32 32 120 [0.5,0.75,1.0] 7 0.5 0.5 0.0 vf_halting_6 200 dh3553"
#    "sensitivity_exp2 folk_pubcov virny_flow 32 32 120 [0.5,0.75,1.0] 8 0.5 0.5 0.0 vf_halting_6 200 dh3553"
#    "sensitivity_exp2 folk_pubcov virny_flow 32 32 120 [0.5,0.75,1.0] 9 0.5 0.5 0.0 vf_halting_6 200 dh3553"
#    "sensitivity_exp2 folk_pubcov virny_flow 32 32 120 [0.5,0.75,1.0] 10 0.5 0.5 0.0 vf_halting_6 200 dh3553"
#    "sensitivity_exp2 folk_pubcov virny_flow 32 32 120 [0.5,0.75,1.0] 11 0.5 0.5 0.0 vf_halting_6 200 dh3553"
#    "sensitivity_exp2 folk_pubcov virny_flow 32 32 120 [0.5,0.75,1.0] 12 0.5 0.5 0.0 vf_halting_6 200 dh3553"

#    "sensitivity_exp2 folk_pubcov virny_flow 32 32 120 [0.1,0.25,0.5,1.0] 1 0.5 0.5 0.0 vf_halting_7 200 dh3553"
#    "sensitivity_exp2 folk_pubcov virny_flow 32 32 120 [0.1,0.25,0.5,1.0] 2 0.5 0.5 0.0 vf_halting_7 200 dh3553"
#    "sensitivity_exp2 folk_pubcov virny_flow 32 32 120 [0.1,0.25,0.5,1.0] 3 0.5 0.5 0.0 vf_halting_7 200 dh3553"
#    "sensitivity_exp2 folk_pubcov virny_flow 32 32 120 [0.1,0.25,0.5,1.0] 4 0.5 0.5 0.0 vf_halting_7 200 dh3553"
#    "sensitivity_exp2 folk_pubcov virny_flow 32 32 120 [0.1,0.25,0.5,1.0] 5 0.5 0.5 0.0 vf_halting_7 200 dh3553"
#    "sensitivity_exp2 folk_pubcov virny_flow 32 32 120 [0.1,0.25,0.5,1.0] 6 0.5 0.5 0.0 vf_halting_7 200 dh3553"
#    "sensitivity_exp2 folk_pubcov virny_flow 32 32 120 [0.1,0.25,0.5,1.0] 7 0.5 0.5 0.0 vf_halting_7 200 dh3553"
#    "sensitivity_exp2 folk_pubcov virny_flow 32 32 120 [0.1,0.25,0.5,1.0] 8 0.5 0.5 0.0 vf_halting_7 200 dh3553"
#    "sensitivity_exp2 folk_pubcov virny_flow 32 32 120 [0.1,0.25,0.5,1.0] 9 0.5 0.5 0.0 vf_halting_7 200 dh3553"
#    "sensitivity_exp2 folk_pubcov virny_flow 32 32 120 [0.1,0.25,0.5,1.0] 10 0.5 0.5 0.0 vf_halting_7 200 dh3553"
#    "sensitivity_exp2 folk_pubcov virny_flow 32 32 120 [0.1,0.25,0.5,1.0] 11 0.5 0.5 0.0 vf_halting_7 200 dh3553"
#    "sensitivity_exp2 folk_pubcov virny_flow 32 32 120 [0.1,0.25,0.5,1.0] 12 0.5 0.5 0.0 vf_halting_7 200 dh3553"

#    "sensitivity_exp2 folk_pubcov virny_flow 32 32 120 [0.1,0.5,0.75,1.0] 1 0.5 0.5 0.0 vf_halting_8 200 dh3553"
#    "sensitivity_exp2 folk_pubcov virny_flow 32 32 120 [0.1,0.5,0.75,1.0] 2 0.5 0.5 0.0 vf_halting_8 200 dh3553"
#    "sensitivity_exp2 folk_pubcov virny_flow 32 32 120 [0.1,0.5,0.75,1.0] 3 0.5 0.5 0.0 vf_halting_8 200 dh3553"
#    "sensitivity_exp2 folk_pubcov virny_flow 32 32 120 [0.1,0.5,0.75,1.0] 4 0.5 0.5 0.0 vf_halting_8 200 dh3553"
#    "sensitivity_exp2 folk_pubcov virny_flow 32 32 120 [0.1,0.5,0.75,1.0] 5 0.5 0.5 0.0 vf_halting_8 200 dh3553"
#    "sensitivity_exp2 folk_pubcov virny_flow 32 32 120 [0.1,0.5,0.75,1.0] 6 0.5 0.5 0.0 vf_halting_8 200 dh3553"
#    "sensitivity_exp2 folk_pubcov virny_flow 32 32 120 [0.1,0.5,0.75,1.0] 7 0.5 0.5 0.0 vf_halting_8 200 dh3553"
#    "sensitivity_exp2 folk_pubcov virny_flow 32 32 120 [0.1,0.5,0.75,1.0] 8 0.5 0.5 0.0 vf_halting_8 200 dh3553"
#    "sensitivity_exp2 folk_pubcov virny_flow 32 32 120 [0.1,0.5,0.75,1.0] 9 0.5 0.5 0.0 vf_halting_8 200 dh3553"
#    "sensitivity_exp2 folk_pubcov virny_flow 32 32 120 [0.1,0.5,0.75,1.0] 10 0.5 0.5 0.0 vf_halting_8 200 dh3553"
#    "sensitivity_exp2 folk_pubcov virny_flow 32 32 120 [0.1,0.5,0.75,1.0] 11 0.5 0.5 0.0 vf_halting_8 200 dh3553"
#    "sensitivity_exp2 folk_pubcov virny_flow 32 32 120 [0.1,0.5,0.75,1.0] 12 0.5 0.5 0.0 vf_halting_8 200 dh3553"
)

# Initialize a counter
index=0

# Iterate through the array of job_configs
for job_config in "${job_configs[@]}"
do
    # Split the job_config into separate variables
    read -r exp_name dataset system_name num_workers cpus mem training_set_fractions_for_halting run_num w1 w2 w3 exp_config_suffix max_total_pipelines_num email <<< "$job_config"
    template_file="/home/${email}/projects/virny-flow-experiments/cluster/${exp_name}/${system_name}-${dataset}-template.sbatch"

    # Define the output file name
    output_file="/home/${email}/projects/virny-flow-experiments/cluster/${exp_name}/sbatch_files/${exp_name}_${dataset}_${num_workers}_${exp_config_suffix}_run_${run_num}_${index}_$(date +"%Y%m%d%H%M%S").sbatch"

    # Create an empty file
    touch $output_file

    # Use sed to replace placeholders with actual values
    sed -e "s/<EXP_NAME>/${exp_name}/g" -e "s/<DATASET>/${dataset}/g" -e "s/<NUM_WORKERS>/${num_workers}/g" -e "s/<CPUS>/${cpus}/g" -e "s/<MEM>/${mem}/g" -e "s/<TRAINING_SET_FRACTIONS_FOR_HALTING>/${training_set_fractions_for_halting}/g" -e "s/<RUN_NUM>/${run_num}/g" -e "s/<W1>/${w1}/g" -e "s/<W2>/${w2}/g" -e "s/<W3>/${w3}/g" -e "s/<MAX_TOTAL_PIPELINES_NUM>/${max_total_pipelines_num}/g" -e "s/<EXP_CONFIG_SUFFIX>/${exp_config_suffix}/g" -e "s/<EMAIL>/${email}/g" $template_file > $output_file

    # Execute a SLURM job
    sbatch $output_file

    echo "Job was executed: $output_file"

    # Increment the index
    ((index++))
done
