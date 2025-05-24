# Define the list of tuples (exp_name, dataset, case_study, run_num, w1, w2, w3, exp_config_suffix, max_trials, email)
declare -a job_configs=(
#    # Case study 1
#    "case_studies_exp folk_pubcov cs1 1 0.5 0.5 0.0 w_acc_0_5_w_fair_0_5 200 dh3553"
#    "case_studies_exp folk_pubcov cs1 1 0.25 0.75 0.0 w_acc_0_25_w_fair_0_75 200 dh3553"
#    "case_studies_exp folk_pubcov cs1 1 0.75 0.25 0.0 w_acc_0_75_w_fair_0_25 200 dh3553"
#
#    "case_studies_exp folk_pubcov cs1 2 0.5 0.5 0.0 w_acc_0_5_w_fair_0_5 200 dh3553"
#    "case_studies_exp folk_pubcov cs1 2 0.25 0.75 0.0 w_acc_0_25_w_fair_0_75 200 dh3553"
#    "case_studies_exp folk_pubcov cs1 2 0.75 0.25 0.0 w_acc_0_75_w_fair_0_25 200 dh3553"
#
#    "case_studies_exp folk_pubcov cs1 3 0.5 0.5 0.0 w_acc_0_5_w_fair_0_5 200 dh3553"
#    "case_studies_exp folk_pubcov cs1 3 0.25 0.75 0.0 w_acc_0_25_w_fair_0_75 200 dh3553"
#    "case_studies_exp folk_pubcov cs1 3 0.75 0.25 0.0 w_acc_0_75_w_fair_0_25 200 dh3553"
#
#    "case_studies_exp folk_pubcov cs1 4 0.5 0.5 0.0 w_acc_0_5_w_fair_0_5 200 dh3553"
#    "case_studies_exp folk_pubcov cs1 4 0.25 0.75 0.0 w_acc_0_25_w_fair_0_75 200 dh3553"
#    "case_studies_exp folk_pubcov cs1 4 0.75 0.25 0.0 w_acc_0_75_w_fair_0_25 200 dh3553"
#
#    "case_studies_exp folk_pubcov cs1 5 0.5 0.5 0.0 w_acc_0_5_w_fair_0_5 200 dh3553"
#    "case_studies_exp folk_pubcov cs1 5 0.25 0.75 0.0 w_acc_0_25_w_fair_0_75 200 dh3553"
#    "case_studies_exp folk_pubcov cs1 5 0.75 0.25 0.0 w_acc_0_75_w_fair_0_25 200 dh3553"
#
#    # Case study 2
#    "case_studies_exp folk_pubcov cs2 1 0.5 0.5 0.0 w_acc_0_5_w_fair_0_5 200 dh3553"
#    "case_studies_exp folk_pubcov cs2 1 0.25 0.75 0.0 w_acc_0_25_w_fair_0_75 200 dh3553"
#    "case_studies_exp folk_pubcov cs2 1 0.75 0.25 0.0 w_acc_0_75_w_fair_0_25 200 dh3553"
#
#    "case_studies_exp folk_pubcov cs2 2 0.5 0.5 0.0 w_acc_0_5_w_fair_0_5 200 dh3553"
#    "case_studies_exp folk_pubcov cs2 2 0.25 0.75 0.0 w_acc_0_25_w_fair_0_75 200 dh3553"
#    "case_studies_exp folk_pubcov cs2 2 0.75 0.25 0.0 w_acc_0_75_w_fair_0_25 200 dh3553"
#
#    "case_studies_exp folk_pubcov cs2 3 0.5 0.5 0.0 w_acc_0_5_w_fair_0_5 200 dh3553"
#    "case_studies_exp folk_pubcov cs2 3 0.25 0.75 0.0 w_acc_0_25_w_fair_0_75 200 dh3553"
#    "case_studies_exp folk_pubcov cs2 3 0.75 0.25 0.0 w_acc_0_75_w_fair_0_25 200 dh3553"
#
#    "case_studies_exp folk_pubcov cs2 4 0.5 0.5 0.0 w_acc_0_5_w_fair_0_5 200 dh3553"
#    "case_studies_exp folk_pubcov cs2 4 0.25 0.75 0.0 w_acc_0_25_w_fair_0_75 200 dh3553"
#    "case_studies_exp folk_pubcov cs2 4 0.75 0.25 0.0 w_acc_0_75_w_fair_0_25 200 dh3553"
#
#    "case_studies_exp folk_pubcov cs2 5 0.5 0.5 0.0 w_acc_0_5_w_fair_0_5 200 dh3553"
#    "case_studies_exp folk_pubcov cs2 5 0.25 0.75 0.0 w_acc_0_25_w_fair_0_75 200 dh3553"
#    "case_studies_exp folk_pubcov cs2 5 0.75 0.25 0.0 w_acc_0_75_w_fair_0_25 200 dh3553"

    # Case study 3
    "case_studies_exp folk_pubcov cs3 1 0.33 0.33 0.33 w_acc_0_33_w_fair1_0_33_w_fair2_0_33 100 dh3553"
    "case_studies_exp folk_pubcov cs3 1 0.5 0.25 0.25 w_acc_0_5_w_fair1_0_25_w_fair2_0_25 100 dh3553"
    "case_studies_exp folk_pubcov cs3 1 0.25 0.5 0.25 w_acc_0_25_w_fair1_0_5_w_fair2_0_25 100 dh3553"
    "case_studies_exp folk_pubcov cs3 1 0.25 0.25 0.5 w_acc_0_25_w_fair1_0_25_w_fair2_0_5 100 dh3553"

    "case_studies_exp folk_pubcov cs3 2 0.33 0.33 0.33 w_acc_0_33_w_fair1_0_33_w_fair2_0_33 100 dh3553"
    "case_studies_exp folk_pubcov cs3 2 0.5 0.25 0.25 w_acc_0_5_w_fair1_0_25_w_fair2_0_25 100 dh3553"
    "case_studies_exp folk_pubcov cs3 2 0.25 0.5 0.25 w_acc_0_25_w_fair1_0_5_w_fair2_0_25 100 dh3553"
    "case_studies_exp folk_pubcov cs3 2 0.25 0.25 0.5 w_acc_0_25_w_fair1_0_25_w_fair2_0_5 100 dh3553"

    "case_studies_exp folk_pubcov cs3 3 0.33 0.33 0.33 w_acc_0_33_w_fair1_0_33_w_fair2_0_33 100 dh3553"
    "case_studies_exp folk_pubcov cs3 3 0.5 0.25 0.25 w_acc_0_5_w_fair1_0_25_w_fair2_0_25 100 dh3553"
    "case_studies_exp folk_pubcov cs3 3 0.25 0.5 0.25 w_acc_0_25_w_fair1_0_5_w_fair2_0_25 100 dh3553"
    "case_studies_exp folk_pubcov cs3 3 0.25 0.25 0.5 w_acc_0_25_w_fair1_0_25_w_fair2_0_5 100 dh3553"

    "case_studies_exp folk_pubcov cs3 4 0.33 0.33 0.33 w_acc_0_33_w_fair1_0_33_w_fair2_0_33 100 dh3553"
    "case_studies_exp folk_pubcov cs3 4 0.5 0.25 0.25 w_acc_0_5_w_fair1_0_25_w_fair2_0_25 100 dh3553"
    "case_studies_exp folk_pubcov cs3 4 0.25 0.5 0.25 w_acc_0_25_w_fair1_0_5_w_fair2_0_25 100 dh3553"
    "case_studies_exp folk_pubcov cs3 4 0.25 0.25 0.5 w_acc_0_25_w_fair1_0_25_w_fair2_0_5 100 dh3553"

    "case_studies_exp folk_pubcov cs3 5 0.33 0.33 0.33 w_acc_0_33_w_fair1_0_33_w_fair2_0_33 100 dh3553"
    "case_studies_exp folk_pubcov cs3 5 0.5 0.25 0.25 w_acc_0_5_w_fair1_0_25_w_fair2_0_25 100 dh3553"
    "case_studies_exp folk_pubcov cs3 5 0.25 0.5 0.25 w_acc_0_25_w_fair1_0_5_w_fair2_0_25 100 dh3553"
    "case_studies_exp folk_pubcov cs3 5 0.25 0.25 0.5 w_acc_0_25_w_fair1_0_25_w_fair2_0_5 100 dh3553"
)

# Initialize a counter
index=0

# Iterate through the array of job_configs
for job_config in "${job_configs[@]}"
do
    # Split the job_config into separate variables
    read -r exp_name dataset case_study run_num w1 w2 w3 exp_config_suffix max_trials email <<< "$job_config"
    template_file="/home/${email}/projects/virny-flow-experiments/cluster/${exp_name}/virny-flow-${dataset}-${case_study}-template.sbatch"

    # Define the output file name
    output_file="/home/${email}/projects/virny-flow-experiments/cluster/${exp_name}/sbatch_files/${exp_name}_${dataset}_${case_study}_${exp_config_suffix}_run_${run_num}_${index}_$(date +"%Y%m%d%H%M%S").sbatch"

    # Create an empty file
    touch $output_file

    # Use sed to replace placeholders with actual values
    sed -e "s/<EXP_NAME>/${exp_name}/g" -e "s/<DATASET>/${dataset}/g" -e "s/<CASE_STUDY>/${case_study}/g" -e "s/<RUN_NUM>/${run_num}/g" -e "s/<W1>/${w1}/g" -e "s/<W2>/${w2}/g" -e "s/<W3>/${w3}/g" -e "s/<MAX_TRIALS>/${max_trials}/g" -e "s/<EXP_CONFIG_SUFFIX>/${exp_config_suffix}/g" -e "s/<EMAIL>/${email}/g" $template_file > $output_file

    # Execute a SLURM job
    sbatch $output_file

    echo "Job was executed: $output_file"

    # Increment the index
    ((index++))
done
