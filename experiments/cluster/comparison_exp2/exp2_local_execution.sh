#!/bin/bash

# Delay between batches
DELAY=3900

# List of commands
COMMANDS=(
"python3 -m tools.benchmark.exps.run_exp comparison_exp2_folk_emp_w32_am folk_emp 31 32 3600 none"
"python3 -m tools.benchmark.exps.run_exp comparison_exp2_folk_emp_w32_am folk_emp 32 32 3600 none"
"python3 -m tools.benchmark.exps.run_exp comparison_exp2_folk_emp_w32_am folk_emp 33 32 3600 none"
"python3 -m tools.benchmark.exps.run_exp comparison_exp2_folk_emp_w32_am folk_emp 34 32 3600 none"
"python3 -m tools.benchmark.exps.run_exp comparison_exp2_folk_emp_w32_am folk_emp 35 32 3600 none"

"python3 -m tools.benchmark.exps.run_exp comparison_exp2_folk_emp_w32_am folk_emp 36 32 3600 none"
"python3 -m tools.benchmark.exps.run_exp comparison_exp2_folk_emp_w32_am folk_emp 37 32 3600 none"
"python3 -m tools.benchmark.exps.run_exp comparison_exp2_folk_emp_w32_am folk_emp 38 32 3600 none"
"python3 -m tools.benchmark.exps.run_exp comparison_exp2_folk_emp_w32_am folk_emp 39 32 3600 none"
"python3 -m tools.benchmark.exps.run_exp comparison_exp2_folk_emp_w32_am folk_emp 40 32 3600 none"

"python3 -m tools.benchmark.exps.run_exp comparison_exp2_folk_pubcov_w32_am folk_pubcov 31 32 3600 none"
"python3 -m tools.benchmark.exps.run_exp comparison_exp2_folk_pubcov_w32_am folk_pubcov 32 32 3600 none"
"python3 -m tools.benchmark.exps.run_exp comparison_exp2_folk_pubcov_w32_am folk_pubcov 33 32 3600 none"
"python3 -m tools.benchmark.exps.run_exp comparison_exp2_folk_pubcov_w32_am folk_pubcov 34 32 3600 none"
"python3 -m tools.benchmark.exps.run_exp comparison_exp2_folk_pubcov_w32_am folk_pubcov 35 32 3600 none"

"python3 -m tools.benchmark.exps.run_exp comparison_exp2_folk_pubcov_w32_am folk_pubcov 36 32 3600 none"
"python3 -m tools.benchmark.exps.run_exp comparison_exp2_folk_pubcov_w32_am folk_pubcov 37 32 3600 none"
"python3 -m tools.benchmark.exps.run_exp comparison_exp2_folk_pubcov_w32_am folk_pubcov 38 32 3600 none"
"python3 -m tools.benchmark.exps.run_exp comparison_exp2_folk_pubcov_w32_am folk_pubcov 39 32 3600 none"
"python3 -m tools.benchmark.exps.run_exp comparison_exp2_folk_pubcov_w32_am folk_pubcov 40 32 3600 none"

"python3 -m tools.benchmark.exps.run_exp comparison_exp2_heart_w32_am heart 31 32 3600 none"
"python3 -m tools.benchmark.exps.run_exp comparison_exp2_heart_w32_am heart 32 32 3600 none"
"python3 -m tools.benchmark.exps.run_exp comparison_exp2_heart_w32_am heart 33 32 3600 none"
"python3 -m tools.benchmark.exps.run_exp comparison_exp2_heart_w32_am heart 34 32 3600 none"
"python3 -m tools.benchmark.exps.run_exp comparison_exp2_heart_w32_am heart 35 32 3600 none"

"python3 -m tools.benchmark.exps.run_exp comparison_exp2_heart_w32_am heart 36 32 3600 none"
"python3 -m tools.benchmark.exps.run_exp comparison_exp2_heart_w32_am heart 37 32 3600 none"
"python3 -m tools.benchmark.exps.run_exp comparison_exp2_heart_w32_am heart 38 32 3600 none"
"python3 -m tools.benchmark.exps.run_exp comparison_exp2_heart_w32_am heart 39 32 3600 none"
"python3 -m tools.benchmark.exps.run_exp comparison_exp2_heart_w32_am heart 40 32 3600 none"
)

# Total number of commands
TOTAL=${#COMMANDS[@]}

# Run 5 jobs at a time
for (( i=0; i<$TOTAL; i+=5 ))
do
    echo "Starting batch $((i / 5 + 1))..."
    for j in {0..4}
    do
        IDX=$((i + j))
        if [ $IDX -lt $TOTAL ]; then
            CMD="${COMMANDS[$IDX]}"
            LOGFILE="log_job_$(printf "%02d" $IDX).log"
            echo "  Launching: $CMD > $LOGFILE"
            $CMD > "$LOGFILE" 2>&1 &
        fi
    done

    # Sleep for 70 minutes before launching next batch (no waiting on jobs)
    if [ $((i + 5)) -lt $TOTAL ]; then
        echo "Sleeping for $DELAY before starting next batch..."
        sleep $DELAY
    fi
done

echo "All job batches submitted."
