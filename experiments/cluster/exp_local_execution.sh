#!/bin/bash

# Delay between batches
DELAY=4500

# List of commands
COMMANDS=(
"python3 -m tools.benchmark.exps.run_exp comparison_exp2_folk_pubcov_w32_am folk_pubcov 5 4 3600 none"
"python3 -m tools.benchmark.exps.run_exp comparison_exp2_folk_pubcov_w32_am folk_pubcov 6 4 3600 none"
"python3 -m tools.benchmark.exps.run_exp comparison_exp2_folk_pubcov_w32_am folk_pubcov 7 4 3600 none"
"python3 -m tools.benchmark.exps.run_exp comparison_exp2_folk_pubcov_w32_am folk_pubcov 8 4 3600 none"
"python3 -m tools.benchmark.exps.run_exp comparison_exp2_folk_pubcov_w32_am folk_pubcov 9 4 3600 none"
"python3 -m tools.benchmark.exps.run_exp comparison_exp2_folk_pubcov_w32_am folk_pubcov 10 4 3600 none"
"python3 -m tools.benchmark.exps.run_exp comparison_exp2_heart_w32_am heart 4 4 3600 none"
"python3 -m tools.benchmark.exps.run_exp comparison_exp2_heart_w32_am heart 8 4 3600 none"
"python3 -m tools.benchmark.exps.run_exp comparison_exp2_heart_w32_am heart 9 4 3600 none"
"python3 -m tools.benchmark.exps.run_exp comparison_exp2_heart_w32_am heart 10 4 3600 none"
)

# Total number of commands
TOTAL=${#COMMANDS[@]}

# Run 2 jobs at a time
for (( i=0; i<$TOTAL; i+=2 ))
do
    echo "Starting batch $((i / 2 + 1))..."
    for j in {0..1}
    do
        IDX=$((i + j))
        if [ $IDX -lt $TOTAL ]; then
            CMD="${COMMANDS[$IDX]}"
            LOGFILE="log_job_$(printf "%02d" $IDX).log"
            echo "  Launching: $CMD > $LOGFILE"
            $CMD > "$LOGFILE" 2>&1 &
        fi
    done

    # Sleep for 75 minutes before launching next batch (no waiting on jobs)
    if [ $((i + 2)) -lt $TOTAL ]; then
        echo "Sleeping for $DELAY before starting next batch..."
        sleep $DELAY
    fi
done

echo "All job batches submitted."
