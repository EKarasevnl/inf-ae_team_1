#!/bin/bash

# ✅ Your jobs directory (adjust if needed)
JOB_DIR="/home/scur2708/inf-ae_team_1/jobs/ml-1m"

# ✅ Loop through all files that start with "run" in that folder
for jobfile in "$JOB_DIR"/run*.job
do
    echo "Submitting: $jobfile"
    sbatch "$jobfile"
done
