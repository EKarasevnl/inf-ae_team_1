#!/bin/bash
USER=$(whoami)

# ✅ Your jobs directory (adjust if needed)
JOB_DIR="/home/$USER/inf-ae_team_1/jobs/steam"

# ✅ Loop through all files that start with "run" in that folder
for jobfile in "$JOB_DIR"/run*.job
do
    echo "Submitting: $jobfile"
    sbatch "$jobfile"
done
