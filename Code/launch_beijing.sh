#!/bin/bash
# Submit one Beijing horizon job per horizon value.
#
#   bash launch_beijing.sh              # default horizon ladder below
#   bash launch_beijing.sh 24 72 168    # custom horizons
#
# Each job runs ~20 fixed-origin fits for its horizon and writes
# beijing_out/beijing_horizon_summary_H<h>_seed<seed>.csv

SEED=${SEED:-22}
HORIZONS=("$@")
if [ ${#HORIZONS[@]} -eq 0 ]; then
  HORIZONS=(12 24 48 72 120 168 240)   # default ladder; edit as needed
fi

mkdir -p logs beijing_out
for h in "${HORIZONS[@]}"; do
  sbatch --export=ALL,HORIZON=$h,SEED=$SEED submit_beijing.slurm
  echo "submitted horizon $h (seed $SEED)"
done
echo "submitted ${#HORIZONS[@]} jobs"
