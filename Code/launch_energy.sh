#!/bin/bash
# Submit one energy horizon job per horizon value.
#   bash launch_energy.sh            # default ladder
#   bash launch_energy.sh 24 72      # custom horizons
SEED=${SEED:-22}
HORIZONS=("$@")
if [ ${#HORIZONS[@]} -eq 0 ]; then
  HORIZONS=(12 24 48 72 120 168 240 336)   # energy ladder (long series, no thin tail)
fi
mkdir -p logs energy_out
for h in "${HORIZONS[@]}"; do
  sbatch --export=ALL,HORIZON=$h,SEED=$SEED submit_energy.slurm
  echo "submitted horizon $h (seed $SEED)"
done
echo "submitted ${#HORIZONS[@]} jobs"
