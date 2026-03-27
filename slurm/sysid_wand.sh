#!/bin/bash
#SBATCH --job-name=sysid-wand
#SBATCH --partition=gpu
#SBATCH --gres=gpu:gb10:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/sysid_wand_%j.out
#SBATCH --error=slurm/sysid_wand_%j.err

# WAND MEG systems identification comparison
# Submit from neurojax project root: sbatch slurm/sysid_wand.sh

set -euo pipefail

cd /home/mhough/dev/neurojax

echo "=== Job $SLURM_JOB_ID on $(hostname) ==="
echo "Started: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"

# Use project venv
source .venv/bin/activate

# Run the comparison pipeline
python examples/compare_sysid_wand.py \
    --subject sub-08033 \
    --wand-root /data/raw/wand \
    --n-states 8 \
    --max-duration 120 \
    --save-dir data/wand_parcellated

echo "Finished: $(date)"
