#!/bin/bash
#SBATCH --job-name=sysid-posthoc
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/sysid_posthoc_%j.out
#SBATCH --error=slurm/sysid_posthoc_%j.err

set -euo pipefail
cd /home/mhough/dev/neurojax
source .venv/bin/activate
echo "=== Job $SLURM_JOB_ID — $(date) ==="
python examples/compare_sysid_wand.py \
    --subject sub-08033 \
    --wand-root /data/raw/wand \
    --n-states 6 \
    --max-duration 60 \
    --skip-source-recon \
    --save-dir data/wand_parcellated
echo "Done: $(date)"
