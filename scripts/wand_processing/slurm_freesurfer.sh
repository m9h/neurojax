#!/bin/bash
# ============================================================================
# Slurm array job: FreeSurfer recon-all for all WAND subjects
# ============================================================================
# Submits one job per subject via Slurm array, each calling podman exec
# into the freesurfer container. GPU shared across sequential jobs.
#
# Usage:
#   # Generate subject list
#   ls -d /data/raw/wand/sub-* | xargs -n1 basename > /data/datasets/wand/derivatives/freesurfer/subjects.txt
#
#   # Submit all (2 concurrent, 8 threads each = 16 cores)
#   sbatch --array=1-$(wc -l < /data/datasets/wand/derivatives/freesurfer/subjects.txt)%2 \
#     scripts/wand_processing/slurm_freesurfer.sh
#
#   # Or a subset
#   sbatch --array=1-10%2 scripts/wand_processing/slurm_freesurfer.sh
# ============================================================================

#SBATCH --job-name=fs-recon
#SBATCH --partition=batch
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=/data/raw/wand/derivatives/freesurfer/logs/fs-recon_%A_%a.out
#SBATCH --error=/data/raw/wand/derivatives/freesurfer/logs/fs-recon_%A_%a.err

set -euo pipefail

SUBJECT_LIST=/data/raw/wand/derivatives/freesurfer/subjects.txt
SUBJECT=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$SUBJECT_LIST")

if [ -z "$SUBJECT" ]; then
    echo "ERROR: No subject at line ${SLURM_ARRAY_TASK_ID} in ${SUBJECT_LIST}"
    exit 1
fi

echo "Job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}: ${SUBJECT}"
echo "Started: $(date)"

podman exec freesurfer bash /scripts/fs_recon_container.sh "${SUBJECT}" "${SLURM_CPUS_PER_TASK}"

echo "Finished: $(date)"
