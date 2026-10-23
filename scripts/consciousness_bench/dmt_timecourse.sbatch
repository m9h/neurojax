#!/usr/bin/env bash
#SBATCH --job-name=dmt-tc
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=20
#SBATCH --mem=80G
#SBATCH --output=/home/mhough/dmt_tc_%j.log
set -euo pipefail
cd /home/mhough/dev/neurojax
export JAX_PLATFORMS=cpu PYTHONPATH=src OMP_NUM_THREADS=1
.venv-models/bin/python scripts/consciousness_bench/dmt_timecourse.py "$@"
