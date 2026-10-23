#!/usr/bin/env bash
#SBATCH --job-name=cbsens
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=20
#SBATCH --mem=60G
#SBATCH --output=/home/mhough/cbsens_%j.log
# usage: sbatch run_sensitivity.sbatch <dataset> <cond_a> <cond_b> [extra args]
set -euo pipefail
cd /home/mhough/dev/neurojax
export JAX_PLATFORMS=cpu PYTHONPATH=src OMP_NUM_THREADS=1
DS=$1; A=$2; B=$3; shift 3
.venv-models/bin/python scripts/consciousness_bench/run_sensitivity.py --dataset "$DS" --cond-a "$A" --cond-b "$B" "$@"
