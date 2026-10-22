#!/usr/bin/env bash
#SBATCH --job-name=cbench
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=20
#SBATCH --mem=60G
#SBATCH --output=/home/mhough/cbench_%j.log
# usage: sbatch run_bench.sbatch <dataset> [extra run_bench.py args]
set -euo pipefail
cd /home/mhough/dev/neurojax
export JAX_PLATFORMS=cpu PYTHONPATH=src OMP_NUM_THREADS=1
DS=$1; shift
.venv-models/bin/python scripts/consciousness_bench/run_bench.py --dataset "$DS" "$@"
