#!/usr/bin/env bash
#SBATCH --job-name=dcm-sim
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=20
#SBATCH --mem=80G
#SBATCH --output=/home/mhough/dcm_sim_%j.log
set -euo pipefail
cd /home/mhough/dev/neurojax
export JAX_PLATFORMS=cpu PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
.venv-models/bin/python scripts/consciousness_bench/dcm_simulate_measures.py "$@"
