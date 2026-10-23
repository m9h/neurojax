#!/usr/bin/env bash
#SBATCH --job-name=cbsens
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=20
#SBATCH --mem=60G
#SBATCH --output=/home/mhough/cbsens_%j.log
# usage: sbatch run_sensitivity.sbatch <dataset> <cond_a> <cond_b> [extra args]
set -euo pipefail
cd /home/mhough/dev/neurojax
export JAX_PLATFORMS=cpu PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# one XLA thread per joblib worker: with 20 workers the default XLA CPU thread pools oversubscribe the node (load 37 on 20 cores)
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
DS=$1; A=$2; B=$3; shift 3
.venv-models/bin/python scripts/consciousness_bench/run_sensitivity.py --dataset "$DS" --cond-a "$A" --cond-b "$B" "$@"
