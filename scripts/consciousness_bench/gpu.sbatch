#!/usr/bin/env bash
#SBATCH --job-name=cbgpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:gb10:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --output=/home/mhough/cbgpu_%j.log
# usage: sbatch gpu.sbatch <script.py under scripts/consciousness_bench> [args]
# The O(T²) measures (recurrence profile, Lyapunov, correlation dimension, 0-1 chaos) run batched on the
# GB10; the rest on 8 joblib workers (LZ76 is sequential and dominates that side) pinned to the CPU backend by score_epochs. Preallocation off so the
# parent can size its chunks from the real free memory and workers never grab the device.
set -euo pipefail
cd /home/mhough/dev/neurojax
export PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
SCRIPT=$1; shift
.venv-models/bin/python scripts/consciousness_bench/"$SCRIPT" "$@"
