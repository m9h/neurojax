#!/usr/bin/env bash
#SBATCH --job-name=dmt-tc
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=20
#SBATCH --mem=80G
#SBATCH --output=/home/mhough/dmt_tc_%j.log
set -euo pipefail
cd /home/mhough/dev/neurojax
export JAX_PLATFORMS=cpu PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# one XLA thread per joblib worker: with 20 workers the default XLA CPU thread pools oversubscribe the node (load 37 on 20 cores)
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
.venv-models/bin/python scripts/consciousness_bench/dmt_timecourse.py "$@"
