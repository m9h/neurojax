"""Compile and run every batched measure at every epoch length the sensitivity
sweep uses (2, 5, 10, 20 s at 250 Hz) with the default chunk, so XLA codegen
failures (Triton fusion element limits) surface before a long job.
sbatch gpu.sbatch gpu_shapes.py"""
import time
import jax, numpy as np
from neurojax.analysis.nltsa import MEASURES
from neurojax.analysis.nltsa_batched import BATCHED, chunk_size, score_batched

print("devices:", jax.devices(), flush=True)
rng = np.random.default_rng(0)
ms = {n: MEASURES[n] for n in BATCHED}
for T in (500, 1250, 2500, 5000):
    B = 2 * 64 + 8  # two full chunks + a remainder
    X = np.cumsum(rng.normal(size=(B, T)), 1).astype(np.float32)
    t = time.perf_counter()
    out = score_batched(X, 250.0, ms)
    print(f"T={T:5d} chunk={chunk_size(T, 3):3d} B={B}: {time.perf_counter()-t:6.1f}s  "
          + " ".join(f"{n}={np.nanmean(v):.3g}" for n, v in out.items()), flush=True)
print("[done]")
