"""Where does score_epochs spend its time on the GPU node? Synthetic recording
(12 epochs × 10 ch × 2500), batched part vs joblib part vs all-CPU.
sbatch gpu.sbatch gpu_profile_epochs.py"""
import os, time
import jax, numpy as np
from scipy.signal import butter, sosfiltfilt
from neurojax.analysis.nltsa import MEASURES
from neurojax.analysis.nltsa_batched import BATCHED, score_batched
from neurojax.bench.consciousness_bench import score_epochs

n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
print("devices:", jax.devices(), "n_jobs", n_jobs, flush=True)
rng = np.random.default_rng(0); fs = 250.0
sos = butter(4, [1, 40], btype="band", fs=fs, output="sos")
ep = [sosfiltfilt(sos, np.cumsum(rng.normal(size=(10, 3000)), 1))[:, 500:] for _ in range(12)]
bm = {n: MEASURES[n] for n in BATCHED}; rest = {n: m for n, m in MEASURES.items() if n not in BATCHED}
stack = np.stack(ep).reshape(-1, 2500)
score_batched(stack, fs, bm)  # jit
t = time.perf_counter(); score_batched(stack, fs, bm); print(f"batched part, 120 items on device: {time.perf_counter()-t:.1f}s", flush=True)
t = time.perf_counter(); score_epochs(ep, fs, rest, n_jobs=n_jobs, batched=False); print(f"rest ({len(rest)} measures) joblib n_jobs={n_jobs} (first call, pool spawn): {time.perf_counter()-t:.1f}s", flush=True)
t = time.perf_counter(); score_epochs(ep, fs, rest, n_jobs=n_jobs, batched=False); print(f"rest again: {time.perf_counter()-t:.1f}s", flush=True)
t = time.perf_counter(); score_epochs(ep, fs, MEASURES, n_jobs=n_jobs, batched=True); print(f"score_epochs batched=True, all 30: {time.perf_counter()-t:.1f}s", flush=True)
t = time.perf_counter(); score_epochs(ep, fs, MEASURES, n_jobs=n_jobs, batched=False); print(f"score_epochs batched=False, all 30, n_jobs={n_jobs}: {time.perf_counter()-t:.1f}s", flush=True)
for n in rest:
    t = time.perf_counter(); [rest[n](ep[0] if rest[n].multivariate else ep[0][c], fs) for c in range(10)]; print(f"  {n:24s} {(time.perf_counter()-t)/10*1000:7.1f} ms/channel (parent, 1 core)", flush=True)
print("[done]")
