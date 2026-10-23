"""GPU validation for the batched NLTSA path: parity tests on the device, then
timing of a recording-sized batch (60 channel-epochs × 2500 samples) against the
scalar numpy path on one core.  sbatch gpu.sbatch gpu_check.py"""
import subprocess, sys, time
import jax, numpy as np
from scipy.signal import butter, sosfiltfilt
from neurojax.analysis import nltsa
from neurojax.analysis.nltsa_batched import BATCHED, chunk_size, score_batched

print("devices:", jax.devices(), flush=True)
r = subprocess.run([sys.executable, "-m", "pytest", "src/neurojax/analysis/tests/test_nltsa_batched.py",
                    "-q", "-o", "addopts="], capture_output=True, text=True)
print(r.stdout[-600:], r.stderr[-600:] if r.returncode else "", flush=True)
rng = np.random.default_rng(0); fs = 250.0; T = 2500; B = 60
sos = butter(4, [1, 40], btype="band", fs=fs, output="sos")
X = np.stack([sosfiltfilt(sos, np.cumsum(rng.normal(size=T + 500)))[500:] for _ in range(B)]); X /= X.std(1, keepdims=True)
ms = {n: nltsa.MEASURES[n] for n in BATCHED}
print("chunk size:", chunk_size(T, 3), flush=True)
t = time.perf_counter(); out = score_batched(X, fs, ms); print(f"gpu first call (jit) {time.perf_counter()-t:.1f}s", flush=True)
t = time.perf_counter(); out = score_batched(X, fs, ms); print(f"gpu B={B}: {time.perf_counter()-t:.2f}s", flush=True)
t = time.perf_counter(); ref = {n: np.array([ms[n](X[b], fs) for b in range(8)]) for n in ms}; print(f"numpy 8 items: {time.perf_counter()-t:.1f}s", flush=True)
for n in ms:
    d = np.abs(out[n][:8] - ref[n]); print(f"{n:26s} max|diff| {d.max():.5f}")
print("[done]")
