"""Per-component timing of the batched NLTSA path on the default device.
sbatch gpu.sbatch gpu_profile.py [T] [B]"""
import sys, time
import jax, jax.numpy as jnp, numpy as np
from functools import partial
from neurojax.analysis.entropy import _embed
from neurojax.analysis import nltsa_batched as nb

T = int(sys.argv[1]) if len(sys.argv) > 1 else 2500
B = int(sys.argv[2]) if len(sys.argv) > 2 else 8
print("devices:", jax.devices(), "T", T, "B", B, flush=True)
rng = np.random.default_rng(0)
X = jnp.asarray(np.cumsum(rng.normal(size=(B, T)), 1), jnp.float32)
X = X - X.mean(1, keepdims=True)


def timeit(name, f, *a):
    f = jax.jit(f)
    t = time.perf_counter(); r = f(*a); jax.block_until_ready(r); tj = time.perf_counter() - t
    t = time.perf_counter(); r = f(*a); jax.block_until_ready(r); tr = time.perf_counter() - t
    print(f"{name:34s} jit {tj:8.2f}s   run {tr:8.3f}s   per item {tr/B*1000:8.1f} ms", flush=True)
    return r


emb = jax.vmap(lambda x: _embed(x, 3, 1))(X)
D = timeit("dist (B,N,N)", jax.vmap(nb._dist), emb)
thr = timeit("percentile", jax.vmap(lambda D: jnp.percentile(D, 5.0)), D)
R = D <= thr[:, None, None]
N = R.shape[1]
eye = jnp.eye(N, dtype=bool)
Roff = R & ~eye
i = jnp.arange(N)[:, None]; k = jnp.arange(2 * N - 1)[None, :]; j = i + k - (N - 1)
valid = (j >= 0) & (j < N)
S = timeit("skew diagonals", jax.vmap(lambda R: jnp.take_along_axis(R, jnp.clip(j, 0, N - 1), axis=1) & valid), Roff)
timeit("run_length_hist (vertical)", jax.vmap(nb._run_length_hist), R)
timeit("run_length_hist (diagonal)", jax.vmap(nb._run_length_hist), S)
A = Roff.astype(jnp.float32)
timeit("adjacency A@A (highest)", jax.vmap(lambda A: jnp.matmul(A, A, precision="highest")), A)
timeit("rqa full (vmap)", jax.vmap(partial(nb._rqa_one, m=3, tau=1, recurrence_rate=0.05, l_min=2, v_min=2)), X)
timeit("lyapunov (vmap)", jax.vmap(partial(nb._lyap_one, fs=250.0, m=2, tau=1, theiler=25, k_max=20)), X)
timeit("corr dim (vmap)", jax.vmap(partial(nb._cd_one, m=5, tau=1, n_r=10)), X)
cs = jnp.asarray(np.random.RandomState(0).uniform(np.pi / 5, 4 * np.pi / 5, 100), jnp.float32)
timeit("chaos_01 (vmap)", jax.vmap(partial(nb._chaos_one, cs=cs, n_cut=T // 10)), X)
print("[done]")

# entropy family and lax.map (chunked) vs plain vmap
timeit("sample_entropy (vmap)", jax.vmap(lambda x: nb._sampen_one(x, 2, 0.2 * jnp.std(x))), X)
timeit("mse_sum (vmap)", jax.vmap(partial(nb._mse_one, scales=20, m=2, r=0.15)), X)
Xb = jnp.concatenate([X] * (64 // B + 1))[:64]
for name, f in (("rqa", partial(nb._rqa_one, m=3, tau=1, recurrence_rate=0.05, l_min=2, v_min=2)),
                ("mse", partial(nb._mse_one, scales=20, m=2, r=0.15))):
    for chunk in (8, 16, 64):
        g = lambda Z, f=f, chunk=chunk: jax.lax.map(f, Z, batch_size=chunk)
        t = time.perf_counter(); r = jax.jit(g)(Xb); jax.block_until_ready(r)
        t = time.perf_counter(); r = jax.jit(g)(Xb); jax.block_until_ready(r)
        print(f"{name} lax.map B=64 chunk={chunk:3d}: run {time.perf_counter()-t:7.3f}s  per item {(time.perf_counter()-t)/64*1000:7.1f} ms", flush=True)
print("[done2]")
