"""Generative effective connectivity (GEC) by true autodiff, for the OU/Langevin brain model.

Follows Berjaga-Buisan et al. (2026) Cell Reports 45:117782 (their Methods, "Model optimization"),
which fits an asymmetric coupling matrix B so the model reproduces two empirical observables:

    FC_empirical      zero-lag correlation
    FS_empirical(tau) time-lagged covariance, normalized by sqrt(K_ii(0) K_jj(0))

Their model is  dK/dt = -BK - KB^T + Q,  steady state  BK + KB^T = Q,  KS(tau) = expm(-tau B) K.
Internally we use A = -B so the Lyapunov equation matches the usual A K + K A^T + Q = 0.

Difference from the reference implementation: the published fit uses a *heuristic pseudo-gradient*
    B_ij += alpha (FC_emp - FC_mod)_ij + delta (FS_emp - FS_mod)_ij
which treats dFC/dB as the identity. Here the loss is differentiated **through** the Lyapunov solve
and the matrix exponential, giving the true gradient; and the 1000 random restarts the reference
farms out to SLURM become a single `vmap`.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl

from .fdt import lyapunov


def normalize_lagged(K_tau: jnp.ndarray, K0: jnp.ndarray) -> jnp.ndarray:
    """FS_ij = KS_ij(tau) / sqrt(K_ii(0) K_jj(0))  (the paper's normalization)."""
    d = jnp.sqrt(jnp.clip(jnp.diag(K0), 1e-12))
    return K_tau / jnp.outer(d, d)


def empirical_observables(X: jnp.ndarray, lag: int):
    """X: (n_channels, n_samples), already filtered/downsampled. Returns (FC, FS(tau))."""
    X = X - X.mean(axis=1, keepdims=True)
    T = X.shape[1]
    K0 = (X @ X.T) / (T - 1)
    KS = (X[:, :T - lag] @ X[:, lag:].T) / (T - lag - 1)
    return normalize_lagged(K0, K0), normalize_lagged(KS, K0)


def _model_observables(A, Q, tau):
    K = lyapunov(A, Q)
    KS = jsl.expm(A * tau) @ K
    return normalize_lagged(K, K), normalize_lagged(KS, K), K


def gec_loss(A, Q, FC_emp, FS_emp, tau, stab_weight=10.0):
    FC_m, FS_m, K = _model_observables(A, Q, tau)
    l_fc = jnp.sum((FC_emp - FC_m) ** 2)
    l_fs = jnp.sum((FS_emp - FS_m) ** 2)
    # stability: the Lyapunov solution must be positive definite (K is symmetric -> eigvalsh)
    min_eig = jnp.min(jnp.linalg.eigvalsh((K + K.T) / 2))
    pen = jax.nn.relu(1e-6 - min_eig) ** 2
    return l_fc + l_fs + stab_weight * pen


def unpack_A(params, eps=0.05):
    """Guaranteed-Hurwitz parameterization that also splits the physics:

        A = -(L L^T + eps I)  +  (M - M^T)
             \____________/      \_________/
              dissipative (P)      irreversible (K, antisymmetric)

    Since A + A^T = -2(L L^T + eps I) < 0, every eigenvalue has negative real part, so the
    Lyapunov solve can never blow up mid-optimization (the reference implementation initializes
    stable and hopes the heuristic updates keep it there). The antisymmetric block K is precisely
    the part that breaks detailed balance -- with K = 0 the model is in equilibrium and the FDT
    violation is exactly zero (see thermo.fdt tests).
    """
    L, M = params["L"], params["M"]
    n = L.shape[0]
    P = L @ L.T + eps * jnp.eye(n)
    K = M - M.T
    return -P + K


def init_params(key, n, scale=0.1):
    kL, kM = jax.random.split(key)
    return {"L": jax.random.normal(kL, (n, n)) * scale,
            "M": jax.random.normal(kM, (n, n)) * scale}


def fit_gec(FC_emp, FS_emp, tau, key, n_steps=400, lr=1e-2, sigma=1.0, scale=0.1, eps=0.05):
    """Fit A by Adam on the TRUE gradient (through the Lyapunov solve). Returns (A, losses)."""
    import optax
    n = FC_emp.shape[0]
    Q = jnp.eye(n) * sigma
    params = init_params(key, n, scale)
    opt = optax.adam(lr)
    state = opt.init(params)

    def loss_of(p):
        return gec_loss(unpack_A(p, eps), Q, FC_emp, FS_emp, tau, stab_weight=0.0)

    loss_fn = jax.value_and_grad(loss_of)

    def step(carry, _):
        p, st = carry
        loss, g = loss_fn(p)
        upd, st = opt.update(g, st)
        return (optax.apply_updates(p, upd), st), loss

    (params, _), losses = jax.lax.scan(step, (params, state), None, length=n_steps)
    return unpack_A(params, eps), losses


def fit_gec_restarts(FC_emp, FS_emp, tau, key, n_restarts=16, **kw):
    """The reference pipeline's 1000 SLURM restarts, as one vmapped computation."""
    keys = jax.random.split(key, n_restarts)
    A, losses = jax.vmap(lambda k: fit_gec(FC_emp, FS_emp, tau, k, **kw))(keys)
    return A, losses[:, -1]
