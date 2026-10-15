"""Canonical microcircuit DCM for cross-spectral density, a line-by-line JAX port of SPM.

State layout follows SPM: ``x`` is ``(n_sources, 8)`` with columns
(ss voltage, ss conductance, sp voltage, sp conductance, ii current,
ii conductance, dp voltage, dp conductance) and the vectorised state is
column-major (all sources for state 1, then state 2, ...), which is what
``spm_vec`` does to an ``n x 8`` matrix and what the lead field
``kron(P.J, L)`` assumes.

Parameters are a dict with SPM's field names, on SPM's scales (log scaling
of fixed defaults): ``A`` (4 matrices, forward i/ii and backward i/ii),
``B`` (per between-trial effect), ``N`` (modulatory per effect), ``M``,
``C``, ``T`` (n x 4), ``G`` (n x 4), ``D`` (n x n), ``S``, ``L``
(electrode gains, linear), ``J`` (1 x 8 state contributions), ``a``
(2 x n innovations), ``b`` (2 x 1 non-specific noise), ``c`` (2 x n
specific noise), ``d`` (nd x n innovation DCT coefficients).

Endogenous input (SPM's ``M.u``) is the convention used for CSD: each
source has one input entering the spiny-stellate conductance scaled by 512.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

CMC_DEFAULTS = {
    "E": np.array([1.0, 0.5, 1.0, 0.5]) * 200.0,              # extrinsic rates
    "G": np.array([4, 4, 8, 4, 4, 2, 4, 4, 2, 1.0]) * 200.0,   # intrinsic rates
    "T": np.array([2.0, 2.0, 16.0, 28.0]),                     # synaptic time constants (ms)
    "R": 2.0 / 3.0,                                            # sigmoid slope
    "di": 1.0,                                                 # intrinsic delay (ms)
    "de": 8.0,                                                 # extrinsic delay (ms)
}
_J_INDEX = np.array([7, 2, 3, 4, 1, 5, 6, 8, 9, 10]) - 1       # P.G(:, i) scales G(:, j(i))


def _vec(x):
    """Column-major vectorisation of an (n, 8) state, as spm_vec does."""
    return x.T.reshape(-1)


def _unvec(v, n):
    return v.reshape(8, n).T


def fx_cmc(x, u, P, pF=CMC_DEFAULTS):
    """``dx/dt`` for the canonical microcircuit; *x* is (n, 8), *u* is (n,) endogenous input."""
    n = x.shape[0]
    E, G0, T0 = pF["E"], pF["G"], pF["T"]
    if n > 1:
        A = [jnp.exp(P["A"][i]) * E[i] for i in range(4)]
        A2 = []
        for Ai in A:
            lat = (Ai > jnp.exp(-8.0)) & (Ai.T > jnp.exp(-8.0))
            A2.append(Ai / (1.0 + 4.0 * lat))
        A = A2
    else:
        A = [jnp.zeros((1, 1))] * 4
    R = pF["R"] * jnp.exp(P["S"])
    F = 1.0 / (1.0 + jnp.exp(-R * x))
    S = F - 0.5
    U = u * 512.0
    T = jnp.ones((n, 1)) * (T0 / 1000.0)[None, :]
    T = T * jnp.exp(P["T"])
    G = jnp.ones((n, 1)) * G0[None, :]
    for i in range(P["G"].shape[1]):
        G = G.at[:, _J_INDEX[i]].set(G[:, _J_INDEX[i]] * jnp.exp(P["G"][:, i]))
    if "M" in P:
        G = G.at[:, _J_INDEX[0]].set(G[:, _J_INDEX[0]] * jnp.exp(-(P["M"] @ S[:, 6]) * 32.0))
    f = jnp.zeros_like(x)
    # granular (spiny stellate)
    uu = A[0] @ S[:, 2] + U
    uu = -G[:, 0] * S[:, 0] - G[:, 2] * S[:, 4] - G[:, 1] * S[:, 2] + uu
    f = f.at[:, 1].set((uu - 2.0 * x[:, 1] - x[:, 0] / T[:, 0]) / T[:, 0])
    # superficial pyramidal
    uu = -A[2] @ S[:, 6]
    uu = G[:, 7] * S[:, 0] - G[:, 6] * S[:, 2] + uu
    f = f.at[:, 3].set((uu - 2.0 * x[:, 3] - x[:, 2] / T[:, 1]) / T[:, 1])
    # inhibitory interneurons
    uu = -A[3] @ S[:, 6]
    uu = G[:, 4] * S[:, 0] + G[:, 5] * S[:, 6] - G[:, 3] * S[:, 4] + uu
    f = f.at[:, 5].set((uu - 2.0 * x[:, 5] - x[:, 4] / T[:, 2]) / T[:, 2])
    # deep pyramidal
    uu = A[1] @ S[:, 2]
    uu = -G[:, 9] * S[:, 6] - G[:, 8] * S[:, 4] + uu
    f = f.at[:, 7].set((uu - 2.0 * x[:, 7] - x[:, 6] / T[:, 3]) / T[:, 3])
    f = f.at[:, 0].set(x[:, 1]).at[:, 2].set(x[:, 3]).at[:, 4].set(x[:, 5]).at[:, 6].set(x[:, 7])
    return f


def jacobians(x, u, P, pF=CMC_DEFAULTS):
    """``(J, dfdu)`` of the vectorised flow at (x, u)."""
    n = x.shape[0]
    fv = lambda v: _vec(fx_cmc(_unvec(v, n), u, P, pF))
    fu = lambda uu: _vec(fx_cmc(x, uu, P, pF))
    return jax.jacfwd(fv)(_vec(x)), jax.jacfwd(fu)(u)


def delay_matrix(P, n, J, pF=CMC_DEFAULTS, nodelay=1):
    """SPM's delay matrix D (seconds) for an (n, 8) state, with voltage-current couplings undelayed."""
    De = jnp.exp(P["D"])
    Di = jnp.diag(jnp.diag(De))
    De = (De - Di) * pF["de"] / 1000.0
    Di = Di * pF["di"] / 1000.0
    nx = 8
    D = jnp.kron(jnp.ones((nx, nx)), De) + jnp.kron(jnp.ones((nx, nx)) - jnp.eye(nx), Di)
    if nodelay == 1:
        D = jnp.where(J == 1.0, 0.0, D)
    elif nodelay == 2:
        D = jnp.where((J == 1.0) | (J.T == 1.0), 0.0, D)
    return D


def delayed_jacobian(J, D, N=64, n_outer=64):
    """Delayed Jacobian ``Q J`` by SPM's Robbins-Monro Taylor scheme (spm_dcm_delay).

    SPM runs the Taylor series and the outer Robbins-Monro update until a
    tolerance is met (up to 2^8 terms each).  Here both loops are
    fixed-length ``lax.scan`` with a convergence mask so that reverse-mode
    differentiation is possible; once converged the updates are zero, so
    the result equals SPM's to the same tolerance provided *N* and
    *n_outer* are large enough (checked in the tests against a plain loop).
    Returns the delayed Jacobian ``QJ`` and the operator ``Q = QJ J^{-1}``.
    """
    m = J.shape[0]
    Dm = -D
    QJ0 = jnp.linalg.solve(jnp.eye(m) - Dm * J, J)
    a = 0.25
    tol = jnp.max(jnp.sum(jnp.abs(QJ0), axis=1)) * 1e-6

    def dn_body(carry, _):
        dn = carry * Dm
        return dn, dn * J
    _, Dn = jax.lax.scan(dn_body, jnp.ones_like(Dm), None, length=N)   # Dn[k] = D^(k+1) .* J

    def inner(QJ):
        def body(c, k):
            QJn, Q, done = c
            QJn = QJn @ QJ / (k + 1.0)
            dQ = Dn[k] @ QJn
            dQ = jnp.where(jnp.isnan(dQ), 0.0, dQ)
            dq = jnp.max(jnp.sum(jnp.abs(dQ), axis=1))
            Q = jnp.where(done, Q, Q + dQ)
            done = done | (dq < tol)
            return (QJn, Q, done), None
        (_, Q, _), _ = jax.lax.scan(body, (jnp.eye(m), J, jnp.array(False)), jnp.arange(N))
        return Q

    def outer(c, _):
        QJ, Q, done = c
        Qn = inner(QJ)
        QJ_new = QJ * (1.0 - a) + Qn * a
        diff = jnp.max(jnp.sum(jnp.abs(QJ_new - Qn), axis=1))
        QJ = jnp.where(done, QJ, QJ_new)
        Q = jnp.where(done, Q, Qn)
        done = done | (diff < tol)
        return (QJ, Q, done), None
    (QJ, Q, _), _ = jax.lax.scan(outer, (QJ0, J, jnp.array(False)), None, length=n_outer)
    Qop = Q @ jnp.linalg.inv(J + jnp.exp(-16.0) * jnp.eye(m))
    return Q, Qop


def _eig(A):
    """Right eigen-decomposition with derivatives enabled (distinct eigenvalues assumed)."""
    out = jax.lax.linalg.eig(A, compute_left_eigenvectors=False, compute_right_eigenvectors=True,
                             enable_eigvec_derivs=True)
    return out[0], out[1]


def transfer_function(Jd, dfdu_d, L, w, conditioning="exp"):
    """Modulation transfer function S(w, i, j) via the eigen-solution (spm_dcm_mtf), with SPM's mode conditioning."""
    # eigen-solution with eigenvector derivatives enabled (eigenvalues are distinct
    # for these models; SPM uses the same decomposition without derivatives)
    s, v = _eig(Jd)
    if conditioning == "exp":   # SPM: max(w) > 1 (any Hz grid); "min" otherwise; static for tracing
        s = 1j * s.imag + s.real - jnp.exp(s.real)
    else:
        s = 1j * s.imag + jnp.minimum(s.real, -1.0 / 32.0)
    dgdv = L.astype(v.dtype) @ v                      # (ng, nk)
    dvdu = jnp.linalg.pinv(v) @ dfdu_d.astype(v.dtype)  # (nk, nu)
    omega = 2j * jnp.pi * jnp.asarray(w, dtype=float)
    Sk = 1.0 / (omega[:, None] - s[None, :])          # (nw, nk)
    return jnp.einsum("ik,wk,kj->wij", dgdv, Sk, dvdu)


def dctmtx(N, K):
    """spm_dctmtx: N x K discrete cosine basis (first column constant)."""
    n = jnp.arange(N)
    C = jnp.zeros((N, K))
    C = C.at[:, 0].set(jnp.ones(N) / jnp.sqrt(N))
    k = jnp.arange(1, K)
    C = C.at[:, 1:].set(jnp.sqrt(2.0 / N) * jnp.cos(jnp.pi * (2 * n[:, None] + 1) * k[None, :] / (2 * N)))
    return C


def innovations(P, w):
    """``(Gu, Gs, Gn)``: neuronal innovations per source, specific and non-specific channel noise."""
    f = jnp.asarray(w, dtype=float)
    a, b, c = P["a"], P["b"], P["c"]
    ns = max(a.shape[1], P["d"].shape[1]) if "d" in P else a.shape[1]
    Gu = jnp.stack([jnp.exp(a[0, i]) * f ** (-jnp.exp(a[1, i])) for i in range(a.shape[1])], axis=1)
    if Gu.shape[1] == 1:
        Gu = Gu * jnp.ones((1, ns))
    if "d" in P:
        nd = P["d"].shape[0]
        X = dctmtx(f.shape[0], nd + 1)
        Mu = jnp.exp(X[:, 1:] @ P["d"])
        if Mu.shape[1] == 1:
            Mu = Mu * jnp.ones((1, ns))
        Gu = Gu * Mu
    Gn = jnp.exp(b[0, 0] - 2.0) * f ** (-jnp.exp(b[1, 0]))
    Gs = jnp.stack([jnp.exp(c[0, i] - 2.0) * f ** (-jnp.exp(c[1, 0])) for i in range(c.shape[1])], axis=1)
    return Gu, Gs, Gn


def gen_Q(P, X_row):
    """Condition-specific parameters (spm_gen_Q): B on every A, N on M, diag(B) on G(:, 1)."""
    Q = dict(P)
    Q.pop("B", None)
    for i, xi in enumerate(np.atleast_1d(X_row)):
        Q["A"] = [Q["A"][j] + xi * P["B"][i] for j in range(len(Q["A"]))]
        if "M" in P and "N" in P:
            Q["M"] = Q["M"] + xi * P["N"][i]
        if "G" in Q:
            Q["G"] = Q["G"].at[:, 0].add(xi * jnp.diag(P["B"][i]))
    Q.pop("N", None)
    return Q


def lead_field(P, n_sources, n_channels, U=None):
    """``dg/dx`` for an LFP model: electrode gains times state contributions, then spatial modes."""
    Lsrc = jnp.zeros((n_channels, n_sources))
    m = min(n_channels, n_sources)
    Lsrc = Lsrc.at[jnp.arange(m), jnp.arange(m)].set(jnp.asarray(P["L"]).reshape(-1)[:m])
    L = jnp.kron(jnp.asarray(P["J"]).reshape(1, -1), Lsrc)   # (nc, 8 n), column-major states
    if U is not None:
        L = U.T @ L
    return L


def cmc_csd(P, w, X, x0=None, U=None, n_channels=None, pF=CMC_DEFAULTS, conditioning="exp"):
    """Predicted cross-spectral densities, one ``(nw, nc, nc)`` complex array per row of *X*."""
    n = P["G"].shape[0]
    nc = n if n_channels is None else n_channels
    x0 = jnp.zeros((n, 8)) if x0 is None else jnp.asarray(x0, dtype=float)
    u0 = jnp.zeros(n)
    Gu, Gs, Gn = innovations(P, w)
    out = []
    rows = np.atleast_2d(np.asarray(X)) if np.size(X) else np.zeros((1, 0))
    for r in rows:
        Q = gen_Q(P, r)
        J, dfdu = jacobians(x0, u0, Q, pF)
        D = delay_matrix(Q, n, J, pF)
        Jd, Qop = delayed_jacobian(J, D)
        dfdu_d = Qop @ dfdu
        L = lead_field(Q, n, nc, U)
        S = transfer_function(Jd, dfdu_d, L, w, conditioning)   # (nw, nc, nu)
        G = jnp.einsum("wik,wk,wjk->wij", S, Gu.astype(S.dtype), jnp.conj(S))
        G = G + jnp.einsum("wi,ij->wij", Gs.astype(S.dtype), jnp.eye(nc, dtype=S.dtype))
        G = G + Gn[:, None, None]
        out.append(G)
    return out


def cmc_priors_from_oracle(o, which="pE"):
    """Parameter dict from the JSON written from SPM's fitted DCM (see docs)."""
    p = o[which]
    P = {"A": [jnp.asarray(a, dtype=float) for a in p["A"]],
         "B": [jnp.asarray(b, dtype=float) for b in p["B"]],
         "C": jnp.asarray(p["C"], dtype=float), "T": jnp.asarray(p["T"], dtype=float),
         "G": jnp.asarray(p["G"], dtype=float), "D": jnp.asarray(p["D"], dtype=float),
         "S": jnp.asarray(p["S"], dtype=float), "L": jnp.asarray(p["L"], dtype=float),
         "J": jnp.asarray(p["J"], dtype=float), "a": jnp.asarray(p["a"], dtype=float),
         "b": jnp.asarray(p["b"], dtype=float), "c": jnp.asarray(p["c"], dtype=float)}
    if "d" in p:
        P["d"] = jnp.asarray(p["d"], dtype=float)
    if "M" in p:
        P["M"] = jnp.asarray(p["M"], dtype=float)
    if "N" in p:
        P["N"] = [jnp.asarray(b, dtype=float) for b in p["N"]]
    return P


# ---------------------------------------------------------------------------
# Priors (spm_cmc_priors + spm_L_priors for LFP + spm_ssr_priors), current SPM
# ---------------------------------------------------------------------------

def cmc_priors(A, B=(), C=None, n_channels=None):
    """Prior expectation and variance dicts for a CMC CSD model with LFP observations.

    *A* is ``(forward, backward, lateral)`` binary matrices; *B* a tuple of
    binary matrices for between-trial effects; *C* the input constraint
    (``n x n_inputs``).  Mirrors ``spm_cmc_priors`` (log-scaling parameters
    with expectation 0 and variance 1/16, 1/32, 1/64 as SPM), ``spm_L_priors``
    for the LFP case (gains 1 with variance 64, ``J`` with state 3 at 1 and
    states 1, 7 free with variance 1/32) and ``spm_ssr_priors`` (``a``,
    ``b``, ``c`` with variance 1/128, four DCT coefficients ``d``, two
    filter coefficients ``f`` with variance 1/64).
    """
    A = [np.asarray(a, dtype=float) for a in A]
    n = A[0].shape[0]
    if C is None:
        C = np.ones((n, 1))
    C = np.asarray(C, dtype=float)
    if len(A) < 3:
        A.append(np.zeros_like(A[0]))
    Dm = [A[0], A[0], A[1], A[1]]
    pE, pC = {}, {}
    pE["M"] = jnp.zeros_like(jnp.asarray(A[2])); pC["M"] = jnp.asarray(A[2] != 0, dtype=float) / 32.0
    Q = np.zeros((n, n), dtype=bool)
    pE["A"], pC["A"] = [], []
    for a in Dm:
        mask = a != 0
        pE["A"].append(jnp.asarray(mask * 32.0 - 32.0)); pC["A"].append(jnp.asarray(mask / 16.0))
        Q |= mask
    Q |= np.eye(n, dtype=bool)
    pE["B"], pC["B"], pE["N"], pC["N"] = [], [], [], []
    for b in B:
        b = np.asarray(b) != 0
        pE["B"].append(jnp.zeros((n, n))); pC["B"].append(jnp.asarray((b & Q & ~(A[2] != 0)) / 8.0))
        pE["N"].append(jnp.zeros((n, n))); pC["N"].append(jnp.asarray((b & Q & (A[2] != 0)) / 8.0))
    Cm = C != 0
    pE["C"] = jnp.asarray(Cm * 32.0 - 32.0); pC["C"] = jnp.asarray(Cm / 32.0)
    pE["T"] = jnp.zeros((1, 4)); pC["T"] = jnp.full((1, 4), 1 / 32.0)   # shared across sources, as SPM
    pE["G"] = jnp.zeros((n, 4)); pC["G"] = jnp.full((n, 4), 1 / 32.0)
    pE["D"] = jnp.zeros((n, n)); pC["D"] = jnp.asarray(Q / 64.0)
    pE["S"] = jnp.zeros(()); pC["S"] = jnp.asarray(1 / 64.0)
    nu = C.shape[1]
    pE["R"] = jnp.zeros((nu, 2)); pC["R"] = jnp.full((nu, 2), 1 / 16.0)
    # LFP lead field
    nc = n if n_channels is None else n_channels
    pE["L"] = jnp.ones((1, nc)); pC["L"] = jnp.full((1, nc), 64.0)
    pE["J"] = jnp.zeros((1, 8)).at[0, 2].set(1.0); pC["J"] = jnp.zeros((1, 8)).at[0, 0].set(1 / 32.0).at[0, 6].set(1 / 32.0)
    # spectral (ssr) priors: m = number of sources (inputs), n = channels
    pE["a"] = jnp.zeros((2, n)); pC["a"] = jnp.full((2, n), 1 / 128.0)
    pE["b"] = jnp.zeros((2, 1)); pC["b"] = jnp.full((2, 1), 1 / 128.0)
    pE["c"] = jnp.zeros((2, nc)); pC["c"] = jnp.full((2, nc), 1 / 128.0)
    pE["d"] = jnp.zeros((4, n)); pC["d"] = jnp.full((4, n), 1 / 128.0)
    pE["f"] = jnp.zeros((2, 1)); pC["f"] = jnp.full((2, 1), 1 / 64.0)
    return pE, pC


def four_node_test_architecture():
    """The four-source network of SPM's CSD forward test (test 6): A, B, C, X."""
    A1 = np.zeros((4, 4)); A1[1, 0] = 1; A1[3, 2] = 1
    A2 = np.zeros((4, 4)); A2[0, 1] = 1; A2[2, 3] = 1
    A3 = np.zeros((4, 4)); A3[2, 0] = A3[0, 2] = A3[3, 1] = A3[1, 3] = 1
    B = [((A1 + A2 + A3) > 0).astype(float)]
    C = np.array([[1.0], [0.0], [1.0], [0.0]])
    X = np.array([[0.0], [1.0]])
    return [A1, A2, A3], B, C, X
