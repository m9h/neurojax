"""Spectral DCM (DCM for cross-spectral density) in JAX.

The generative model follows Moran et al. (2009) / Friston et al. (2012,
``spm_dcm_csd``): a network of Jansen-Rit neural masses is linearised
around its fixed point, the resulting transfer function maps 1/f neuronal
innovations to a source cross-spectral density, a leadfield maps that to
sensors, and channel noise (specific + common) is added.  All parameters
are log-scaled around their defaults (``value = default * exp(theta)``)
with Gaussian priors on ``theta``; the posterior is found by variational
Laplace (Gauss-Newton EM with a single noise-precision hyperparameter,
``spm_nlsi_GN``) and the free energy ``F`` is returned for model
comparison.

Everything is written with ``jax.numpy`` so ``jax.grad`` flows from the
sensor CSD back to every parameter, and the Jacobian used by variational
Laplace is forward-mode autodiff rather than finite differences.

Pharmacological handles
-----------------------
``tau_i`` (inhibitory time constant, GABA-A decay: propofol lengthens it),
``Hi`` (inhibitory gain), ``He``/``tau_e`` (AMPA/NMDA-like excitatory
kinetics; ketamine), ``A_forward`` (between-region gains) and ``C``
(intrinsic connectivity).  A canonical-microcircuit or NMDA-specific
variant slots in by providing another ``vector_field``/``output`` pair.

References
----------
Jansen BH, Rit VG (1995) Biol Cybern 73:357-366.
Moran RJ et al. (2009) NeuroImage 44:796-811.
Friston KJ et al. (2012) NeuroImage 59:1222-1231 (DCM for CSD).
Friston K et al. (2007) NeuroImage 34:220-234 (variational Laplace).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

# ---------------------------------------------------------------------------
# Linear systems -> cross-spectral density
# ---------------------------------------------------------------------------


def transfer_function(A: jnp.ndarray, B: jnp.ndarray, C: jnp.ndarray, omega: jnp.ndarray) -> jnp.ndarray:
    """H(ω) = C (iωI − A)⁻¹ B for every ω.  Returns ``(n_omega, n_out, n_in)`` complex."""
    n = A.shape[0]
    eye = jnp.eye(n, dtype=A.dtype)

    def one(w):
        return C @ jnp.linalg.solve(1j * w * eye - A, B.astype(jnp.complex128))

    return jax.vmap(one)(omega)


# ---------------------------------------------------------------------------
# Column models
# ---------------------------------------------------------------------------
#
# A column model is a frozen dataclass with ``n_regions``, ``n_states`` (per
# region), ``default_theta()`` (a NamedTuple of arrays), ``vector_field(theta,
# x, u)``, ``output(x)`` and ``param_spec()`` -> [(field, index, prior_var)]
# listing the log-scaled free parameters.  ``fixed_point`` / ``linearise``
# below work for any of them.


class JRTheta(NamedTuple):
    """Jansen-Rit parameters (mV, s).  Arrays are per region ``(n,)``."""

    He: jnp.ndarray  # excitatory synaptic gain (mV)
    Hi: jnp.ndarray  # inhibitory synaptic gain (mV)
    tau_e: jnp.ndarray  # excitatory time constant (s)
    tau_i: jnp.ndarray  # inhibitory time constant (s)
    C: jnp.ndarray  # intrinsic connectivity scale
    A_forward: jnp.ndarray  # between-region gains (n, n), row = target
    p0: float = 500.0  # mean afferent input (1/s): upper stable branch, ~10 Hz inhibition-governed resonance (JR 1995's 120-320 is the limit cycle)
    e0: float = 2.5  # half max firing rate (1/s)
    v0: float = 6.0  # sigmoid threshold (mV)
    r: float = 0.56  # sigmoid slope (1/mV)


def _sigmoid(v, theta: JRTheta):
    return 2.0 * theta.e0 / (1.0 + jnp.exp(theta.r * (theta.v0 - v)))


@dataclass(frozen=True)
class JansenRitSpec:
    """Network of Jansen-Rit columns (6 states per region).

    ``A_forward[i, j] > 0`` declares a connection from region ``j`` into the
    excitatory interneurons of region ``i``; its value is the default gain
    (log-scaled by the corresponding free parameter).
    """

    n_regions: int = 1
    A_forward: jnp.ndarray | None = None
    He: float = 3.25
    Hi: float = 22.0
    tau_e: float = 0.010
    tau_i: float = 0.020
    C: float = 135.0
    n_states: int = 6

    def default_theta(self) -> JRTheta:
        n = self.n_regions
        A = jnp.zeros((n, n)) if self.A_forward is None else jnp.asarray(self.A_forward, dtype=jnp.float64)
        return JRTheta(
            He=jnp.full((n,), self.He),
            Hi=jnp.full((n,), self.Hi),
            tau_e=jnp.full((n,), self.tau_e),
            tau_i=jnp.full((n,), self.tau_i),
            C=jnp.full((n,), self.C),
            A_forward=A,
        )

    def param_spec(self) -> list[tuple[str, tuple[int, ...], float]]:
        n = self.n_regions
        out = [(name, (i,), 1 / 16) for name in ("He", "Hi", "tau_e", "tau_i", "C") for i in range(n)]
        A = np.asarray(self.default_theta().A_forward)
        out += [("A_forward", (int(i), int(j)), 1 / 8) for i, j in zip(*np.nonzero(A))]
        return out

    def vector_field(self, theta: JRTheta, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        """``x`` is ``(6 n,)`` region-major ``[y0, y1, y2, dy0, dy1, dy2]``; ``u`` ``(n,)`` drives the excitatory population."""
        n = self.n_regions
        X = x.reshape(n, 6)
        y0, y1, y2, d0, d1, d2 = (X[:, k] for k in range(6))
        a, b = 1.0 / theta.tau_e, 1.0 / theta.tau_i
        C1, C2, C3, C4 = theta.C, 0.8 * theta.C, 0.25 * theta.C, 0.25 * theta.C
        v = y1 - y2
        S_v = _sigmoid(v, theta)
        coupling = theta.A_forward @ S_v
        dd0 = theta.He * a * S_v - 2 * a * d0 - a**2 * y0
        dd1 = theta.He * a * (theta.p0 + u + C2 * _sigmoid(C1 * y0, theta) + coupling) - 2 * a * d1 - a**2 * y1
        dd2 = theta.Hi * b * C4 * _sigmoid(C3 * y0, theta) - 2 * b * d2 - b**2 * y2
        return jnp.stack([d0, d1, d2, dd0, dd1, dd2], axis=1).reshape(-1)

    def output(self, x: jnp.ndarray) -> jnp.ndarray:
        """Pyramidal membrane potential ``y1 − y2`` per region."""
        X = x.reshape(self.n_regions, 6)
        return X[:, 1] - X[:, 2]


class CMCTheta(NamedTuple):
    """Canonical-microcircuit parameters (mV, s).  Populations: 0 spiny stellate, 1 superficial pyramidal, 2 inhibitory interneurons, 3 deep pyramidal."""

    T: jnp.ndarray  # (n, 4) synaptic time constants (s)
    G: jnp.ndarray  # (n, 10) intrinsic gains, see CMCSpec
    A_forward: jnp.ndarray  # (n, n) sp(source) -> ss and dp (target)
    A_backward: jnp.ndarray  # (n, n) dp(source) -> sp and ii (target)
    R: float = 2.0 / 3.0  # sigmoid slope


@dataclass(frozen=True)
class CMCSpec:
    """Network of canonical microcircuits (Bastos et al. 2012), after ``spm_fx_cmc``.

    Four populations per region, each with a voltage and a current (8 states):
    spiny stellate (ss, granular; receives the exogenous input and forward
    afferents), superficial pyramidal (sp; sends forward connections and is
    the main source of the measured signal), inhibitory interneurons (ii) and
    deep pyramidal (dp; sends backward connections).  Second-order kinetics
    ``dv/dt = i``, ``di/dt = (input − 2 i − v / T) / T`` with sigmoid firing
    ``S(v) = 1 / (1 + exp(−R v)) − 1/2``.  The ten intrinsic gains ``G``
    (defaults ``[4 4 8 4 4 2 4 4 2 1] × 200``) are:

    ====  ============  ====
    idx   connection    sign
    ====  ============  ====
    0     ss → ss       −
    1     sp → ss       −
    2     ii → ss       −
    3     ii → ii       −
    4     ss → ii       +
    5     dp → ii       +
    6     sp → sp       −
    7     ss → sp       +
    8     ii → dp       −
    9     dp → dp       −
    ====  ============  ====

    Pharmacology: ``T[:, 2]`` is the inhibitory (GABA-A) time constant,
    ``G[:, 2]``, ``G[:, 3]``, ``G[:, 8]`` the inhibitory gains, ``T[:, 0]`` /
    ``T[:, 1]`` the glutamatergic kinetics, ``A_*`` the extrinsic gains.  The
    fixed point is at ``x = 0`` (the sigmoid is centred), so the linearisation
    has gain ``R/4`` on every connection.  Parity with SPM25's
    ``spm_fx_cmc`` / ``spm_gx_erp`` at the prior was verified with
    SPM-Python (2026-10-25): identical Jacobian eigenvalues and transfer
    function (``tests/test_csd_dcm.py::TestCMC::test_matches_spm_linearisation``).
    Output: superficial-pyramidal depolarisation.
    """

    n_regions: int = 1
    A_forward: jnp.ndarray | None = None
    A_backward: jnp.ndarray | None = None
    T: tuple[float, float, float, float] = (0.002, 0.002, 0.016, 0.028)
    G: tuple[float, ...] = tuple(g * 200.0 for g in (4, 4, 8, 4, 4, 2, 4, 4, 2, 1))
    u_gain: float = 512.0  # exogenous input gain into ss, as in spm_fx_cmc (B = 512 / T_ss on the ss current)
    n_states: int = 8
    population = ("ss", "sp", "ii", "dp")

    def _mat(self, A):
        n = self.n_regions
        return jnp.zeros((n, n)) if A is None else jnp.asarray(A, dtype=jnp.float64)

    def default_theta(self) -> CMCTheta:
        n = self.n_regions
        return CMCTheta(
            T=jnp.tile(jnp.asarray(self.T), (n, 1)),
            G=jnp.tile(jnp.asarray(self.G), (n, 1)),
            A_forward=self._mat(self.A_forward),
            A_backward=self._mat(self.A_backward),
        )

    def param_spec(self) -> list[tuple[str, tuple[int, ...], float]]:
        n = self.n_regions
        out = [("T", (i, k), 1 / 16) for i in range(n) for k in range(4)]
        out += [("G", (i, k), 1 / 16) for i in range(n) for k in range(10)]
        th = self.default_theta()
        for name in ("A_forward", "A_backward"):
            A = np.asarray(getattr(th, name))
            out += [(name, (int(i), int(j)), 1 / 8) for i, j in zip(*np.nonzero(A))]
        return out

    def vector_field(self, theta: CMCTheta, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        """``x`` is ``(8 n,)`` region-major ``[v_ss, v_sp, v_ii, v_dp, i_ss, i_sp, i_ii, i_dp]``; ``u`` ``(n,)`` drives ss."""
        n = self.n_regions
        X = x.reshape(n, 8)
        v, i = X[:, :4], X[:, 4:]
        S = 1.0 / (1.0 + jnp.exp(-theta.R * v)) - 0.5  # (n, 4)
        G, T = theta.G, theta.T
        fwd = theta.A_forward @ S[:, 1]  # sp of sources
        bwd = theta.A_backward @ S[:, 3]  # dp of sources
        inp = jnp.stack(
            [
                fwd - G[:, 0] * S[:, 0] - G[:, 1] * S[:, 1] - G[:, 2] * S[:, 2] + self.u_gain * u,
                G[:, 7] * S[:, 0] - G[:, 6] * S[:, 1] + bwd,
                G[:, 4] * S[:, 0] + G[:, 5] * S[:, 3] - G[:, 3] * S[:, 2] + bwd,
                fwd - G[:, 8] * S[:, 2] - G[:, 9] * S[:, 3],
            ],
            axis=1,
        )
        di = (inp - 2.0 * i - v / T) / T
        return jnp.concatenate([i, di], axis=1).reshape(-1)

    def output(self, x: jnp.ndarray) -> jnp.ndarray:
        """Superficial-pyramidal depolarisation per region."""
        return x.reshape(self.n_regions, 8)[:, 1]


def fixed_point(spec, theta, n_newton: int = 40) -> jnp.ndarray:
    """Fixed point of the deterministic network (``u = 0``) by Newton iteration.

    The iteration is unrolled under ``lax.fori_loop`` so gradients flow
    through it; at convergence they agree with the implicit derivative.
    """
    n = spec.n_regions
    u0 = jnp.zeros(n)
    f = lambda x: spec.vector_field(theta, x, u0)  # noqa: E731
    jac = jax.jacfwd(f)

    def step(_, x):
        return x - jnp.linalg.solve(jac(x), f(x))

    return jax.lax.fori_loop(0, n_newton, step, jnp.zeros(spec.n_states * n))


def linearise(spec, theta):
    """Linear state-space ``(A, B, C)`` of the network at its fixed point."""
    x0 = fixed_point(spec, theta)
    u0 = jnp.zeros(spec.n_regions)
    A = jax.jacfwd(lambda x: spec.vector_field(theta, x, u0))(x0)
    B = jax.jacfwd(lambda u: spec.vector_field(theta, x0, u))(u0)
    C = jax.jacfwd(spec.output)(x0)
    return A, B, C


# Jansen-Rit aliases
def jr_vector_field(spec, theta, x, u):
    return spec.vector_field(theta, x, u)


def jr_output(spec, x):
    return spec.output(x)


jr_fixed_point = fixed_point
jr_linearise = linearise


# ---------------------------------------------------------------------------
# Variational Laplace
# ---------------------------------------------------------------------------


class VLResult(NamedTuple):
    Ep: jnp.ndarray  # posterior mean of parameters
    Cp: jnp.ndarray  # posterior covariance
    h: jnp.ndarray  # log noise precision (scalar)
    F: jnp.ndarray  # variational free energy (log-evidence bound)
    F_trace: np.ndarray  # F after each accepted iteration
    prediction: jnp.ndarray  # f(Ep)


def _logdet(M):
    sign, ld = jnp.linalg.slogdet(M)
    return ld


def variational_laplace(
    f: Callable[[jnp.ndarray], jnp.ndarray],
    y: jnp.ndarray,
    pE: jnp.ndarray,
    pC: jnp.ndarray,
    n_iter: int = 64,
    hE: float = 0.0,
    hC: float = 16.0,
    tol: float = 1e-3,
    p0: jnp.ndarray | None = None,
) -> VLResult:
    """Gauss-Newton variational Laplace (after ``spm_nlsi_GN``).

    Model ``y = f(p) + ε``, ``ε ~ N(0, exp(−h) I)``, ``p ~ N(pE, pC)``,
    ``h ~ N(hE, hC)``; ``p0`` is the starting point (default ``pE``).  Alternates a Newton update of ``h`` (holding the
    posterior over ``p`` fixed) with a trust-region Gauss-Newton update of
    ``p``; a step is kept only if the free energy ``F`` rises.  Returns the
    Laplace posterior and ``F``.
    """
    y = jnp.asarray(y, dtype=jnp.float64)
    pE = jnp.asarray(pE, dtype=jnp.float64)
    pC = jnp.asarray(pC, dtype=jnp.float64)
    n = y.shape[0]
    ipC = jnp.linalg.inv(pC)
    f_jit = jax.jit(f)
    jac = jax.jit(jax.jacfwd(f))

    def free_energy(p, h, e, J):
        iS = jnp.exp(h)
        Cp = jnp.linalg.inv(J.T @ J * iS + ipC)
        ep = p - pE
        L1 = 0.5 * n * h - 0.5 * iS * (e @ e) - 0.5 * n * jnp.log(2 * jnp.pi)
        L2 = 0.5 * _logdet(ipC) - 0.5 * ep @ ipC @ ep + 0.5 * _logdet(Cp)
        Ch = 1.0 / (0.5 * iS * (e @ e + jnp.trace(J @ Cp @ J.T)) + 1.0 / hC)
        L3 = -0.5 * jnp.log(hC) - 0.5 * (h - hE) ** 2 / hC + 0.5 * jnp.log(Ch)
        return L1 + L2 + L3, Cp

    def update_h(h, e, J, Cp):
        # Newton steps on the concave F(h) with the parameter posterior fixed,
        # started at the maximum-likelihood value log(n/q) so the first step
        # is small (from h=0 the step is ~hC*n/2 and exp(h) overflows).
        q = e @ e + jnp.trace(J @ Cp @ J.T)
        h = jnp.log(n / q)
        for _ in range(8):
            iS = jnp.exp(h)
            dF = 0.5 * (n - iS * q) - (h - hE) / hC
            ddF = -0.5 * iS * q - 1.0 / hC
            h = h - dF / ddF
        return h

    p = pE if p0 is None else jnp.asarray(p0, dtype=jnp.float64)
    h = jnp.asarray(hE, dtype=jnp.float64)
    e = y - f_jit(p)
    J = jac(p)
    for _ in range(4):
        _, Cp = free_energy(p, h, e, J)
        h = update_h(h, e, J, Cp)
    F, Cp = free_energy(p, h, e, J)
    trace = [float(F)]
    v = 1.0  # trust-region scale on the Gauss-Newton step
    for _ in range(n_iter):
        iS = jnp.exp(h)
        g = J.T @ e * iS - ipC @ (p - pE)
        dp = Cp @ g
        accepted = False
        for _ in range(8):
            p_new = p + v * dp
            e_new = y - f_jit(p_new)
            if not bool(jnp.all(jnp.isfinite(e_new))):
                v *= 0.5
                continue
            J_new = jac(p_new)
            _, Cp_new = free_energy(p_new, h, e_new, J_new)
            h_new = update_h(h, e_new, J_new, Cp_new)
            F_new, Cp_new = free_energy(p_new, h_new, e_new, J_new)
            if bool(F_new >= F - 1e-12):
                accepted = True
                break
            v *= 0.5
        if not accepted:
            break
        dF = float(F_new - F)
        p, e, J, h, Cp, F = p_new, e_new, J_new, h_new, Cp_new, F_new
        trace.append(float(F))
        v = min(2.0 * v, 1.0)
        if 0 <= dF < tol:
            break
    return VLResult(Ep=p, Cp=Cp, h=h, F=F, F_trace=np.asarray(trace), prediction=f_jit(p))


def model_posterior(F: jnp.ndarray | np.ndarray) -> np.ndarray:
    """Posterior model probabilities from free energies (flat model prior)."""
    F = np.asarray(F, dtype=float)
    w = np.exp(F - F.max())
    return w / w.sum()


# ---------------------------------------------------------------------------
# Spectral DCM
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Param:
    name: str
    index: tuple[int, ...]
    prior_var: float


@dataclass(frozen=True)
class SpectralDCM:
    """Jansen-Rit DCM for cross-spectral density.

    Parameters (log-scalings except ``L``, which is additive; prior mean 0):

    ``leadfield`` is ``(n_sensors, n_regions)`` for fixed dipoles or
    ``(n_sensors, n_regions, 3)`` for free dipole moments, in which case
    ``orientation`` ``(n_regions, 3)`` is the prior-mean moment (e.g. the
    cortical normal) and ``moment_var`` the prior variance of each component.

    ===========  ======================  ==========  ===========
    name         shape                   prior var   default
    ===========  ======================  ==========  ===========
    column       spec.param_spec()       1/16, 1/8   Jansen-Rit: He, Hi, tau_e, tau_i, C, A_forward; CMC: T, G, A_forward, A_backward
    L            per region x 3          moment_var  dipole moment minus ``orientation`` (free-moment leadfield only)
    a_u          per region              1/4         innovation amplitude
    b_u          scalar                  1/16        innovation 1/f exponent (1)
    a_n          scalar                  1/4         channel-specific noise amplitude
    a_c          scalar                  1/4         common channel noise amplitude
    b_n          scalar                  1/16        channel-noise 1/f exponent (1)
    ===========  ======================  ==========  ===========

    Channel-noise defaults are ``noise_fraction`` of the mean sensor power
    predicted by the neural part at the prior (reported by ``repr``).  Data
    are rescaled on ``invert`` so that their mean diagonal power equals the
    prior prediction (the scale is returned in the result's ``scale``).
    """

    spec: JansenRitSpec | CMCSpec
    leadfield: jnp.ndarray
    freqs: jnp.ndarray
    noise_fraction: float = 0.01
    fixed: tuple[str, ...] = ()
    orientation: jnp.ndarray | None = None
    moment_var: float = 1.0
    params: tuple[_Param, ...] = field(init=False)
    _noise_default: float = field(init=False)

    def __post_init__(self):
        n = self.spec.n_regions
        L = jnp.asarray(self.leadfield, dtype=jnp.float64)
        if L.shape[1] != n:
            raise ValueError(f"leadfield has {L.shape[1]} sources, spec has {n} regions")
        if L.ndim == 3:
            if L.shape[2] != 3 or self.orientation is None:
                raise ValueError("a free-moment leadfield is (n_sensors, n_regions, 3) and needs `orientation` (n_regions, 3)")
            object.__setattr__(self, "orientation", jnp.asarray(self.orientation, dtype=jnp.float64).reshape(n, 3))
        elif L.ndim != 2:
            raise ValueError("leadfield must be (n_sensors, n_regions) or (n_sensors, n_regions, 3)")
        object.__setattr__(self, "leadfield", L)
        object.__setattr__(self, "freqs", jnp.asarray(self.freqs, dtype=jnp.float64))
        params = [_Param(name, idx, var) for name, idx, var in self.spec.param_spec()]
        if L.ndim == 3:
            # dipole moment of region i = orientation[i] + L[i, :] (additive, so the
            # moment can rotate and flip; cf. spm_L_priors ECD moments, prior mean 0 /
            # variance 64 there — a zero-moment prior mean is a stationary point of
            # our log-power features, hence the given orientation as prior mean)
            params += [_Param("L", (i, k), self.moment_var) for i in range(n) for k in range(3)]
        params += [_Param("a_u", (i,), 1 / 16) for i in range(n)]
        # spectral nuisance priors are tight (as in spm_csd_mtf, 1/128) so the
        # 1/f terms cannot absorb what the neural-mass parameters should explain
        params += [_Param("b_u", (), 1 / 128), _Param("a_n", (), 1 / 128), _Param("a_c", (), 1 / 128), _Param("b_n", (), 1 / 128)]
        object.__setattr__(self, "params", tuple(params))
        object.__setattr__(self, "_noise_default", 0.0)
        neural = self._sensor_csd(jnp.zeros(len(params)), include_noise=False)
        mean_power = float(jnp.mean(jnp.real(jnp.diagonal(neural, axis1=1, axis2=2))))
        object.__setattr__(self, "_noise_default", self.noise_fraction * mean_power)

    # -- bookkeeping --------------------------------------------------------

    @property
    def n_params(self) -> int:
        return len(self.params)

    @property
    def names(self) -> list[str]:
        return [p.name + ("" if not p.index else str(list(p.index))) for p in self.params]

    def index(self, name: str, *idx: int) -> int:
        for k, p in enumerate(self.params):
            if p.name == name and (not idx or p.index == tuple(idx)) and (idx or len(p.index) <= 1):
                return k
        raise KeyError((name, idx))

    @property
    def prior_mean(self) -> jnp.ndarray:
        return jnp.zeros(self.n_params)

    @property
    def prior_cov(self) -> jnp.ndarray:
        var = jnp.array([1e-8 if p.name in self.fixed else p.prior_var for p in self.params])
        return jnp.diag(var)

    def with_fixed(self, *names: str) -> "SpectralDCM":
        """Reduced model: the named parameters are pinned at their prior (variance 1e-8)."""
        return replace(self, fixed=tuple(self.fixed) + names)

    def __repr__(self):
        return (
            f"SpectralDCM(n_regions={self.spec.n_regions}, n_sensors={self.leadfield.shape[0]}, "
            f"n_freqs={self.freqs.shape[0]}, n_params={self.n_params}, fixed={self.fixed}, "
            f"channel_noise_default={self._noise_default:.3g})"
        )

    # -- forward model ------------------------------------------------------

    def theta(self, p: jnp.ndarray):
        """Map the log-scaling vector onto the column model's parameters."""
        th = self.spec.default_theta()
        fields = {}
        for k, prm in enumerate(self.params):
            if prm.name in th._fields:
                cur = fields.get(prm.name, getattr(th, prm.name))
                fields[prm.name] = cur.at[prm.index].multiply(jnp.exp(p[k]))
        return th._replace(**fields)

    @property
    def dynamic_names(self) -> tuple[str, ...]:
        return tuple(self.spec.default_theta()._fields)

    def _spectral(self, p: jnp.ndarray, name: str) -> jnp.ndarray:
        ks = [k for k, prm in enumerate(self.params) if prm.name == name]
        return jnp.exp(p[jnp.array(ks)])

    def moments(self, p: jnp.ndarray) -> jnp.ndarray | None:
        """Dipole moments ``(n_regions, 3)`` for a free-moment leadfield (``None`` for a fixed one)."""
        if self.leadfield.ndim != 3:
            return None
        ks = [k for k, prm in enumerate(self.params) if prm.name == "L"]
        return self.orientation + p[jnp.array(ks)].reshape(self.spec.n_regions, 3)

    def effective_leadfield(self, p: jnp.ndarray) -> jnp.ndarray:
        """``(n_sensors, n_regions)`` leadfield at parameters ``p``."""
        if self.leadfield.ndim == 2:
            return self.leadfield
        return jnp.einsum("cik,ik->ci", self.leadfield, self.moments(p))

    def _sensor_csd(self, p: jnp.ndarray, include_noise: bool = True) -> jnp.ndarray:
        th = self.theta(p)
        A, B, C = linearise(self.spec, th)
        H = transfer_function(A, B, C, 2 * jnp.pi * self.freqs)  # (nf, n, n)
        f = self.freqs[:, None]
        b_u = self._spectral(p, "b_u")[0]
        G_u = self._spectral(p, "a_u")[None, :] * f ** (-b_u)  # (nf, n)
        S_src = jnp.einsum("fij,fj,fkj->fik", H, G_u.astype(jnp.complex128), jnp.conj(H))
        L = self.effective_leadfield(p)
        S = jnp.einsum("ci,fij,dj->fcd", L, S_src, L)
        if include_noise:
            nc = L.shape[0]
            b_n = self._spectral(p, "b_n")[0]
            shape = (f ** (-b_n))[:, :, None]  # (nf,1,1)
            a_n = self._spectral(p, "a_n")[0] * self._noise_default
            a_c = self._spectral(p, "a_c")[0] * self._noise_default
            S = S + shape * (a_n * jnp.eye(nc) + a_c * jnp.ones((nc, nc)))
        return S

    def predict(self, p: jnp.ndarray) -> jnp.ndarray:
        """Sensor cross-spectral density ``(n_freq, n_sensors, n_sensors)``."""
        return self._sensor_csd(p)

    # -- data features -------------------------------------------------------

    def features(self, csd: jnp.ndarray) -> jnp.ndarray:
        """Log power on the diagonal and complex coherency off it.

        Log power keeps a shifted spectral peak informative (a plain CSD
        residual is flat once the predicted and observed peaks stop
        overlapping, which traps Gauss-Newton), and coherency is
        dimensionless and O(1).  Layout: ``(n_blocks, n_freq)`` flattened,
        frequency last, so ``_smooth`` can act along frequency.
        """
        nc = csd.shape[1]
        diag = jnp.real(jnp.diagonal(csd, axis1=1, axis2=2))  # (nf, nc)
        isu, jsu = np.triu_indices(nc, k=1)
        coh = csd[:, isu, jsu] / jnp.sqrt(diag[:, isu] * diag[:, jsu])
        blocks = jnp.concatenate([jnp.log(diag), jnp.real(coh), jnp.imag(coh)], axis=1)  # (nf, n_blocks)
        return blocks.T.reshape(-1)

    def _smoother(self, sigma_hz: float) -> jnp.ndarray:
        """Row-normalised Gaussian smoothing matrix along frequency (identity for sigma 0)."""
        f = self.freqs
        if sigma_hz <= 0:
            return jnp.eye(f.shape[0])
        K = jnp.exp(-0.5 * ((f[:, None] - f[None, :]) / sigma_hz) ** 2)
        return K / K.sum(axis=1, keepdims=True)

    def _smooth(self, feat: jnp.ndarray, K: jnp.ndarray) -> jnp.ndarray:
        return (feat.reshape(-1, self.freqs.shape[0]) @ K.T).reshape(-1)

    def invert(
        self,
        csd: jnp.ndarray,
        n_iter: int = 64,
        smooth_schedule: tuple[float, ...] = (4.0, 0.0),
        n_starts: int = 4,
        seed: int = 0,
        **kw,
    ) -> "SpectralDCMResult":
        """Variational-Laplace inversion of an observed sensor CSD.

        The free energy of a neural-mass CSD model is multimodal in the
        synaptic parameters (e.g. a lower inhibitory gain with a shorter
        inhibitory time constant mimics a longer one), so the inversion is
        run from ``n_starts`` points — the prior mean plus ``n_starts - 1``
        draws from the prior over the neural-mass parameters and dipole
        moments (seeded) — and
        the start with the highest ``F`` is returned (``F_starts`` lists
        all).  ``smooth_schedule`` is a coarse-to-fine continuation within
        each start: features (model and data alike) are Gaussian-smoothed
        along frequency with each ``sigma`` in Hz in turn, each stage
        starting from the previous posterior mean.  ``F`` and the posterior
        come from the last stage of the best start.
        """
        csd = jnp.asarray(csd, dtype=jnp.complex128)
        if csd.shape != (self.freqs.shape[0], self.leadfield.shape[0], self.leadfield.shape[0]):
            raise ValueError(f"csd shape {csd.shape} does not match (n_freq, n_sensors, n_sensors)")
        prior_power = float(jnp.mean(jnp.real(jnp.diagonal(self.predict(self.prior_mean), axis1=1, axis2=2))))
        data_power = float(jnp.mean(jnp.real(jnp.diagonal(csd, axis1=1, axis2=2))))
        scale = prior_power / data_power
        y_full = self.features(csd * scale)
        # randomise the neural-mass parameters and (free) dipole moments: the log-power
        # features have a barrier wherever a sensor projection L·m changes sign, so a
        # moment of the wrong sign pattern is a local optimum of its own
        dyn = jnp.array([p.name in self.dynamic_names or p.name == "L" for p in self.params])
        sd = jnp.sqrt(jnp.diag(self.prior_cov)) * dyn
        draws = jax.random.normal(jax.random.PRNGKey(seed), (max(n_starts - 1, 0), self.n_params)) * sd
        starts = [self.prior_mean] + [self.prior_mean + d for d in draws]
        best, F_starts = None, []
        for p in starts:
            for sigma in smooth_schedule:
                K = self._smoother(sigma)
                res = variational_laplace(
                    lambda q, K=K: self._smooth(self.features(self.predict(q)), K),
                    self._smooth(y_full, K),
                    self.prior_mean,
                    self.prior_cov,
                    n_iter=n_iter,
                    p0=p,
                    **kw,
                )
                p = res.Ep
            F_starts.append(float(res.F))
            if best is None or res.F > best.F:
                best = res
        res = best
        return SpectralDCMResult(
            res.Ep, res.Cp, res.h, res.F, res.F_trace, self.predict(res.Ep) / scale, scale, self, np.asarray(F_starts)
        )


class SpectralDCMResult(NamedTuple):
    Ep: jnp.ndarray
    Cp: jnp.ndarray
    h: jnp.ndarray
    F: jnp.ndarray
    F_trace: np.ndarray
    prediction: jnp.ndarray  # predicted CSD in data units
    scale: float  # data were multiplied by this before fitting
    model: SpectralDCM
    F_starts: np.ndarray  # free energy reached from each start

    def summary(self) -> dict[str, tuple[float, float]]:
        """``{name: (posterior mean, posterior sd)}`` of the log-scalings."""
        sd = np.sqrt(np.diag(np.asarray(self.Cp)))
        return {n: (float(m), float(s)) for n, m, s in zip(self.model.names, np.asarray(self.Ep), sd)}


__all__ = [
    "CMCSpec",
    "CMCTheta",
    "JRTheta",
    "JansenRitSpec",
    "SpectralDCM",
    "SpectralDCMResult",
    "VLResult",
    "fixed_point",
    "jr_fixed_point",
    "jr_linearise",
    "jr_output",
    "jr_vector_field",
    "linearise",
    "model_posterior",
    "transfer_function",
    "variational_laplace",
]
