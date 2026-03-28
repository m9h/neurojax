"""Sign flipping for MEG source reconstruction.

Resolves the sign ambiguity inherent in beamformer/MNE source estimates:
each parcel's timeseries has an arbitrary ±1 sign that can differ across
subjects.  Before group-level analyses (HMM, DyNeMo) this ambiguity must
be aligned so that covariance structures are comparable.

Algorithm (following osl-ephys):

1. Compute a template covariance (mean across subjects).
2. For each subject, initialise the sign vector using the leading
   eigenvectors of the covariance (comparing their element-wise signs
   to the template's eigenvectors).
3. Refine with greedy coordinate descent: flip one parcel at a time to
   maximise correlation between ``S @ C_subj @ S`` and the template.
4. Iterate: recompute the template from corrected covariances and repeat.

All operations use JAX (jax.numpy).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def apply_sign_flips(
    data: jnp.ndarray,
    signs: jnp.ndarray,
) -> jnp.ndarray:
    """Apply a sign vector to parcellated timeseries.

    Parameters
    ----------
    data : (n_parcels, n_times)
    signs : (n_parcels,) array of ±1

    Returns
    -------
    flipped : (n_parcels, n_times)
    """
    return data * signs[:, None]


def compute_template_covariance(
    data_list: List[jnp.ndarray],
) -> jnp.ndarray:
    """Compute the mean covariance matrix across subjects.

    Parameters
    ----------
    data_list : list of (n_parcels, n_times) arrays.

    Returns
    -------
    template : (n_parcels, n_parcels) — mean covariance.
    """
    covs = []
    for x in data_list:
        n_times = x.shape[1]
        covs.append(x @ x.T / n_times)
    return jnp.mean(jnp.stack(covs), axis=0)


# ---------------------------------------------------------------------------
# Covariance correlation (Frobenius-based)
# ---------------------------------------------------------------------------


def _cov_correlation(C1: jnp.ndarray, C2: jnp.ndarray) -> float:
    """Pearson correlation between the upper-triangle elements of two matrices."""
    idx = jnp.triu_indices(C1.shape[0], k=1)
    v1 = C1[idx]
    v2 = C2[idx]
    v1c = v1 - jnp.mean(v1)
    v2c = v2 - jnp.mean(v2)
    num = jnp.sum(v1c * v2c)
    denom = jnp.sqrt(jnp.sum(v1c ** 2) * jnp.sum(v2c ** 2))
    return num / jnp.maximum(denom, 1e-12)


def _apply_sign_to_cov(cov: jnp.ndarray, signs: jnp.ndarray) -> jnp.ndarray:
    """Apply sign flips to a covariance: S @ C @ S where S = diag(signs)."""
    return cov * (signs[:, None] * signs[None, :])


# ---------------------------------------------------------------------------
# Eigenvector-based sign initialisation
# ---------------------------------------------------------------------------


def _eigenvector_init(
    cov: jnp.ndarray,
    template: jnp.ndarray,
    n_evecs: int = 3,
) -> jnp.ndarray:
    """Initialise sign vector by comparing leading eigenvectors.

    For each of the ``n_evecs`` leading eigenvectors of the subject's
    covariance and the template, compute the per-parcel sign that best
    aligns them.  The final sign vector is determined by majority vote
    across eigenvectors.

    Parameters
    ----------
    cov : (P, P) subject covariance.
    template : (P, P) template covariance.
    n_evecs : int — number of leading eigenvectors to use.

    Returns
    -------
    signs : (P,) array of ±1.
    """
    n_parcels = cov.shape[0]
    n_evecs = min(n_evecs, n_parcels)

    # Eigendecomposition (eigenvalues in ascending order in JAX)
    evals_s, evecs_s = jnp.linalg.eigh(cov)
    evals_t, evecs_t = jnp.linalg.eigh(template)

    # Take the n_evecs largest eigenvectors (last columns)
    evecs_s = evecs_s[:, -n_evecs:]  # (P, n_evecs)
    evecs_t = evecs_t[:, -n_evecs:]  # (P, n_evecs)

    # For each eigenvector pair, compute alignment signs
    # The sign of evec is arbitrary, so align each subject evec to template evec
    votes = jnp.zeros(n_parcels)
    for k in range(n_evecs):
        vs = evecs_s[:, k]
        vt = evecs_t[:, k]
        # Align global sign of eigenvector
        if jnp.sum(vs * vt) < 0:
            vs = -vs
        # Per-parcel vote: +1 if signs agree, -1 if they disagree
        votes = votes + jnp.sign(vs) * jnp.sign(vt)

    # Majority vote determines the sign per parcel
    signs = jnp.where(votes >= 0, 1.0, -1.0)
    return signs


# ---------------------------------------------------------------------------
# Greedy sign-flip optimisation
# ---------------------------------------------------------------------------


def _find_signs_for_subject(
    cov: jnp.ndarray,
    template: jnp.ndarray,
    init_signs: Optional[jnp.ndarray] = None,
    max_iter: int = 500,
) -> jnp.ndarray:
    """Find optimal ±1 signs for a single subject via coordinate descent.

    Greedily flips one parcel at a time to maximise the correlation between
    the sign-corrected subject covariance and the template.

    Parameters
    ----------
    cov : (P, P) subject covariance.
    template : (P, P) template covariance.
    init_signs : (P,) optional initial sign vector. If ``None``, uses
        eigenvector-based initialisation.
    max_iter : int — maximum sweeps over all parcels.

    Returns
    -------
    signs : (P,) array of ±1.
    """
    n_parcels = cov.shape[0]

    if init_signs is not None:
        signs = init_signs
    else:
        signs = _eigenvector_init(cov, template)

    current_corr = _cov_correlation(_apply_sign_to_cov(cov, signs), template)

    for _ in range(max_iter):
        improved = False
        for p in range(n_parcels):
            # Try flipping parcel p
            trial_signs = signs.at[p].set(-signs[p])
            trial_cov = _apply_sign_to_cov(cov, trial_signs)
            trial_corr = _cov_correlation(trial_cov, template)

            if trial_corr > current_corr:
                signs = trial_signs
                current_corr = trial_corr
                improved = True

        if not improved:
            break

    return signs


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------


def find_sign_flips(
    data_list: List[jnp.ndarray],
    template: Optional[jnp.ndarray] = None,
    max_iter: int = 500,
    n_outer: int = 5,
) -> List[jnp.ndarray]:
    """Find optimal sign patterns that align subjects to a common template.

    Uses an iterative procedure: find signs per subject, recompute the
    template from sign-corrected covariances, repeat.  This is important
    when no external template is provided because the initial mean
    covariance is contaminated by the sign ambiguity.

    Parameters
    ----------
    data_list : list of (n_parcels, n_times) arrays.
        Parcellated source-space timeseries, one per subject.
    template : (n_parcels, n_parcels) optional.
        Reference covariance matrix.  If ``None``, the mean covariance
        across subjects is used (and iteratively refined).
    max_iter : int
        Maximum number of coordinate-descent iterations per subject.
    n_outer : int
        Number of outer template-refinement iterations (only used when
        ``template`` is ``None``).

    Returns
    -------
    signs_list : list of (n_parcels,) arrays of ±1.
        Optimal sign pattern for each subject.
    """
    n_subjects = len(data_list)
    n_parcels = data_list[0].shape[0]

    # Compute per-subject covariances
    covs = []
    for x in data_list:
        n_times = x.shape[1]
        covs.append(x @ x.T / n_times)

    template_provided = template is not None

    if template_provided:
        # Single pass against the provided template
        signs_list = []
        for cov in covs:
            signs = _find_signs_for_subject(cov, template, max_iter=max_iter)
            signs_list.append(signs)
        return signs_list

    # No template provided: use iterative refinement.
    # First pass: use each subject's covariance against the first subject
    # as a starting reference (the first subject is the "anchor").
    tpl = covs[0]
    signs_list = []
    for cov in covs:
        signs = _find_signs_for_subject(cov, tpl, max_iter=max_iter)
        signs_list.append(signs)

    # Iterative refinement: recompute template from corrected covs
    for _ in range(n_outer - 1):
        corrected_covs = [
            _apply_sign_to_cov(cov, signs)
            for cov, signs in zip(covs, signs_list)
        ]
        tpl = jnp.mean(jnp.stack(corrected_covs), axis=0)

        new_signs_list = []
        for cov in covs:
            signs = _find_signs_for_subject(cov, tpl, max_iter=max_iter)
            new_signs_list.append(signs)

        signs_list = new_signs_list

    return signs_list


def sign_flip_metrics(
    data_list_before: List[jnp.ndarray],
    data_list_after: List[jnp.ndarray],
) -> Dict:
    """Report improvement metrics from sign flipping.

    Parameters
    ----------
    data_list_before : list of (n_parcels, n_times) — original data.
    data_list_after : list of (n_parcels, n_times) — sign-corrected data.

    Returns
    -------
    dict with keys:
        ``"n_flips"`` : int — total number of parcels flipped across subjects.
        ``"mean_correlation_before"`` : float — mean pairwise cov correlation before.
        ``"mean_correlation_after"`` : float — mean pairwise cov correlation after.
    """
    # Count flips: where sign(mean timeseries) differs
    n_flips = 0
    for before, after in zip(data_list_before, data_list_after):
        sign_before = jnp.sign(jnp.sum(before, axis=1))
        sign_after = jnp.sign(jnp.sum(after, axis=1))
        n_flips += int(jnp.sum(sign_before != sign_after))

    def _mean_pairwise_corr(data_list):
        covs = []
        for x in data_list:
            nt = x.shape[1]
            covs.append(x @ x.T / nt)
        corrs = []
        for i in range(len(covs)):
            for j in range(i + 1, len(covs)):
                corrs.append(float(_cov_correlation(covs[i], covs[j])))
        if not corrs:
            return 0.0
        return sum(corrs) / len(corrs)

    return {
        "n_flips": n_flips,
        "mean_correlation_before": _mean_pairwise_corr(data_list_before),
        "mean_correlation_after": _mean_pairwise_corr(data_list_after),
    }
