# © NeuroJAX developers
#
# License: BSD (3-clause)
"""Automatic bad-channel detection (OSL preprocessing parity).

Flags channels by three montage-free criteria, mirroring OSL's bad-channel step:

* **flat** — variance below ``flat_atol`` (dead / disconnected channel);
* **variance outlier** — robust z-score of log-variance above ``z_thresh``
  (median/MAD based, so a few bad channels don't mask each other);
* **low correlation** — maximum absolute correlation with any other channel
  below ``corr_thresh`` (a channel not sharing the common neural signal).

``detect_bad_channels`` returns the per-criterion index lists plus their sorted
union under ``"bad"``.
"""

import jax.numpy as jnp

__all__ = ["detect_bad_channels"]


def detect_bad_channels(data: jnp.ndarray, *, flat_atol: float = 1e-10,
                        z_thresh: float = 4.0, corr_thresh: float = 0.4) -> dict:
    """Detect bad channels in a ``(n_channels, n_samples)`` array.

    Returns
    -------
    dict with keys ``"flat"``, ``"variance"``, ``"low_correlation"`` (index
    lists) and ``"bad"`` (their sorted union).
    """
    C = data.shape[0]
    var = jnp.var(data, axis=1)

    flat = var < flat_atol

    logv = jnp.log(jnp.maximum(var, 1e-30))
    med = jnp.median(logv)
    mad = jnp.median(jnp.abs(logv - med)) + 1e-12
    robz = 0.6745 * (logv - med) / mad
    var_outlier = (jnp.abs(robz) > z_thresh) & (~flat)

    corr = jnp.nan_to_num(jnp.corrcoef(data))  # flat channels -> 0 correlation
    corr = corr - jnp.eye(C)
    max_abs_corr = jnp.max(jnp.abs(corr), axis=1)
    low_corr = (max_abs_corr < corr_thresh) & (~flat)

    def _idx(mask):
        return sorted(int(i) for i in jnp.where(mask)[0])

    flat_i, var_i, corr_i = _idx(flat), _idx(var_outlier), _idx(low_corr)
    bad = sorted(set(flat_i) | set(var_i) | set(corr_i))
    return {
        "flat": flat_i,
        "variance": var_i,
        "low_correlation": corr_i,
        "bad": bad,
    }
