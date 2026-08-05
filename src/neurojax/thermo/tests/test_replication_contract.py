"""Red-green contract for the Berjaga-Buisan et al. (2026) mouse-arm replication.

RED   before the pipeline has been run against the real EBRAINS data (no results file).
GREEN once `python -m neurojax.thermo.replicate_ebrains` has produced results that actually
      reproduce the published effect.

The point of encoding it this way: the replication claim is not a sentence in a README that can
drift, it is an assertion that re-runs. If a preprocessing change, a solver change, or a different
lag silently kills the effect, these go red.

    python -m neurojax.thermo.replicate_ebrains --restarts 4 --steps 300
    pytest src/neurojax/thermo/tests -q

Results location can be overridden with THERMO_RESULTS.
"""
import collections
import json
import os
import pathlib

import numpy as np
import pytest

_HERE = pathlib.Path(__file__).resolve().parent
DEFAULT = _HERE.parent / "results" / "fdt_results.json"
RESULTS = pathlib.Path(os.environ.get("THERMO_RESULTS", DEFAULT))

# The paper's own numbers, for reference
PAPER_RHO = 0.885           # |Spearman| between anesthesia depth and FDT violations (their Fig 2B)
N_MICE, N_LEVELS = 8, 3


def _load():
    if not RESULTS.exists():
        pytest.fail(
            f"RED: no results at {RESULTS}. Run the replication first:\n"
            f"  python -m neurojax.thermo.replicate_ebrains --restarts 4 --steps 300")
    return json.load(open(RESULTS))


def _ranked():
    """Attach depth_rank (0=lightest .. 2=deepest) within each mouse."""
    rows = _load()
    bysub = collections.defaultdict(list)
    for r in rows:
        bysub[r["subject"]].append(r)
    for rs in bysub.values():
        for k, r in enumerate(sorted(rs, key=lambda x: x["iso"])):
            r["depth_rank"] = k
    return rows, bysub


# ---------------------------------------------------------------- data integrity
def test_all_recordings_analyzed():
    """8 mice x 3 isoflurane levels, none silently dropped."""
    rows, bysub = _ranked()
    assert len(rows) == N_MICE * N_LEVELS, f"expected 24 recordings, got {len(rows)}"
    assert len(bysub) == N_MICE
    for s, rs in bysub.items():
        assert len(rs) == N_LEVELS, f"sub-{s} has {len(rs)} levels"


def test_fits_are_finite_and_converged():
    rows, _ = _ranked()
    for r in rows:
        assert np.isfinite(r["loss"]), f"non-finite loss for sub-{r['subject']} ISO{r['iso']}"
        assert np.isfinite(r["fdt_violation"]) and r["fdt_violation"] >= 0
        assert np.isfinite(r["entropy_production"])


def test_empirical_data_is_actually_asymmetric():
    """If the empirical lagged covariance were symmetric there would be no non-equilibrium to find."""
    rows, _ = _ranked()
    assert np.mean([r["emp_fs_asym"] for r in rows]) > 0.05


# ---------------------------------------------------------------- the published claim
def test_fdt_violations_decrease_with_anesthesia_depth():
    """THE claim (paper Fig 2B): deeper anesthesia => smaller departure from equilibrium."""
    from scipy.stats import spearmanr
    rows, _ = _ranked()
    rho, p = spearmanr([r["depth_rank"] for r in rows], [r["fdt_violation"] for r in rows])
    assert rho < 0, f"expected FDT violations to DECREASE with depth, got rho={rho:+.3f}"
    assert p < 0.05, f"not significant: rho={rho:+.3f}, p={p:.4g}"


def test_group_means_are_ordered():
    rows, _ = _ranked()
    m = [np.mean([r["fdt_violation"] for r in rows if r["depth_rank"] == k]) for k in range(3)]
    assert m[0] > m[1] > m[2], f"light/mid/deep means not ordered: {[round(x, 1) for x in m]}"


def test_entropy_production_independently_agrees():
    """A second, independently computed thermodynamic quantity must point the same way."""
    from scipy.stats import spearmanr
    rows, _ = _ranked()
    rho, p = spearmanr([r["depth_rank"] for r in rows], [r["entropy_production"] for r in rows])
    assert rho < 0 and p < 0.05, f"entropy production disagrees: rho={rho:+.3f}, p={p:.4g}"


# ---------------------------------------------------------------- the gap we have NOT closed
@pytest.mark.xfail(reason="known gap: our |rho| ~0.54 vs the paper's 0.885 -- pipeline differs "
                          "(tau, downsampling, 4 restarts vs their 1000). Kept as a failing "
                          "target rather than hidden in prose.", strict=True)
def test_effect_size_matches_paper():
    from scipy.stats import spearmanr
    rows, _ = _ranked()
    rho, _p = spearmanr([r["depth_rank"] for r in rows], [r["fdt_violation"] for r in rows])
    assert abs(rho) >= 0.8, f"|rho|={abs(rho):.3f} vs paper {PAPER_RHO}"
