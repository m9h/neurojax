#!/usr/bin/env python
"""Reconcile the n=26 individual-anatomy directed-PDC/DTF cohort map with the harmonic
circulation (Leg B). WAND_DYNAMICS_COMPARISON.md flags this as the natural next step: the
cohort pipeline (real FreeSurfer anatomy, group-mean alpha-band PDC over 41 common Desikan-68
parcels, `run_wand_cohort.py` -> ~/wand_cohort_connectivity.npz) has never been mapped onto the
harmonic basis or checked against the ~0.1 Hz solenoidal circulation the template-coreg /
connectome-harmonics analysis found rotating among low-order geometric (H1,H2,H7) or structural
(H13,H8,H15) modes.

Method: express the real-anatomy group directed-PDC matrix (41x41, driver->receiver) in the SAME
harmonic coordinates as the circulation analysis. Build the 68-node harmonic basis (geometric or
structural, same recipe as wand_connectome_harmonics.py), restrict the eigenvector rows to the 41
parcels present in the cohort map (by name, via desikan68_names.txt), and transform the directed
matrix as a bilinear form: M = Phi41.T @ P @ Phi41. The antisymmetric part of M is the directed/
rotational structure in harmonic coordinates -- the PDC analogue of the Langevin circulation's
A_sol. Report its low-order concentration (same centre-of-mass statistic as the circulation
analysis) and top rotational harmonic pairs, and test significance with a parcel-identity
permutation null (shuffle which of the 41 cohort parcels maps to which harmonic-basis row, so
the null controls for "low-order harmonics are just smoother" and asks whether THESE specific
anatomical regions' harmonic loadings are non-trivially aligned with THEIR PDC structure).

    WAND_OUT=/data/datasets/wand_src PYTHONPATH=src \\
      .venv-models/bin/python scripts/real_data/wand_reconcile_pdc_harmonics.py
"""
import os

import numpy as np
import jax.numpy as jnp

from neurojax.spatial import connectome_harmonics

OUT = os.environ.get("WAND_OUT", "/data/datasets/wand_src")
COHORT_NPZ = os.path.expanduser("~/wand_cohort_connectivity.npz")
N_HARM = 20
N_PERM = 2000
SEED = 0


def gaussian_graph(centroids):
    d2 = np.sum((centroids[:, None, :] - centroids[None, :, :]) ** 2, axis=-1)
    sigma2 = np.median(d2[d2 > 0])
    W = np.exp(-d2 / sigma2)
    np.fill_diagonal(W, 0.0)
    return W


def harmonic_basis(graph):
    if graph == "structural":
        SC = np.load(os.path.join(OUT, "desikan68_SC.npy"))
        W = np.log1p(SC)
    else:
        cent = np.load(os.path.join(OUT, "desikan68_centroids.npy"))
        W = gaussian_graph(cent)
    evals, Phi = connectome_harmonics(jnp.asarray(W), normalized=False)
    evals, Phi = np.asarray(evals), np.asarray(Phi)
    dc = 1                                                     # mode 0 = DC/constant
    return evals[dc:dc + N_HARM], Phi[:, dc:dc + N_HARM]        # (N_HARM,), (68, N_HARM)


def centre_of_mass(au):
    """Same statistic as wand_connectome_harmonics.py: energy-weighted mean harmonic order
    (1-indexed) of the antisymmetric matrix's off-diagonal upper triangle. Low = low-order."""
    n = au.shape[0]
    idx = (np.arange(n)[:, None] + np.arange(n)[None, :] + 2) / 2.0
    return float(np.sum(au * idx) / (au.sum() + 1e-30))


def top_pairs(au, evals, k=5):
    pairs = np.dstack(np.unravel_index(np.argsort(au.ravel())[::-1], au.shape))[0][:k]
    out = []
    for i, j in pairs:
        out.append((int(i) + 1, int(j) + 1, float(au[i, j]), float(evals[i]), float(evals[j])))
    return out


def reconcile(graph, P, idx41, rng):
    evals, Phi = harmonic_basis(graph)
    Phi41 = Phi[idx41]                                          # (41, N_HARM)
    M = Phi41.T @ P @ Phi41                                      # (N_HARM, N_HARM), directed
    au = np.abs(np.triu(0.5 * (M - M.T), 1))
    com = centre_of_mass(au)
    pairs = top_pairs(au, evals)

    # permutation null: shuffle which cohort parcel maps to which harmonic-basis row, so the
    # null preserves each basis's smoothness/energy structure and only scrambles the specific
    # anatomical correspondence between PDC-active regions and harmonic loadings.
    null_com = np.empty(N_PERM)
    n = len(idx41)
    for p in range(N_PERM):
        perm = rng.permutation(n)
        Phi41_p = Phi41[perm]
        Mp = Phi41_p.T @ P @ Phi41_p
        aup = np.abs(np.triu(0.5 * (Mp - Mp.T), 1))
        null_com[p] = centre_of_mass(aup)
    z = (com - null_com.mean()) / (null_com.std() + 1e-12)
    p_val = (np.sum(null_com <= com) + 1) / (N_PERM + 1)         # one-sided: observed <= null?
    return dict(com=com, null_mean=float(null_com.mean()), null_std=float(null_com.std()),
                z=float(z), p=float(p_val), pairs=pairs, evals=evals, Phi41=Phi41)


def loading_enrichment(P, Phi41, low_idx, rng, n_perm=N_PERM):
    """Complementary, more targeted test: do the anatomically PDC-ACTIVE parcels (top total
    directed degree) load more heavily on the SPECIFIC low-order harmonics the circulation
    analysis flagged (`low_idx`, 0-indexed into the N_HARM kept modes) than an average parcel
    does? Tests loading concentration directly, independent of whether the antisymmetric
    rotational structure lines up (the reconcile() test above) -- a weaker requirement, so
    potentially more sensitive to a real but partial correspondence."""
    deg = np.abs(P).sum(0) + np.abs(P).sum(1)                  # total directed degree per parcel
    k = max(3, len(deg) // 4)                                   # top quartile of active parcels
    top = np.argsort(deg)[::-1][:k]
    low_load = np.abs(Phi41[:, low_idx]).mean(1)                # per-parcel mean |loading| on flagged modes
    obs = float(low_load[top].mean() / (low_load.mean() + 1e-30))  # enrichment ratio vs all parcels

    null = np.empty(n_perm)
    n = len(deg)
    for p in range(n_perm):
        perm_top = rng.choice(n, size=k, replace=False)
        null[p] = low_load[perm_top].mean() / (low_load.mean() + 1e-30)
    z = (obs - null.mean()) / (null.std() + 1e-12)
    pval = (np.sum(null >= obs) + 1) / (n_perm + 1)
    return dict(obs=obs, null_mean=float(null.mean()), null_std=float(null.std()),
               z=float(z), p=float(pval), top_parcels=top)


def main():
    print(f"Reconciling n=26 cohort directed-PDC with harmonic circulation "
          f"(N_HARM={N_HARM}, {N_PERM} permutations)\n")

    names68 = [l.strip() for l in open(os.path.join(OUT, "desikan68_names.txt"))]
    d = np.load(COHORT_NPZ, allow_pickle=True)
    pdc, freqs, names41 = d["pdc"], d["freqs"], list(d["parcels"])
    print(f"cohort map: n={int(d['n'])} subjects, {len(names41)}/{len(names68)} common parcels")

    alpha = (freqs >= 8) & (freqs <= 12)
    P = pdc[alpha].mean(0).astype(np.float64)
    np.fill_diagonal(P, 0.0)

    missing = [nm for nm in names41 if nm not in names68]
    if missing:
        raise ValueError(f"cohort parcel(s) not in the 68-node harmonic order: {missing}")
    idx41 = np.array([names68.index(nm) for nm in names41])

    # the SPECIFIC low-order harmonics the template-coreg circulation flagged (0-indexed into
    # the N_HARM kept modes: geometric H1,H2,H7 -> 0,1,6; structural H13,H8,H15 -> 12,7,14)
    flagged = {"geometric": [0, 1, 6], "structural": [12, 7, 14]}

    rng = np.random.default_rng(SEED)
    for graph in ("geometric", "structural"):
        r = reconcile(graph, P, idx41, rng)
        print(f"\n=== {graph} harmonic basis ===")
        print(f"circulation-centre-of-mass of the PDC-derived directed structure: "
              f"{r['com']:.2f} (of {N_HARM})")
        print(f"  permutation null: {r['null_mean']:.2f} +/- {r['null_std']:.2f}  "
              f"z={r['z']:.2f}  p={r['p']:.4f}  (lower COM = more low-order-concentrated)")
        print("  top rotational harmonic pairs (from the PDC directed structure):")
        for hi, hj, mag, li, lj in r["pairs"]:
            print(f"    H{hi:>2d} <-> H{hj:>2d}   |M_anti|={mag:.4f}  (lambda={li:.3g},{lj:.3g})")

        e = loading_enrichment(P, r["Phi41"], flagged[graph], rng)
        flagged_str = ",".join(f"H{i+1}" for i in flagged[graph])
        print(f"  loading enrichment on the circulation's flagged harmonics ({flagged_str}): "
              f"top-PDC-active parcels load {e['obs']:.2f}x an average parcel's loading "
              f"(null {e['null_mean']:.2f}+/-{e['null_std']:.2f}, z={e['z']:.2f}, p={e['p']:.4f})")
        print(f"    top-PDC-active parcels: {[names41[i] for i in e['top_parcels']]}")

    print("\nReference (from docs/WAND_DYNAMICS_COMPARISON.md, template-coreg n=10, harmonic "
          "circulation): geometric top pairs H2<->H7, H1<->H12, H1<->H8, COM~9.3/20; "
          "structural COM~10.6/20, top harmonics H13/H8/H15.")


if __name__ == "__main__":
    main()
