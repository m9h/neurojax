"""Systems identification comparison on WAND resting-state MEG.

Redesigned pipeline with full neurojax analysis stack:

  1. Load CTF .ds from WAND (sub-08033 task-resting)
  2. Preprocess (filter, resample)
  3. Source reconstruct + parcellate (with sensitivity analysis over choices)
  4. Prepare data: standard TDE+PCA *and* TuckerTDE (tensor alternative)
  5. Run all 5 sysid methods (HMM, DyNeMo, SINDy, LogSig, DMD)
  6. Post-hoc analysis:
     - State spectral decomposition (power maps + coherence per state)
     - Summary statistics (occupancy, lifetime, switching rate)
     - Static baseline ("is dynamics even needed?")
     - Functional networks (MI, graph measures per state)
  7. Source imaging sensitivity: compare results across different
     parcellation/spacing/method choices

Usage:
  python examples/compare_sysid_wand.py --subject sub-08033 --wand-root /data/raw/wand
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from neurojax.data import Data
from neurojax.io.wand_meg import WANDMEGLoader
from neurojax.models.hmm import GaussianHMM
from neurojax.models.dynemo import DyNeMo, DyNeMoConfig
from neurojax.dynamics.windowed import (
    windowed_sindy,
    windowed_dmd,
    windowed_signatures,
)
from neurojax.analysis.state_spectra import InvertiblePCA, get_state_spectra
from neurojax.analysis.summary_stats import (
    fractional_occupancy,
    mean_lifetime,
    switching_rate,
)
from neurojax.analysis.static import static_summary
from neurojax.analysis.multitaper import multitaper_psd
from neurojax.analysis.tensor_tde import TuckerTDE

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_WAND_ROOT = Path("/data/raw/wand")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subject", default="sub-08033")
    p.add_argument("--wand-root", type=Path, default=DEFAULT_WAND_ROOT)
    p.add_argument("--n-states", type=int, default=8)
    p.add_argument("--max-duration", type=float, default=120.0,
                   help="Max seconds to source-reconstruct (memory limit)")
    p.add_argument("--save-dir", type=Path, default=Path("data/wand_parcellated"))
    p.add_argument("--skip-source-recon", action="store_true",
                   help="Load pre-saved parcellated data")
    p.add_argument("--sensitivity", action="store_true",
                   help="Run source imaging sensitivity analysis")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Source imaging sensitivity analysis
# ---------------------------------------------------------------------------

SOURCE_CONFIGS = {
    "baseline": {"parcellation": "aparc", "spacing": "oct5", "method": "sLORETA"},
    "fine_src":  {"parcellation": "aparc", "spacing": "oct6", "method": "sLORETA"},
    "dspm":      {"parcellation": "aparc", "spacing": "oct5", "method": "dSPM"},
    "mne":       {"parcellation": "aparc", "spacing": "oct5", "method": "MNE"},
    "destrieux": {"parcellation": "aparc.a2009s", "spacing": "oct5", "method": "sLORETA"},
}


def run_source_sensitivity(loader, subject, args):
    """Compare HMM results across different source imaging choices."""
    print("\n" + "=" * 70)
    print("  SOURCE IMAGING SENSITIVITY ANALYSIS")
    print("=" * 70)

    results = {}
    for name, cfg in SOURCE_CONFIGS.items():
        print(f"\n--- Config: {name} ({cfg}) ---")
        parc_file = args.save_dir / f"{subject}_{name}_parc.npy"

        if parc_file.exists():
            parcellated = np.load(parc_file)
            logger.info("Loaded cached %s", parc_file)
        else:
            try:
                raw = loader.load_resting(subject)
                raw = loader.preprocess(raw)
                parcellated = loader.source_reconstruct(
                    raw,
                    parcellation=cfg["parcellation"],
                    spacing=cfg["spacing"],
                    method=cfg["method"],
                    max_duration=args.max_duration,
                )
                args.save_dir.mkdir(parents=True, exist_ok=True)
                np.save(parc_file, parcellated)
            except Exception as e:
                print(f"  FAILED: {e}")
                continue

        n_parcels = parcellated.shape[1]
        n_pca = min(80, n_parcels * 15)

        data = Data([parcellated])
        data.prepare({
            "tde_pca": {"n_embeddings": 15, "n_pca_components": n_pca},
            "standardize": {},
        })
        prepared = jnp.array(np.array(data.prepared_data[0]))

        hmm = GaussianHMM(n_states=args.n_states, n_channels=prepared.shape[1])
        hmm.fit([prepared], n_epochs=20, n_init=3)
        gamma = np.array(hmm.infer([prepared])[0])

        fo = np.array(fractional_occupancy(jnp.array(gamma)))
        lt = mean_lifetime(jnp.array(gamma), fs=250.0)
        sr = switching_rate(jnp.array(gamma), fs=250.0)
        trans_diag = np.diag(np.array(hmm.transition_matrix))

        results[name] = {
            "n_parcels": n_parcels,
            "fo": fo,
            "mean_lifetime_ms": float(np.mean(lt) * 1000),
            "switching_rate_hz": float(sr),
            "trans_diag_mean": float(trans_diag.mean()),
            "final_ll": hmm.history[-1] if hmm.history else None,
        }
        print(f"  {n_parcels} parcels | lifetime={results[name]['mean_lifetime_ms']:.0f}ms | "
              f"switch={sr:.2f}Hz | trans_diag={trans_diag.mean():.3f}")

    # Report
    print("\n--- Sensitivity Summary ---")
    print(f"{'Config':<15} {'Parcels':>8} {'Lifetime':>10} {'Switch':>8} {'Trans diag':>10}")
    print("-" * 55)
    for name, r in results.items():
        print(f"{name:<15} {r['n_parcels']:>8d} {r['mean_lifetime_ms']:>9.0f}ms "
              f"{r['switching_rate_hz']:>7.2f}Hz {r['trans_diag_mean']:>10.3f}")

    return results


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    print("=" * 70)
    print("  WAND MEG Systems Identification Comparison")
    print(f"  Subject: {args.subject}")
    print("=" * 70)

    loader = WANDMEGLoader(args.wand_root)
    print(f"\n{loader}")
    print(f"Tasks: {loader.list_tasks(args.subject)}")

    # ---- Step 1: Parcellated data ----
    parc_file = args.save_dir / f"{args.subject}_task-resting_parc.npy"

    if args.skip_source_recon and parc_file.exists():
        print(f"\nLoading pre-saved: {parc_file}")
        parcellated = np.load(parc_file)
    else:
        if not loader.is_available(args.subject, "resting"):
            print(f"\n*** MEG data not yet available (annex pointer). ***")
            print(f"Run: cd {args.wand_root} && datalad get "
                  f"{args.subject}/ses-01/meg/{args.subject}_ses-01_task-resting.ds/")
            return

        print(f"\nRunning pipeline: load → preprocess → source recon → parcellate")
        t0 = time.time()
        parcellated = loader.run_pipeline(
            args.subject, task="resting", save_dir=args.save_dir,
        )
        print(f"Pipeline: {time.time() - t0:.1f}s")

    sfreq = 250.0

    # Crop parcellated data if max_duration is set (saves memory for HMM)
    max_samples = int(args.max_duration * sfreq)
    if parcellated.shape[0] > max_samples:
        logger.info("Cropping parcellated data from %d to %d samples (%.0fs)",
                     parcellated.shape[0], max_samples, args.max_duration)
        parcellated = parcellated[:max_samples]

    n_samples, n_parcels = parcellated.shape
    duration = n_samples / sfreq
    print(f"\nParcellated: {parcellated.shape} ({duration:.0f}s at {sfreq}Hz, {n_parcels} parcels)")

    # ---- Step 2a: Standard TDE+PCA preparation ----
    n_embeddings = 15
    n_pca = min(80, n_parcels * n_embeddings)
    print(f"\nPreparing: TDE({n_embeddings}) → PCA({n_pca}) → standardize")

    data = Data([parcellated])
    data.prepare({
        "tde_pca": {"n_embeddings": n_embeddings, "n_pca_components": n_pca},
        "standardize": {},
    })
    prepared = jnp.array(np.array(data.prepared_data[0]))
    T, C = prepared.shape
    print(f"Prepared: {prepared.shape}")

    # ---- Step 2b: TuckerTDE alternative ----
    print(f"\nAlso preparing: TuckerTDE (tensor decomposition)")
    try:
        tucker = TuckerTDE(n_embeddings=n_embeddings, rank_channels=10, rank_lags=8)
        prepared_tucker = tucker.fit_transform(jnp.array(parcellated))
        print(f"TuckerTDE: {prepared_tucker.shape}")
        has_tucker = True
    except Exception as e:
        print(f"TuckerTDE skipped: {e}")
        has_tucker = False

    # ---- Step 3: Static baseline ("is dynamics needed?") ----
    print("\n--- Static Baseline ---")
    static = static_summary([jnp.array(parcellated)], fs=sfreq)
    print(f"  Static power shape: {static['psd'].shape}")
    print(f"  Static connectivity shape: {static['connectivity'].shape}")

    # ---- Step 4: Run 5 sysid methods ----
    results = {}

    # 4a. HMM
    print(f"\n--- [1/5] HMM (n_states={args.n_states}) ---")
    t0 = time.time()
    hmm = GaussianHMM(n_states=args.n_states, n_channels=C)
    hmm.fit([prepared], n_epochs=30, n_init=3)
    hmm_gamma = np.array(hmm.infer([prepared])[0])
    results["hmm"] = {"gamma": hmm_gamma, "time": time.time() - t0}
    print(f"  Time: {results['hmm']['time']:.1f}s")

    # 4b. DyNeMo
    print(f"\n--- [2/5] DyNeMo (n_modes={args.n_states}) ---")
    t0 = time.time()
    dynemo = DyNeMo(DyNeMoConfig(
        n_modes=args.n_states, n_channels=C,
        sequence_length=200, inference_n_units=32, model_n_units=32,
        n_epochs=20, batch_size=32, learning_rate=5e-4,
    ))
    dynemo.fit([prepared])
    dynemo_alpha = np.array(dynemo.infer([prepared])[0])
    results["dynemo"] = {"alpha": dynemo_alpha, "time": time.time() - t0}
    print(f"  Time: {results['dynemo']['time']:.1f}s")

    # 4c-e. Windowed methods
    window_samples = int(30 * sfreq)
    stride_samples = int(5 * sfreq)

    print("\n--- [3/5] Windowed SINDy ---")
    t0 = time.time()
    sindy_result = windowed_sindy(
        prepared, window_size=window_samples, stride=stride_samples,
        n_pca=3, degree=2, threshold=0.01,
    )
    results["sindy"] = {"result": sindy_result, "time": time.time() - t0}
    print(f"  {len(sindy_result.times)} windows | Time: {results['sindy']['time']:.1f}s")

    print("\n--- [4/5] Windowed Log-Signatures ---")
    t0 = time.time()
    sig_result = windowed_signatures(
        prepared, window_size=window_samples, stride=stride_samples,
        n_pca=4, depth=3,
    )
    results["logsig"] = {"result": sig_result, "time": time.time() - t0}
    print(f"  {len(sig_result.change_points)} change points | Time: {results['logsig']['time']:.1f}s")

    print("\n--- [5/5] Windowed DMD/Koopman ---")
    t0 = time.time()
    dmd_result = windowed_dmd(
        prepared, window_size=window_samples, stride=stride_samples,
        rank=10, dt=1.0 / sfreq,
    )
    results["dmd"] = {"result": dmd_result, "time": time.time() - t0}
    print(f"  Time: {results['dmd']['time']:.1f}s")

    # ---- Step 5: Post-hoc analysis ----
    print("\n" + "=" * 70)
    print("  POST-HOC ANALYSIS")
    print("=" * 70)

    # 5a. Summary statistics (need hard states for lifetime)
    from neurojax.analysis.summary_stats import state_time_courses
    hmm_hard = np.array(state_time_courses(jnp.array(hmm_gamma)))
    dynemo_hard = np.array(state_time_courses(jnp.array(dynemo_alpha)))

    print("\n--- HMM Summary Statistics ---")
    hmm_fo = np.array(fractional_occupancy(jnp.array(hmm_gamma)))
    hmm_lt = mean_lifetime(jnp.array(hmm_hard), fs=sfreq)
    hmm_sr = switching_rate(jnp.array(hmm_hard), fs=sfreq)
    print(f"  Fractional occupancy: {np.round(hmm_fo, 3)}")
    print(f"  Mean lifetime: {np.mean(hmm_lt)*1000:.0f} ms")
    print(f"  Switching rate: {hmm_sr:.2f} Hz")

    print("\n--- DyNeMo Summary Statistics ---")
    dynemo_fo = np.array(fractional_occupancy(jnp.array(dynemo_alpha)))
    dynemo_sr = switching_rate(jnp.array(dynemo_hard), fs=sfreq)
    print(f"  Fractional occupancy: {np.round(dynemo_fo, 3)}")
    print(f"  Switching rate: {dynemo_sr:.2f} Hz")

    # 5b. State spectral decomposition
    print("\n--- State Spectral Decomposition ---")
    try:
        # Need InvertiblePCA for the round-trip
        ipca = InvertiblePCA(n_components=n_pca)
        # Re-prepare with invertible PCA to get the projection matrix
        from neurojax.data.loading import prepare_tde
        tde_data = np.array(prepare_tde(jnp.array(parcellated), n_embeddings=n_embeddings))
        pca_data = ipca.fit_transform(jnp.array(tde_data))

        hmm_covs = np.array(hmm.state_covariances)
        spectra = get_state_spectra(
            jnp.array(hmm_covs), ipca, n_parcels, n_embeddings, fs=sfreq,
        )
        print(f"  State PSD shape: {spectra.psd.shape}")
        print(f"  State coherence shape: {spectra.coherence.shape}")
        print(f"  Frequency axis: {float(spectra.freqs[0]):.1f} - {float(spectra.freqs[-1]):.1f} Hz")
        has_spectra = True
    except Exception as e:
        print(f"  State spectra failed: {e}")
        has_spectra = False

    # ---- Step 6: Cross-method comparison ----
    print("\n" + "=" * 70)
    print("  CROSS-METHOD COMPARISON")
    print("=" * 70)

    # hmm_hard and dynemo_hard already computed above in summary stats

    from sklearn.metrics import normalized_mutual_info_score
    T_common = min(len(hmm_hard), len(dynemo_hard))
    nmi = normalized_mutual_info_score(hmm_hard[:T_common], dynemo_hard[:T_common])
    print(f"\nHMM ↔ DyNeMo NMI: {nmi:.3f}")

    hmm_transitions = np.where(np.diff(hmm_hard) != 0)[0]
    if len(hmm_transitions) > 1:
        print(f"HMM transitions: {len(hmm_transitions)} ({len(hmm_transitions)/duration:.1f}/s)")

    sindy = results["sindy"]["result"]
    stable_frac = np.mean(sindy.max_real_eig < 0)
    print(f"SINDy stable windows: {stable_frac:.0%}")

    logsig = results["logsig"]["result"]
    if len(logsig.change_points) > 0 and len(hmm_transitions) > 0:
        cp_times = logsig.times[logsig.change_points]
        min_dists = [np.min(np.abs(hmm_transitions - t)) for t in cp_times]
        print(f"Signature CPs ↔ HMM transitions: {np.mean(min_dists)/sfreq:.2f}s mean distance")

    dmd = results["dmd"]["result"]
    dom_freq = np.abs(dmd.frequencies[:, 0])
    print(f"DMD dominant freq: {np.median(dom_freq):.1f} Hz (median)")

    # ---- Step 7: Figure ----
    n_rows = 7 if has_spectra else 6
    fig, axes = plt.subplots(n_rows, 1, figsize=(18, n_rows * 3),
                              gridspec_kw={"hspace": 0.35})

    time_axis = np.arange(T) / sfreq

    # (a) HMM states + occupancy
    axes[0].plot(time_axis, hmm_hard, linewidth=0.3, color="tab:blue")
    occ_str = " | ".join([f"S{k}:{hmm_fo[k]:.0%}" for k in range(min(args.n_states, 4))])
    axes[0].set_ylabel("State")
    axes[0].set_title(f"(a) HMM States — lifetime={np.mean(hmm_lt)*1000:.0f}ms, switch={hmm_sr:.1f}Hz | {occ_str}...")

    # (b) DyNeMo alpha
    for k in range(min(args.n_states, 8)):
        axes[1].plot(time_axis[:len(dynemo_alpha)], dynemo_alpha[:, k], linewidth=0.3, alpha=0.7)
    axes[1].set_ylabel("α")
    axes[1].set_title(f"(b) DyNeMo Mixing — switch={dynemo_sr:.1f}Hz")

    # (c) SINDy eigenvalue
    sindy_t = sindy.times / sfreq
    axes[2].plot(sindy_t, sindy.max_real_eig, "k-", linewidth=1)
    axes[2].axhline(0, color="red", linestyle="--", alpha=0.5)
    axes[2].set_ylabel("max Re(λ)")
    axes[2].set_title(f"(c) SINDy Jacobian — {stable_frac:.0%} stable")

    # (d) Signature distances
    sig_t = logsig.times[1:] / sfreq
    axes[3].plot(sig_t, logsig.distances, "k-", linewidth=1)
    for cp in logsig.change_points:
        axes[3].axvline(logsig.times[cp + 1] / sfreq, color="green", alpha=0.5)
    axes[3].set_ylabel("‖Δ logsig‖")
    axes[3].set_title(f"(d) Log-Signature — {len(logsig.change_points)} change points")

    # (e) DMD dominant frequency
    dmd_t = dmd.times / sfreq
    axes[4].plot(dmd_t, dom_freq, "k-", linewidth=1)
    axes[4].set_ylabel("Freq (Hz)")
    axes[4].set_title("(e) DMD Dominant Frequency")

    # (f) HMM transition matrix + fractional occupancy
    ax_trans = axes[5]
    im = ax_trans.imshow(np.array(hmm.transition_matrix), cmap="Blues", vmin=0)
    ax_trans.set_title("(f) HMM Transition Matrix")
    plt.colorbar(im, ax=ax_trans, shrink=0.6)

    # (g) State spectra (if available)
    if has_spectra:
        ax_spec = axes[6]
        freqs_np = np.array(spectra.freqs)
        psd_np = np.array(spectra.psd)
        for k in range(min(args.n_states, 8)):
            # Mean over parcels for each state
            ax_spec.semilogy(freqs_np, psd_np[k].mean(axis=-1), linewidth=1, label=f"S{k}")
        ax_spec.set_xlabel("Frequency (Hz)")
        ax_spec.set_ylabel("PSD")
        ax_spec.set_title("(g) State Power Spectral Density (mean over parcels)")
        ax_spec.legend(ncol=4, fontsize=8)
        ax_spec.set_xlim(1, 45)

    axes[-1].set_xlabel("Time (s)")

    plt.suptitle(
        f"WAND MEG — {args.subject} task-resting | {n_parcels} parcels, {duration:.0f}s\n"
        f"HMM↔DyNeMo NMI={nmi:.3f} | "
        f"SINDy {stable_frac:.0%} stable | "
        f"DMD {np.median(dom_freq):.1f}Hz",
        fontsize=13,
    )

    out_fig = f"sysid_wand_{args.subject}.png"
    plt.savefig(out_fig, dpi=150, bbox_inches="tight")
    print(f"\nFigure: {out_fig}")

    # Save results
    out_npz = f"data/sysid_wand_{args.subject}.npz"
    save_dict = {
        "hmm_gamma": hmm_gamma,
        "hmm_trans": np.array(hmm.transition_matrix),
        "hmm_fo": hmm_fo,
        "dynemo_alpha": dynemo_alpha,
        "sindy_times": sindy.times,
        "sindy_max_eig": sindy.max_real_eig,
        "logsig_distances": logsig.distances,
        "dmd_frequencies": dmd.frequencies,
        "parcellated_shape": parcellated.shape,
    }
    if has_spectra:
        save_dict["state_psd"] = np.array(spectra.psd)
        save_dict["state_freqs"] = np.array(spectra.freqs)
    np.savez(out_npz, **save_dict)
    print(f"Results: {out_npz}")

    # ---- Optional: Source imaging sensitivity ----
    if args.sensitivity:
        run_source_sensitivity(loader, args.subject, args)


if __name__ == "__main__":
    main()
