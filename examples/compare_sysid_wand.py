"""Systems identification comparison on WAND resting-state MEG.

End-to-end pipeline:
  1. Load CTF .ds from WAND (sub-08033 task-resting)
  2. Preprocess (filter, resample)
  3. Source reconstruct + parcellate (fsaverage, aparc)
  4. Prepare data (TDE + PCA, matching osl-dynamics)
  5. Run all 5 methods (HMM, DyNeMo, SINDy, LogSig, DMD)
  6. Cross-method comparison

Prerequisites:
  - WAND data with .meg4 resolved (datalad get)
  - fsaverage (auto-downloaded by MNE)

Usage:
  python examples/compare_sysid_wand.py [--subject sub-08033] [--wand-root PATH]
"""

from __future__ import annotations

import argparse
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

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Default WAND location (in dmipy checkout)
DEFAULT_WAND_ROOT = Path.home() / "dev" / "dmipy" / "data" / "wand"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", default="sub-08033")
    parser.add_argument("--wand-root", type=Path, default=DEFAULT_WAND_ROOT)
    parser.add_argument("--parcellation", default="aparc",
                        help="FreeSurfer atlas (aparc=68, aparc.a2009s=148)")
    parser.add_argument("--n-states", type=int, default=8)
    parser.add_argument("--n-embeddings", type=int, default=15)
    parser.add_argument("--n-pca", type=int, default=80)
    parser.add_argument("--save-dir", type=Path, default=Path("data/wand_parcellated"))
    parser.add_argument("--skip-source-recon", action="store_true",
                        help="Skip source recon, load pre-saved parcellated data")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("  WAND MEG Systems Identification Comparison")
    print(f"  Subject: {args.subject}")
    print("=" * 70)

    loader = WANDMEGLoader(args.wand_root)
    print(f"\n{loader}")
    print(f"Tasks for {args.subject}: {loader.list_tasks(args.subject)}")

    # ---- Step 1: Get parcellated data ----
    parc_file = args.save_dir / f"{args.subject}_task-resting_parc.npy"

    if args.skip_source_recon and parc_file.exists():
        print(f"\nLoading pre-saved parcellated data: {parc_file}")
        parcellated = np.load(parc_file)
    else:
        if not loader.is_available(args.subject, "resting"):
            print(f"\n*** MEG data not yet available (annex pointer). ***")
            print(f"Run: cd {args.wand_root} && datalad get "
                  f"{args.subject}/ses-01/meg/{args.subject}_ses-01_task-resting.ds/")
            print("\nMeanwhile, here's what we know about this recording:")
            print(f"  Headshape points: {len(loader.load_headshape(args.subject))}")
            print(f"  Available tasks: {loader.list_tasks(args.subject)}")
            print(f"\nTo run with synthetic data instead:")
            print(f"  python examples/compare_sysid.py")
            return

        print(f"\nRunning full pipeline: load → preprocess → source recon → parcellate")
        t0 = time.time()
        parcellated = loader.run_pipeline(
            args.subject,
            task="resting",
            parcellation=args.parcellation,
            save_dir=args.save_dir,
        )
        print(f"Pipeline done in {time.time() - t0:.1f}s")

    n_samples, n_parcels = parcellated.shape
    sfreq = 250.0  # after resampling
    duration = n_samples / sfreq
    print(f"\nParcellated data: {parcellated.shape} ({duration:.0f}s at {sfreq} Hz)")

    # ---- Step 2: Prepare data (TDE + PCA) ----
    print(f"\nPreparing: TDE({args.n_embeddings}) → PCA({args.n_pca})")
    data = Data([parcellated])
    data.prepare({
        "tde_pca": {
            "n_embeddings": args.n_embeddings,
            "n_pca_components": min(args.n_pca, n_parcels * args.n_embeddings),
        },
        "standardize": {},
    })
    prepared = data.prepared_data[0]
    print(f"Prepared shape: {prepared.shape}")

    prepared_jax = jnp.array(np.array(prepared))
    T, C = prepared_jax.shape

    # ---- Step 3: Run all 5 methods ----
    results = {}

    # 3a. HMM
    print(f"\n--- [1/5] HMM (n_states={args.n_states}) ---")
    t0 = time.time()
    hmm = GaussianHMM(n_states=args.n_states, n_channels=C)
    hmm.fit([prepared_jax], n_epochs=30, n_init=3)
    results["hmm"] = {
        "gamma": np.array(hmm.infer([prepared_jax])[0]),
        "trans_prob": np.array(hmm.transition_matrix),
        "time": time.time() - t0,
    }
    print(f"  Time: {results['hmm']['time']:.1f}s")

    # 3b. DyNeMo
    print(f"\n--- [2/5] DyNeMo (n_modes={args.n_states}) ---")
    t0 = time.time()
    dynemo = DyNeMo(DyNeMoConfig(
        n_modes=args.n_states, n_channels=C,
        sequence_length=200, inference_n_units=32, model_n_units=32,
        n_epochs=20, batch_size=32, learning_rate=5e-4,
    ))
    dynemo.fit([prepared_jax])
    results["dynemo"] = {
        "alpha": np.array(dynemo.infer([prepared_jax])[0]),
        "time": time.time() - t0,
    }
    print(f"  Time: {results['dynemo']['time']:.1f}s")

    # 3c. Windowed SINDy
    print("\n--- [3/5] Windowed SINDy ---")
    window_samples = int(30 * sfreq)  # 30s windows
    stride_samples = int(5 * sfreq)   # 5s stride
    t0 = time.time()
    results["sindy"] = windowed_sindy(
        prepared_jax, window_size=window_samples, stride=stride_samples,
        n_pca=3, degree=2, threshold=0.01,
    )
    results["sindy_time"] = time.time() - t0
    print(f"  Windows: {len(results['sindy'].times)}  Time: {results['sindy_time']:.1f}s")

    # 3d. Windowed Log-Signatures
    print("\n--- [4/5] Windowed Log-Signatures ---")
    t0 = time.time()
    results["logsig"] = windowed_signatures(
        prepared_jax, window_size=window_samples, stride=stride_samples,
        n_pca=4, depth=3,
    )
    results["logsig_time"] = time.time() - t0
    print(f"  Change points: {len(results['logsig'].change_points)}  "
          f"Time: {results['logsig_time']:.1f}s")

    # 3e. Windowed DMD
    print("\n--- [5/5] Windowed DMD/Koopman ---")
    t0 = time.time()
    results["dmd"] = windowed_dmd(
        prepared_jax, window_size=window_samples, stride=stride_samples,
        rank=10, dt=1.0 / sfreq,
    )
    results["dmd_time"] = time.time() - t0
    print(f"  Time: {results['dmd_time']:.1f}s")

    # ---- Step 4: Cross-method comparison ----
    print("\n" + "=" * 70)
    print("  CROSS-METHOD RESULTS")
    print("=" * 70)

    hmm_hard = np.argmax(results["hmm"]["gamma"], axis=1)
    dynemo_hard = np.argmax(results["dynemo"]["alpha"], axis=1)

    from sklearn.metrics import normalized_mutual_info_score
    T_common = min(len(hmm_hard), len(dynemo_hard))
    nmi = normalized_mutual_info_score(hmm_hard[:T_common], dynemo_hard[:T_common])
    print(f"\nHMM ↔ DyNeMo NMI: {nmi:.3f}")

    # HMM state lifetimes
    hmm_transitions = np.where(np.diff(hmm_hard) != 0)[0]
    if len(hmm_transitions) > 1:
        mean_lifetime = np.mean(np.diff(hmm_transitions)) / sfreq
        print(f"HMM mean state lifetime: {mean_lifetime:.2f}s")

    # SINDy stability
    sindy = results["sindy"]
    stable_frac = np.mean(sindy.max_real_eig < 0)
    print(f"SINDy: {stable_frac:.0%} of windows are stable (Re(λ)<0)")

    # Signature change points vs HMM transitions
    logsig = results["logsig"]
    if len(logsig.change_points) > 0 and len(hmm_transitions) > 0:
        cp_times = logsig.times[logsig.change_points]
        min_dists = [np.min(np.abs(hmm_transitions - t)) for t in cp_times]
        print(f"Signature CPs ↔ HMM transitions: {np.mean(min_dists)/sfreq:.2f}s mean distance")

    # DMD dominant frequencies
    dmd = results["dmd"]
    dom_freq = np.abs(dmd.frequencies[:, 0])
    print(f"DMD dominant freq: {np.median(dom_freq):.1f} Hz (median)")

    # ---- Step 5: Figure ----
    time_axis = np.arange(T) / sfreq  # seconds

    fig, axes = plt.subplots(6, 1, figsize=(18, 20), sharex=True,
                              gridspec_kw={"hspace": 0.3})

    # (a) HMM states
    axes[0].plot(time_axis, hmm_hard, linewidth=0.3, color="tab:blue")
    axes[0].set_ylabel("State")
    axes[0].set_title(f"(a) HMM States ({args.n_states} states)")

    # (b) DyNeMo alpha
    alpha = results["dynemo"]["alpha"]
    for k in range(min(args.n_states, 8)):
        axes[1].plot(time_axis[:len(alpha)], alpha[:, k], linewidth=0.3, alpha=0.7)
    axes[1].set_ylabel("α")
    axes[1].set_title(f"(b) DyNeMo Mixing Coefficients")

    # (c) SINDy eigenvalue
    sindy_t = sindy.times / sfreq
    axes[2].plot(sindy_t, sindy.max_real_eig, "k-", linewidth=1)
    axes[2].axhline(0, color="red", linestyle="--", alpha=0.5)
    axes[2].set_ylabel("max Re(λ)")
    axes[2].set_title("(c) SINDy Jacobian Stability")

    # (d) Signature distances
    sig_t = logsig.times[1:] / sfreq
    axes[3].plot(sig_t, logsig.distances, "k-", linewidth=1)
    for cp in logsig.change_points:
        axes[3].axvline(logsig.times[cp + 1] / sfreq, color="green", alpha=0.5)
    axes[3].set_ylabel("‖Δ logsig‖")
    axes[3].set_title(f"(d) Log-Signature Distances ({len(logsig.change_points)} change points)")

    # (e) DMD dominant frequency
    dmd_t = dmd.times / sfreq
    axes[4].plot(dmd_t, dom_freq, "k-", linewidth=1)
    axes[4].set_ylabel("Freq (Hz)")
    axes[4].set_title("(e) DMD Dominant Frequency")

    # (f) HMM transition probability matrix
    im = axes[5].imshow(results["hmm"]["trans_prob"], cmap="Blues", vmin=0)
    axes[5].set_xlabel("To state")
    axes[5].set_ylabel("From state")
    axes[5].set_title("(f) HMM Transition Matrix")
    plt.colorbar(im, ax=axes[5], shrink=0.6)

    plt.suptitle(
        f"WAND MEG Systems Identification — {args.subject} task-resting\n"
        f"HMM↔DyNeMo NMI={nmi:.3f} | "
        f"SINDy stable={stable_frac:.0%} | "
        f"DMD dom.freq={np.median(dom_freq):.1f}Hz",
        fontsize=13,
    )

    out_fig = f"sysid_wand_{args.subject}.png"
    plt.savefig(out_fig, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved to {out_fig}")

    # Save results
    out_npz = f"data/sysid_wand_{args.subject}.npz"
    np.savez(
        out_npz,
        hmm_gamma=results["hmm"]["gamma"],
        hmm_trans=results["hmm"]["trans_prob"],
        dynemo_alpha=results["dynemo"]["alpha"],
        sindy_times=sindy.times,
        sindy_max_eig=sindy.max_real_eig,
        logsig_distances=logsig.distances,
        dmd_frequencies=dmd.frequencies,
    )
    print(f"Results saved to {out_npz}")


if __name__ == "__main__":
    main()
