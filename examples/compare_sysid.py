"""Systems identification comparison: 5 methods on the same data.

Runs HMM, DyNeMo, windowed SINDy, windowed log-signatures, and
windowed DMD/Koopman on the same synthetic dataset and compares what
each method discovers about the underlying dynamics.

Prerequisites:
    # Generate synthetic data (uses osl-dynamics oracle container):
    docker run --gpus all --ipc=host -v $(pwd)/data:/data \
        neurojax/oracle-osl python /scripts/generate_synthetic.py

Usage:
    python examples/compare_sysid.py
"""

import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment

from neurojax.models.hmm import GaussianHMM
from neurojax.models.dynemo import DyNeMo, DyNeMoConfig
from neurojax.dynamics.windowed import (
    windowed_sindy,
    windowed_dmd,
    windowed_signatures,
)


def hungarian_accuracy(true_hard, pred_hard, n_states):
    T = min(len(true_hard), len(pred_hard))
    cost = np.zeros((n_states, n_states))
    for i in range(n_states):
        for j in range(n_states):
            cost[i, j] = -np.sum((true_hard[:T] == i) & (pred_hard[:T] == j))
    _, col_ind = linear_sum_assignment(cost)
    inv = {v: k for k, v in enumerate(col_ind)}
    remapped = np.array([inv[s] for s in pred_hard[:T]])
    return np.mean(remapped == true_hard[:T])


def main():
    print("=" * 70)
    print("  Systems Identification Comparison")
    print("  HMM | DyNeMo | SINDy | Log-Signature | Koopman/DMD")
    print("=" * 70)

    # ---- Load data ----
    ts = np.load("data/osl_sim_timeseries.npy")
    true_states_oh = np.load("data/osl_sim_states_true.npy")
    true_hard = np.argmax(true_states_oh, axis=1)
    n_states = true_states_oh.shape[1]
    n_channels = ts.shape[1]
    T = ts.shape[0]

    print(f"\nData: {ts.shape}  |  {n_states} states, {n_channels} channels")

    data_jax = [jnp.array(ts)]

    # ---- Method 1: HMM ----
    print("\n--- [1/5] HMM (neurojax) ---")
    t0 = time.time()
    hmm = GaussianHMM(n_states=n_states, n_channels=n_channels)
    hmm.fit(data_jax, n_epochs=30, n_init=3)
    hmm_time = time.time() - t0
    hmm_gamma = np.array(hmm.infer(data_jax)[0])
    hmm_hard = np.argmax(hmm_gamma, axis=1)
    hmm_acc = hungarian_accuracy(true_hard, hmm_hard, n_states)
    print(f"  Accuracy: {hmm_acc:.3f}  |  Time: {hmm_time:.1f}s")

    # Compute HMM state transition times
    hmm_transitions = np.where(np.diff(hmm_hard) != 0)[0]

    # ---- Method 2: DyNeMo ----
    print("\n--- [2/5] DyNeMo (neurojax) ---")
    t0 = time.time()
    dynemo_config = DyNeMoConfig(
        n_modes=n_states,
        n_channels=n_channels,
        sequence_length=200,
        inference_n_units=32,
        model_n_units=32,
        n_epochs=15,
        batch_size=32,
        learning_rate=5e-4,
        do_kl_annealing=True,
        kl_annealing_n_epochs=5,
    )
    dynemo = DyNeMo(dynemo_config)
    dynemo.fit(data_jax)
    dynemo_time = time.time() - t0
    dynemo_alpha = np.array(dynemo.infer(data_jax)[0])
    dynemo_hard = np.argmax(dynemo_alpha, axis=1)
    dynemo_acc = hungarian_accuracy(true_hard, dynemo_hard, n_states)
    print(f"  Accuracy: {dynemo_acc:.3f}  |  Time: {dynemo_time:.1f}s")

    # DyNeMo peak transitions (where dominant mode changes)
    dynemo_transitions = np.where(np.diff(dynemo_hard) != 0)[0]

    # ---- Method 3: Windowed SINDy ----
    print("\n--- [3/5] Windowed SINDy ---")
    t0 = time.time()
    sindy_result = windowed_sindy(
        jnp.array(ts),
        window_size=2000,
        stride=500,
        n_pca=3,
        degree=2,
        threshold=0.01,
    )
    sindy_time = time.time() - t0
    print(f"  Windows: {len(sindy_result.times)}  |  Time: {sindy_time:.1f}s")
    print(f"  Max Re(eig) range: [{sindy_result.max_real_eig.min():.3f}, "
          f"{sindy_result.max_real_eig.max():.3f}]")

    # Detect sign changes in max eigenvalue (bifurcation points)
    eig_sign = np.sign(sindy_result.max_real_eig)
    sindy_sign_changes = np.where(np.diff(eig_sign) != 0)[0]

    # ---- Method 4: Windowed Log-Signatures ----
    print("\n--- [4/5] Windowed Log-Signatures ---")
    t0 = time.time()
    sig_result = windowed_signatures(
        jnp.array(ts),
        window_size=2000,
        stride=500,
        n_pca=4,
        depth=3,
        change_threshold=2.0,
    )
    sig_time = time.time() - t0
    print(f"  Windows: {len(sig_result.times)}  |  Sig dim: {sig_result.signatures.shape[1]}")
    print(f"  Change points detected: {len(sig_result.change_points)}")
    print(f"  Time: {sig_time:.1f}s")

    # ---- Method 5: Windowed DMD/Koopman ----
    print("\n--- [5/5] Windowed DMD/Koopman ---")
    t0 = time.time()
    dmd_result = windowed_dmd(
        jnp.array(ts),
        window_size=2000,
        stride=500,
        rank=10,
        dt=0.01,
    )
    dmd_time = time.time() - t0
    print(f"  Windows: {len(dmd_result.times)}  |  Rank: {dmd_result.frequencies.shape[1]}")

    # Dominant frequency per window
    dom_freq = np.abs(dmd_result.frequencies[:, 0])
    print(f"  Dominant freq range: [{dom_freq.min():.1f}, {dom_freq.max():.1f}] Hz")
    print(f"  Time: {dmd_time:.1f}s")

    # ---- Cross-method comparison ----
    print("\n" + "=" * 70)
    print("  CROSS-METHOD COMPARISON")
    print("=" * 70)

    # 1. HMM vs DyNeMo: mutual information on hard assignments
    from sklearn.metrics import normalized_mutual_info_score
    T_common = min(len(hmm_hard), len(dynemo_hard))
    nmi = normalized_mutual_info_score(hmm_hard[:T_common], dynemo_hard[:T_common])
    print(f"\nHMM ↔ DyNeMo NMI: {nmi:.3f}")

    # 2. Signature change points vs HMM transitions
    if len(sig_result.change_points) > 0:
        sig_cp_times = sig_result.times[sig_result.change_points]
        # For each change point, find nearest HMM transition
        min_dists = []
        for cp_time in sig_cp_times:
            if len(hmm_transitions) > 0:
                dist = np.min(np.abs(hmm_transitions - cp_time))
                min_dists.append(dist)
        if min_dists:
            print(f"Signature CPs ↔ HMM transitions: mean dist = {np.mean(min_dists):.0f} samples")

    # 3. SINDy eigenvalue dynamics vs HMM state
    # Correlate max_real_eig with HMM state entropy
    hmm_entropy_at_windows = []
    for t in sindy_result.times:
        t_int = int(t)
        if t_int < len(hmm_gamma):
            p = hmm_gamma[t_int]
            ent = -np.sum(p * np.log(p + 1e-10))
            hmm_entropy_at_windows.append(ent)
    if hmm_entropy_at_windows:
        corr = np.corrcoef(
            sindy_result.max_real_eig[:len(hmm_entropy_at_windows)],
            hmm_entropy_at_windows,
        )[0, 1]
        print(f"SINDy max_Re(eig) ↔ HMM entropy: r = {corr:.3f}")

    # Summary table
    print(f"\n{'Method':<25} {'Accuracy':>10} {'Time':>10}")
    print("-" * 50)
    print(f"{'HMM (neurojax)':<25} {hmm_acc:>10.3f} {hmm_time:>9.1f}s")
    print(f"{'DyNeMo (neurojax)':<25} {dynemo_acc:>10.3f} {dynemo_time:>9.1f}s")
    print(f"{'Windowed SINDy':<25} {'(eig track)':>10} {sindy_time:>9.1f}s")
    print(f"{'Windowed LogSig':<25} {f'{len(sig_result.change_points)} CPs':>10} {sig_time:>9.1f}s")
    print(f"{'Windowed DMD':<25} {'(freq track)':>10} {dmd_time:>9.1f}s")

    # ---- Figure ----
    fig, axes = plt.subplots(6, 1, figsize=(16, 18), sharex=True,
                              gridspec_kw={"hspace": 0.3})

    samples = np.arange(T)

    # (a) Ground truth
    axes[0].plot(true_hard, linewidth=0.5, color="black")
    axes[0].set_ylabel("State")
    axes[0].set_title("(a) Ground Truth States")

    # (b) HMM
    axes[1].plot(hmm_hard, linewidth=0.5, color="tab:blue")
    axes[1].set_ylabel("State")
    axes[1].set_title(f"(b) HMM States (acc={hmm_acc:.3f})")

    # (c) DyNeMo alpha
    for k in range(min(n_states, 8)):
        axes[2].plot(dynemo_alpha[:, k], linewidth=0.3, alpha=0.7)
    axes[2].set_ylabel("alpha")
    axes[2].set_title(f"(c) DyNeMo Mixing Coefficients (acc={dynemo_acc:.3f})")

    # (d) SINDy max eigenvalue
    axes[3].plot(sindy_result.times, sindy_result.max_real_eig, "k-", linewidth=1)
    axes[3].axhline(0, color="red", linestyle="--", alpha=0.5)
    for sc in sindy_sign_changes:
        axes[3].axvline(sindy_result.times[sc], color="red", alpha=0.3, linewidth=0.5)
    axes[3].set_ylabel("max Re(λ)")
    axes[3].set_title(f"(d) SINDy Jacobian — {len(sindy_sign_changes)} sign changes")

    # (e) Signature distances
    axes[4].plot(sig_result.times[1:], sig_result.distances, "k-", linewidth=1)
    for cp in sig_result.change_points:
        axes[4].axvline(sig_result.times[cp + 1], color="green", alpha=0.5, linewidth=1)
    axes[4].set_ylabel("||Δ logsig||")
    axes[4].set_title(f"(e) Log-Signature Distances — {len(sig_result.change_points)} change points")

    # (f) DMD dominant frequency
    axes[5].plot(dmd_result.times, dom_freq, "k-", linewidth=1)
    axes[5].set_ylabel("Freq (Hz)")
    axes[5].set_xlabel("Sample")
    axes[5].set_title("(f) DMD Dominant Frequency")

    plt.suptitle("Systems Identification Comparison — Same Synthetic Data", fontsize=14)
    plt.savefig("sysid_comparison.png", dpi=150, bbox_inches="tight")
    print(f"\nFigure saved to sysid_comparison.png")


if __name__ == "__main__":
    main()
