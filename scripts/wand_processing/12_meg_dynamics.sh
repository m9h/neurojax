#!/bin/bash
# ============================================================================
# Stage K: MEG Dynamics — HMM + DyNeMo Brain State Analysis
# ============================================================================
# Runs the full osl-dynamics-compatible pipeline in neurojax:
#   1. TDE + PCA (or TuckerTDE) preparation
#   2. GaussianHMM (Baum-Welch EM, 8 states)
#   3. DyNeMo (variational autoencoder, 8 modes)
#   4. State spectral decomposition (PSD + coherence per state)
#   5. Summary statistics (occupancy, lifetime, switching rate)
#   6. NNMF spectral band separation
#   7. Static baseline for comparison
#
# Also runs windowed SINDy/DMD/signatures for cross-method comparison
# ============================================================================

set -euo pipefail
source "$(dirname "$0")/00_setup_env.sh"

SUBJECT="${1:?Usage: $0 <subject_id>}"
N_STATES="${2:-8}"
N_PCA="${3:-80}"

MEG_DIR="${WAND_DERIVATIVES}/neurojax-meg/${SUBJECT}/ses-01"
PARC_FILE="${MEG_DIR}/source/parcellated_desikan.npy"

echo "=== MEG Dynamics: ${SUBJECT} (${N_STATES} states) ==="

if [ ! -f "${PARC_FILE}" ]; then
    echo "ERROR: Parcellated data not found at ${PARC_FILE}"
    echo "Run 11_meg_source_recon.sh first."
    exit 1
fi

uv run python3 << PYEOF
import os, sys, json
import numpy as np
import jax.numpy as jnp
import jax.random as jr
sys.path.insert(0, '${WAND_ROOT}/../neurojax/src')

from neurojax.data.loading import Data
from neurojax.models.hmm import GaussianHMM
from neurojax.models.dynemo import DyNeMo, DyNeMoConfig
from neurojax.analysis.state_spectra import InvertiblePCA, get_state_spectra, undo_tde_covariance
from neurojax.analysis.summary_stats import (
    fractional_occupancy, mean_lifetime, mean_interval, switching_rate, state_time_courses
)
from neurojax.analysis.nnmf import separate_spectral_components
from neurojax.analysis.static import static_summary
from neurojax.analysis.tensor_tde import TuckerTDE

n_states = ${N_STATES}
n_pca = ${N_PCA}
n_embeddings = 15
fs = 250.0  # Hz (after resampling)

# Load parcellated data
print("Loading parcellated data...")
parcellated = np.load("${PARC_FILE}")
print(f"  Shape: {parcellated.shape}")  # (T, n_parcels)
data = Data([jnp.array(parcellated)])

# --- TDE + PCA preparation ---
print(f"Preparing: TDE (n_emb={n_embeddings}) + PCA (n_pca={n_pca})...")
data.prepare({"tde_pca": {"n_embeddings": n_embeddings, "n_pca_components": n_pca}})
prepared = data.prepared_data
print(f"  Prepared shape: {prepared[0].shape}")

# Also fit InvertiblePCA for spectral inversion
from neurojax.data.loading import prepare_tde
tde_data = prepare_tde(jnp.array(parcellated), n_embeddings=n_embeddings)
pca = InvertiblePCA(n_components=n_pca)
pca.fit_transform(tde_data)

# --- Static baseline ---
print("\nStatic baseline...")
static = static_summary(prepared, fs=fs)
os.makedirs("${MEG_DIR}/static", exist_ok=True)
np.save("${MEG_DIR}/static/psd.npy", np.array(static["psd"]))
np.save("${MEG_DIR}/static/connectivity.npy", np.array(static["connectivity"]))
np.save("${MEG_DIR}/static/frequencies.npy", np.array(static["frequencies"]))
print(f"  Static PSD shape: {static['psd'].shape}")

# --- HMM ---
print(f"\nFitting HMM ({n_states} states)...")
hmm = GaussianHMM(n_states=n_states, n_channels=n_pca)
hmm_history = hmm.fit(prepared, n_epochs=20, n_init=3)
print(f"  Final LL: {hmm_history[-1]:.2f}")

# Infer states
gammas = hmm.infer(prepared)
states = state_time_courses(gammas[0])

# Save HMM results
os.makedirs("${MEG_DIR}/hmm", exist_ok=True)
np.save("${MEG_DIR}/hmm/state_probabilities.npy", np.array(gammas[0]))
np.save("${MEG_DIR}/hmm/state_means.npy", np.array(hmm.state_means))
np.save("${MEG_DIR}/hmm/state_covariances.npy", np.array(hmm.state_covariances))

# Summary stats
fo = fractional_occupancy(states, n_states=n_states)
lt = mean_lifetime(states, n_states=n_states, fs=fs)
sr = switching_rate(states, fs=fs)
mi = mean_interval(states, n_states=n_states, fs=fs)

hmm_stats = {
    "fractional_occupancy": np.array(fo).tolist(),
    "mean_lifetime_ms": (np.array(lt) * 1000).tolist(),
    "mean_interval_ms": (np.array(mi) * 1000).tolist(),
    "switching_rate_hz": float(sr),
    "n_states": n_states,
    "n_epochs": 20,
    "final_ll": hmm_history[-1],
}
with open("${MEG_DIR}/hmm/summary_stats.json", "w") as f:
    json.dump(hmm_stats, f, indent=2)
print(f"  FO: {np.array(fo).round(3)}")
print(f"  Lifetimes (ms): {(np.array(lt)*1000).round(1)}")
print(f"  Switching rate: {float(sr):.2f} Hz")

# State spectra
print("  Computing state spectra...")
n_parcels = parcellated.shape[1]
spectra = get_state_spectra(
    jnp.array(hmm.state_covariances), pca, n_parcels, n_embeddings, fs=fs
)
np.save("${MEG_DIR}/hmm/state_psd.npy", np.array(spectra["psd"]))
np.save("${MEG_DIR}/hmm/state_coherence.npy", np.array(spectra["coherence"]))
np.save("${MEG_DIR}/hmm/frequencies.npy", np.array(spectra["frequencies"]))
print(f"  State PSD shape: {spectra['psd'].shape}")

# NNMF
print("  NNMF spectral separation...")
nnmf = separate_spectral_components(spectra["psd"], spectra["frequencies"], n_components=3)
np.save("${MEG_DIR}/hmm/nnmf_spectral_components.npy", np.array(nnmf["spectral_components"]))
np.save("${MEG_DIR}/hmm/nnmf_activation_maps.npy", np.array(nnmf["activation_maps"]))

# --- DyNeMo ---
print(f"\nFitting DyNeMo ({n_states} modes)...")
dynemo_config = DyNeMoConfig(
    n_modes=n_states, n_channels=n_pca,
    sequence_length=200, n_epochs=15,
    inference_n_units=64, model_n_units=64,
    batch_size=16, learning_rate=1e-3,
)
dynemo = DyNeMo(dynemo_config)
dynemo_history = dynemo.fit(prepared, n_epochs=15)
print(f"  Final loss: {dynemo_history[-1]['loss']:.4f}")

# Infer alpha
alphas = dynemo.infer(prepared)

os.makedirs("${MEG_DIR}/dynemo", exist_ok=True)
np.save("${MEG_DIR}/dynemo/alpha.npy", np.array(alphas[0]))
np.save("${MEG_DIR}/dynemo/mode_means.npy", np.array(dynemo.get_means()))
np.save("${MEG_DIR}/dynemo/mode_covariances.npy", np.array(dynemo.get_covariances()))

# DyNeMo state spectra
dyn_spectra = get_state_spectra(
    dynemo.get_covariances(), pca, n_parcels, n_embeddings, fs=fs
)
np.save("${MEG_DIR}/dynemo/mode_psd.npy", np.array(dyn_spectra["psd"]))
np.save("${MEG_DIR}/dynemo/mode_coherence.npy", np.array(dyn_spectra["coherence"]))

print("\n=== MEG dynamics complete ===")
print(f"HMM:    ${MEG_DIR}/hmm/")
print(f"DyNeMo: ${MEG_DIR}/dynemo/")
print(f"Static: ${MEG_DIR}/static/")
PYEOF

echo "=== MEG dynamics complete for ${SUBJECT} ==="
