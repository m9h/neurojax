#!/bin/bash
# ============================================================================
# Stage J: MEG Source Reconstruction
# ============================================================================
# Uses FreeSurfer surfaces + MNE-Python for:
#   1. BEM forward model from FreeSurfer surfaces
#   2. Source space (ico-5 decimation)
#   3. LCMV beamformer or dSPM inverse
#   4. Parcellation (Desikan-Killiany 68 or Destrieux 148)
#   5. Sign-flip correction across parcels
#
# Output: parcellated source timeseries for HMM/DyNeMo
# ============================================================================

set -euo pipefail
source "$(dirname "$0")/00_setup_env.sh"

SUBJECT="${1:?Usage: $0 <subject_id>}"
ATLAS="${2:-desikan}"
N_PARCELS="${3:-68}"

MEG_OUT="${WAND_DERIVATIVES}/neurojax-meg/${SUBJECT}/ses-01"
mkdir -p "${MEG_OUT}/source"

echo "=== MEG Source Reconstruction: ${SUBJECT} ==="

uv run python3 << PYEOF
import os, sys
sys.path.insert(0, '${WAND_ROOT}/../neurojax/src')

from neurojax.io.wand_meg import WANDMEGLoader

loader = WANDMEGLoader("${WAND_ROOT}")

# Load resting-state MEG
print("Loading resting-state MEG...")
raw = loader.load_resting("${SUBJECT}")
print(f"  {raw.info['nchan']} channels, {raw.n_times} samples, {raw.info['sfreq']} Hz")

# Preprocess: filter + resample
print("Preprocessing...")
raw = loader.preprocess(raw, l_freq=1.0, h_freq=45.0, resample_freq=250.0)
print(f"  After preproc: {raw.n_times} samples at {raw.info['sfreq']} Hz")

# Source reconstruction
print("Source reconstruction (${ATLAS}, ${N_PARCELS} parcels)...")
parcellated = loader.source_reconstruct(
    raw,
    subjects_dir="${SUBJECTS_DIR}",
    subject="${SUBJECT}",
    atlas="${ATLAS}",
    n_parcels=${N_PARCELS}
)
print(f"  Parcellated shape: {parcellated.shape}")  # (T, n_parcels)

# Save
import numpy as np
np.save("${MEG_OUT}/source/parcellated_${ATLAS}.npy", parcellated)
print(f"  Saved: ${MEG_OUT}/source/parcellated_${ATLAS}.npy")

# Also save for other tasks if available
for task in ['auditorymotor', 'visual']:
    try:
        raw_task = loader.load_task("${SUBJECT}", task)
        raw_task = loader.preprocess(raw_task, l_freq=1.0, h_freq=45.0, resample_freq=250.0)
        parc_task = loader.source_reconstruct(
            raw_task,
            subjects_dir="${SUBJECTS_DIR}",
            subject="${SUBJECT}",
            atlas="${ATLAS}",
            n_parcels=${N_PARCELS}
        )
        np.save("${MEG_OUT}/source/parcellated_${ATLAS}_task-${task}.npy", parc_task)
        print(f"  Saved task-{task}: {parc_task.shape}")
    except Exception as e:
        print(f"  task-{task}: skipped ({e})")

print("Source reconstruction complete.")
PYEOF

echo "=== MEG source reconstruction complete for ${SUBJECT} ==="
