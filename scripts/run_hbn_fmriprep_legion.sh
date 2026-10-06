#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# Run on the LEGION (x86_64 + Docker). Computes fMRIPrep (WITH FreeSurfer) for a
# couple of HBN subjects, so we can compare modern FS/fMRIPrep against HBN's
# stale 2018 FreeSurfer v6.0.0 derivatives.
#
# fMRIPrep runs recon-all internally, so one run gives BOTH modern FreeSurfer
# surfaces (fsnative/fsaverage) AND fMRIPrep-format functional (task + rest).
#
# Usage:   FS_LICENSE=~/freesurfer/license.txt ./run_hbn_fmriprep_legion.sh
# Needs:   docker, awscli, ~30 GB free, an internet connection, a FreeSurfer
#          license (free: https://surfer.nmr.mgh.harvard.edu/registration.html).
# Runtime: ~12-24 h PER subject (recon-all dominates; fMRIPrep is CPU-bound).
# ----------------------------------------------------------------------------
set -euo pipefail

# Two HBN subjects that have raw T1 + task-rest BOLD + the old FS6 (for the diff).
SUBJECTS=("NDARAD481FXF" "NDARAE199TDD")
SITE="Site-SI"

ROOT="${HBN_ROOT:-$HOME/hbn_fmriprep}"
BIDS="$ROOT/bids"; OUT="$ROOT/derivatives"; WORK="$ROOT/work"
FS_LICENSE="${FS_LICENSE:-$HOME/freesurfer/license.txt}"
# Image already present on the Legion: docker.io/nipreps/fmriprep:25.2.3
FMRIPREP_VER="${FMRIPREP_VER:-25.2.3}"
NTHREADS="${NTHREADS:-12}"; MEM_MB="${MEM_MB:-24000}"

[ -f "$FS_LICENSE" ] || { echo "ERROR: FreeSurfer license not at $FS_LICENSE (set FS_LICENSE=...)"; exit 1; }
mkdir -p "$BIDS" "$OUT" "$WORK"

# --- minimal BIDS wrapper around the HBN per-site raw (already BIDS sub-dirs) ---
cat > "$BIDS/dataset_description.json" <<'JSON'
{ "Name": "HBN subset (fMRIPrep modern vs FS6 comparison)", "BIDSVersion": "1.8.0", "DatasetType": "raw" }
JSON

for sub in "${SUBJECTS[@]}"; do
  echo "[stage] downloading sub-$sub raw BIDS (anat + func) from HBN S3 ..."
  aws s3 sync --no-sign-request \
    "s3://fcp-indi/data/Projects/HBN/MRI/$SITE/sub-$sub/" "$BIDS/sub-$sub/"
done
{ echo "participant_id"; for s in "${SUBJECTS[@]}"; do echo "sub-$s"; done; } > "$BIDS/participants.tsv"

# --- fMRIPrep (with FreeSurfer recon-all) -----------------------------------
ENGINE="$(command -v podman || command -v docker)"
# Run subjects SEQUENTIALLY and lift podman's PID cap. Running multiple subjects
# in parallel x recon-all -openmp blew past podman's default --pids-limit (2048)
# -> "libgomp: Thread creation failed: Resource temporarily unavailable" and a
# nipype deadlock. One subject at a time + --pids-limit=0 + omp 8 fixes it.
for sub in "${SUBJECTS[@]}"; do
  echo "[fmriprep] $FMRIPREP_VER via $ENGINE on sub-$sub"
  # rootless podman on Fedora/SELinux: :z relabels bind mounts; bump shm for fMRIPrep.
  "$ENGINE" run --rm --shm-size=8g --pids-limit=0 \
    -v "$BIDS:/data:ro,z" -v "$OUT:/out:z" -v "$WORK:/work:z" \
    -v "$FS_LICENSE:/opt/freesurfer/license.txt:ro,z" \
    "nipreps/fmriprep:${FMRIPREP_VER}" \
    /data /out participant \
    --participant-label "$sub" \
    --output-spaces MNI152NLin2009cAsym fsaverage5 fsnative \
    --fs-license-file /opt/freesurfer/license.txt \
    --nthreads "$NTHREADS" --omp-nthreads 8 --mem-mb "$MEM_MB" \
    --work-dir /work --notrack --skip-bids-validation
done

# --- pull HBN's 2018 FS6 for the same subjects, to diff against ------------
for sub in "${SUBJECTS[@]}"; do
  aws s3 sync --no-sign-request \
    "s3://fcp-indi/data/Projects/HBN/derivatives/Freesurfer_version6.0.0/$sub/" \
    "$ROOT/freesurfer6/$sub/"
done

echo "DONE."
echo "  modern fMRIPrep(+FS) : $OUT  (sourcedata/freesurfer = new recon-all)"
echo "  HBN 2018 FS v6.0.0   : $ROOT/freesurfer6"
echo "Compare e.g.: aparc.stats thickness/area, aseg volumes, white/pial surfaces."
