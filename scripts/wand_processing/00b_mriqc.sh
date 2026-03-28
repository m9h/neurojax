#!/bin/bash
# ============================================================================
# Stage 0b: MRIQC — Quality Control on ALL Raw Inputs
# ============================================================================
# Run BEFORE any processing. MRIQC produces automated quality metrics
# and visual reports for every structural and functional input:
#
#   T1w, T2w     → SNR, CNR, INU, EFC, WM/GM contrast
#   BOLD (rest + task) → tSNR, DVARS, FD, outlier fraction, GCOR
#
# Run via Neurodesk container (preferred) or Docker fallback.
# Output: BIDS derivatives/mriqc/ with HTML reports per subject.
#
# Runtime: ~30-60 min per subject (depends on number of runs)
# ============================================================================

set -euo pipefail
source "$(dirname "$0")/00_setup_env.sh"

SUBJECT="${1:?Usage: $0 <subject_id>}"

MRIQC_DIR="${WAND_DERIVATIVES}/mriqc"
mkdir -p "${MRIQC_DIR}"

echo "=== MRIQC: ${SUBJECT} ==="

# Check Docker is running
if ! docker info >/dev/null 2>&1; then
    echo "ERROR: Docker is not running. Start Docker Desktop first."
    exit 1
fi

# Pull MRIQC if not already available
if ! docker images | grep -q "nipreps/mriqc"; then
    echo "Pulling MRIQC Docker image..."
    docker pull nipreps/mriqc:24.0.2
fi

echo "Running MRIQC via Docker..."
echo "  BIDS root: ${WAND_ROOT}"
echo "  Output:    ${MRIQC_DIR}"
echo "  Subject:   ${SUBJECT}"

# Run MRIQC participant level
docker run --rm \
    -v "${WAND_ROOT}:/data:ro" \
    -v "${MRIQC_DIR}:/out" \
    -v "${MRIQC_DIR}/work:/work" \
    nipreps/mriqc:24.0.2 \
    /data /out \
    participant \
    --participant-label "${SUBJECT#sub-}" \
    --nprocs ${OMP_NUM_THREADS} \
    --no-sub \
    -w /work \
    --verbose-reports

echo ""
echo "=== MRIQC complete for ${SUBJECT} ==="
echo ""
echo "Reports:"
find "${MRIQC_DIR}" -name "${SUBJECT}*.html" 2>/dev/null | while read f; do
    echo "  ${f}"
done
echo ""
echo "Metrics: ${MRIQC_DIR}/${SUBJECT}/"
echo "Group report: Run MRIQC group level after all subjects are done:"
echo "  docker run --rm -v ${WAND_ROOT}:/data:ro -v ${MRIQC_DIR}:/out nipreps/mriqc:24.0.2 /data /out group"
