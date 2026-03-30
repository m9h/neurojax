#!/usr/bin/env bash
# Download publicly available MRS test datasets for MEGA-PRESS pipeline validation
# Usage: bash download_all.sh
# Skip datasets with: bash download_all.sh --skip-large
#
# Datasets requiring registration (Big GABA) are noted but not auto-downloaded.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SKIP_LARGE=false
if [[ "${1:-}" == "--skip-large" ]]; then
    SKIP_LARGE=true
fi

echo "=== MRS Test Dataset Downloader ==="
echo "Target directory: $SCRIPT_DIR"
echo ""

# ---------------------------------------------------------------------------
# 1. ISMRM MRS Fitting Challenge
# ---------------------------------------------------------------------------
echo "--- [1/4] ISMRM MRS Fitting Challenge ---"
DEST="$SCRIPT_DIR/ismrm_fitting_challenge"
mkdir -p "$DEST"

if [ -f "$DEST/.downloaded" ]; then
    echo "  Already downloaded, skipping. Remove $DEST/.downloaded to re-download."
else
    echo "  Downloading from UMN Data Repository (DOI: 10.13020/3bk2-bv32)..."

    # Try the UMN Conservancy direct link first
    if wget -q --show-progress -O "$DEST/FittingChallenge.zip" \
        "https://conservancy.umn.edu/bitstreams/9aca2375-fc25-4e13-b96f-5203b265d642/download" 2>/dev/null; then
        echo "  Extracting..."
        cd "$DEST" && unzip -qo FittingChallenge.zip && cd "$SCRIPT_DIR"
        touch "$DEST/.downloaded"
        echo "  Done."
    else
        echo "  Direct download failed. Try manually:"
        echo "    1. Visit: https://conservancy.umn.edu/handle/11299/217895"
        echo "    2. Download the ZIP file"
        echo "    3. Extract to: $DEST/"
        echo ""
        echo "  Alternative: search Zenodo for 'ISMRM MRS Fitting Challenge'"
    fi
fi
echo ""

# ---------------------------------------------------------------------------
# 2. Big GABA (requires registration)
# ---------------------------------------------------------------------------
echo "--- [2/4] Big GABA ---"
DEST="$SCRIPT_DIR/big_gaba"
mkdir -p "$DEST"

echo "  REQUIRES REGISTRATION -- cannot auto-download."
echo "  Steps:"
echo "    1. Register at https://www.nitrc.org/account/register.php"
echo "    2. Visit https://www.nitrc.org/projects/biggaba/"
echo "    3. Accept the data use agreement"
echo "    4. Download from the 'Downloads' tab"
echo "    5. Extract to: $DEST/"
echo ""
echo "  The full dataset is ~50 GB. For testing, download a single site."
echo "  Recommended: Site 01 (Siemens) or any Siemens TWIX site."
echo ""

# ---------------------------------------------------------------------------
# 3. SMART MRS
# ---------------------------------------------------------------------------
echo "--- [3/4] SMART MRS ---"
DEST="$SCRIPT_DIR/smart_mrs"
mkdir -p "$DEST"

if [ -f "$DEST/.downloaded" ]; then
    echo "  Already downloaded, skipping."
else
    echo "  Cloning SMART-MRS from GitHub..."

    # Try the known GitHub repo
    if git clone --depth 1 https://github.com/HarryBugler/SMART-MRS.git "$DEST/SMART-MRS" 2>/dev/null; then
        touch "$DEST/.downloaded"
        echo "  Done."
    else
        echo "  GitHub clone failed. Trying alternative names..."
        # Try alternative repo names
        for repo in "HarryBugler/smart-mrs" "HarryBugler/SMARTMRS" "CIC-methods/SMART-MRS"; do
            if git clone --depth 1 "https://github.com/$repo.git" "$DEST/SMART-MRS" 2>/dev/null; then
                touch "$DEST/.downloaded"
                echo "  Cloned from $repo. Done."
                break
            fi
        done

        if [ ! -f "$DEST/.downloaded" ]; then
            echo "  Could not find SMART-MRS repository automatically."
            echo "  Search GitHub for: 'SMART MRS Bugler MEGA-PRESS'"
            echo "  Or check the paper: Bugler et al. MRM 2024, DOI: 10.1002/mrm.30042"
            echo "  Look for data availability statement for Zenodo/OSF links."
        fi
    fi
fi
echo ""

# ---------------------------------------------------------------------------
# 4. MRSHub Edited MRS Examples
# ---------------------------------------------------------------------------
echo "--- [4/4] MRSHub Edited MRS Examples ---"
DEST="$SCRIPT_DIR/mrshub_edited_examples"
mkdir -p "$DEST"

if [ -f "$DEST/.downloaded" ]; then
    echo "  Already downloaded, skipping."
else
    # 4a. Osprey MEGA-PRESS example data
    echo "  [4a] Downloading Osprey MEGA-PRESS example data..."
    OSPREY_DEST="$DEST/osprey_mega"
    mkdir -p "$OSPREY_DEST"

    # Use git sparse checkout to get just the MEGA-PRESS examples
    if command -v git &>/dev/null; then
        TMPDIR=$(mktemp -d)
        if git clone --depth 1 --filter=blob:none --sparse \
            https://github.com/schorschinho/osprey.git "$TMPDIR/osprey" 2>/dev/null; then
            cd "$TMPDIR/osprey"
            git sparse-checkout set exampledata/sdat/MEGA 2>/dev/null || true
            if [ -d "exampledata/sdat/MEGA" ]; then
                cp -r exampledata/sdat/MEGA/* "$OSPREY_DEST/" 2>/dev/null || true
                echo "    Osprey MEGA-PRESS SDAT examples downloaded."
            else
                echo "    Osprey sparse checkout did not yield expected path."
                echo "    Trying full exampledata..."
                git sparse-checkout set exampledata 2>/dev/null || true
                if [ -d "exampledata" ]; then
                    cp -r exampledata "$OSPREY_DEST/" 2>/dev/null || true
                fi
            fi
            cd "$SCRIPT_DIR"
        else
            echo "    Failed to clone Osprey. Skipping."
        fi
        rm -rf "$TMPDIR"
    fi

    # 4b. spec2nii test data (NIfTI-MRS MEGA-PRESS examples)
    echo "  [4b] Downloading spec2nii test data (NIfTI-MRS format)..."
    SPEC2NII_DEST="$DEST/spec2nii_tests"
    mkdir -p "$SPEC2NII_DEST"

    if [ "$SKIP_LARGE" = false ]; then
        TMPDIR=$(mktemp -d)
        if git clone --depth 1 --filter=blob:none --sparse \
            https://github.com/wtclarke/spec2nii.git "$TMPDIR/spec2nii" 2>/dev/null; then
            cd "$TMPDIR/spec2nii"
            git sparse-checkout set tests 2>/dev/null || true
            if [ -d "tests" ]; then
                cp -r tests/* "$SPEC2NII_DEST/" 2>/dev/null || true
                echo "    spec2nii test data downloaded."
            fi
            cd "$SCRIPT_DIR"
        else
            echo "    Failed to clone spec2nii. Skipping."
        fi
        rm -rf "$TMPDIR"
    else
        echo "    Skipped (--skip-large). Run without flag to include."
    fi

    # 4c. Gannet example data
    echo "  [4c] Downloading Gannet example data..."
    GANNET_DEST="$DEST/gannet_examples"
    mkdir -p "$GANNET_DEST"

    TMPDIR=$(mktemp -d)
    if git clone --depth 1 --filter=blob:none --sparse \
        https://github.com/markmikkelsen/Gannet.git "$TMPDIR/gannet" 2>/dev/null; then
        cd "$TMPDIR/gannet"
        git sparse-checkout set ExampleData 2>/dev/null || true
        if [ -d "ExampleData" ]; then
            cp -r ExampleData/* "$GANNET_DEST/" 2>/dev/null || true
            echo "    Gannet example data downloaded."
        fi
        cd "$SCRIPT_DIR"
    else
        echo "    Failed to clone Gannet. Skipping."
    fi
    rm -rf "$TMPDIR"

    touch "$DEST/.downloaded"
    echo "  Done."
fi
echo ""

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "=== Download Summary ==="
echo ""
for d in ismrm_fitting_challenge big_gaba smart_mrs mrshub_edited_examples; do
    if [ -f "$SCRIPT_DIR/$d/.downloaded" ]; then
        SIZE=$(du -sh "$SCRIPT_DIR/$d" 2>/dev/null | cut -f1)
        echo "  [OK]   $d ($SIZE)"
    else
        echo "  [SKIP] $d (not downloaded or requires manual action)"
    fi
done
echo ""
echo "Done. See README.md files in each subdirectory for details."
