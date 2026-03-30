#!/usr/bin/env bash
# Build QUIT (QUantitative Imaging Tools) from source on ARM64/aarch64 DGX Spark
# Usage: bash build_quit.sh 2>&1 | tee build_quit.log
#
# Prerequisites (already confirmed available on this system):
#   cmake, g++, make, libeigen3-dev, zlib1g-dev
#
# QUIT uses a SuperBuild CMake pattern that automatically downloads and builds
# ITK as an external project. This means no separate ITK installation is needed,
# but the first build will be slow (~20-30 min) as it compiles ITK from source.
set -euo pipefail

QUIT_SRC="/home/mhough/dev/QUIT"
QUIT_BUILD="${QUIT_SRC}/build"
INSTALL_PREFIX="/home/mhough/.local"
NPROC="$(nproc)"

echo "=============================================="
echo "  QUIT Build Script for ARM64 (aarch64)"
echo "  $(date)"
echo "  Cores: ${NPROC}"
echo "=============================================="

# ---- Step 1: Check build dependencies ----
echo ""
echo "=== Step 1: Checking build dependencies ==="
MISSING=()
which cmake   >/dev/null 2>&1 || MISSING+=("cmake")
which g++     >/dev/null 2>&1 || MISSING+=("build-essential")
which make    >/dev/null 2>&1 || MISSING+=("build-essential")
test -f /usr/include/eigen3/Eigen/Core || MISSING+=("libeigen3-dev")
test -f /usr/include/zlib.h            || MISSING+=("zlib1g-dev")

if [ ${#MISSING[@]} -gt 0 ]; then
    echo "Installing missing packages: ${MISSING[*]}"
    sudo apt-get update -qq
    sudo apt-get install -y "${MISSING[@]}"
else
    echo "All dependencies found (cmake, g++, eigen3, zlib). Skipping apt."
fi

# Also need nlohmann-json and fmt (QUIT deps that may not be vendored)
for pkg in nlohmann-json3-dev libfmt-dev; do
    if ! dpkg -s "$pkg" >/dev/null 2>&1; then
        echo "Installing optional dependency: $pkg"
        sudo apt-get install -y "$pkg" || echo "WARNING: $pkg not available, QUIT may vendor it"
    fi
done

# ---- Step 2: Clone QUIT ----
echo ""
echo "=== Step 2: Clone QUIT ==="
if [ -d "${QUIT_SRC}/.git" ]; then
    echo "QUIT already cloned at ${QUIT_SRC}"
    cd "${QUIT_SRC}"
    git log --oneline -1
else
    git clone --recursive https://github.com/spinicist/QUIT.git "${QUIT_SRC}"
fi

# ---- Step 3: Configure with CMake ----
echo ""
echo "=== Step 3: Configure CMake build ==="
mkdir -p "${QUIT_BUILD}"
cd "${QUIT_BUILD}"

# QUIT's top-level CMakeLists.txt is typically a SuperBuild that handles ITK.
# Check if there's a SuperBuild structure.
if grep -q "ExternalProject" "${QUIT_SRC}/CMakeLists.txt" 2>/dev/null; then
    echo "Detected SuperBuild pattern (will download & build ITK automatically)"
fi

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" \
    -DBUILD_SHARED_LIBS=OFF

# ---- Step 4: Build ----
echo ""
echo "=== Step 4: Build (using ${NPROC} cores) ==="
echo "This may take 20-40 minutes on first build (ITK compilation)..."
time make -j"${NPROC}"

# ---- Step 5: Install ----
echo ""
echo "=== Step 5: Install to ${INSTALL_PREFIX} ==="
make install

# ---- Step 6: Verify ----
echo ""
echo "=== Step 6: Verify installation ==="

# QUIT may install a single 'qi' binary or individual tools like qi_despot1
FOUND=0
if [ -x "${INSTALL_PREFIX}/bin/qi" ]; then
    echo "SUCCESS: qi binary found"
    "${INSTALL_PREFIX}/bin/qi" --help 2>&1 | head -5 || true
    FOUND=1
fi

# Check for individual QUIT tools
for tool in qi_despot1 qi_despot2 qi_vfa qi_afi qi_dream qi_ssfp_planet \
            qi_mpm_r2s qi_jsr qi_zshim qi_asl qi_perfusion; do
    if [ -x "${INSTALL_PREFIX}/bin/${tool}" ]; then
        echo "  Found: ${tool}"
        FOUND=1
    fi
done

if [ "$FOUND" -eq 0 ]; then
    echo "WARNING: No qi binaries found in ${INSTALL_PREFIX}/bin/"
    echo "Checking build directory for binaries..."
    find "${QUIT_BUILD}" -name "qi*" -executable -type f 2>/dev/null | head -10
    echo ""
    echo "Build may have failed or install prefix may differ."
    echo "Check: ${QUIT_BUILD}/CMakeCache.txt for CMAKE_INSTALL_PREFIX"
fi

echo ""
echo "=============================================="
echo "  Build finished at $(date)"
echo "  Binaries: ${INSTALL_PREFIX}/bin/"
echo "  Add to PATH: export PATH=\"${INSTALL_PREFIX}/bin:\$PATH\""
echo "=============================================="
