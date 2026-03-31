#!/bin/bash
# Build QUIT + dependencies from source on ARM64/aarch64 (DGX Spark)
# Tested on Ubuntu 24.04, aarch64, GCC 13.3
#
# Prerequisites (all available via apt on Ubuntu 24.04):
#   cmake (>=3.29 via pip), build-essential, libeigen3-dev, zlib1g-dev,
#   libceres-dev, nlohmann-json3-dev, libargs-dev
#
# Usage: bash build_quit.sh
set -euo pipefail

PREFIX="${HOME}/.local"
NPROC=$(nproc)

# 1. CMake >= 3.29 (system cmake 3.28 is too old)
pip install cmake --upgrade 2>/dev/null

# 2. libfmt 11 (system has libfmt 9, QUIT needs 11)
if [ ! -f "${PREFIX}/lib/libfmt.a" ]; then
    echo "=== Building libfmt 11 ==="
    cd /tmp && rm -rf fmt
    git clone --depth 1 --branch 11.0.0 https://github.com/fmtlib/fmt.git
    cd fmt && mkdir build && cd build
    cmake .. -DCMAKE_INSTALL_PREFIX="${PREFIX}" -DFMT_TEST=OFF
    make -j${NPROC} && make install
fi

# 3. args cmake config (libargs-dev has no cmake integration)
mkdir -p "${PREFIX}/share/cmake/args"
cat > "${PREFIX}/share/cmake/args/argsConfig.cmake" << 'CMEOF'
add_library(taywee_args INTERFACE)
target_include_directories(taywee_args INTERFACE /usr/include)
add_library(taywee::args ALIAS taywee_args)
set(args_FOUND TRUE)
CMEOF

# 4. ITK 5.4 from source (with system Eigen to avoid version conflicts)
ITK_SRC="${HOME}/dev/ITK"
ITK_BUILD="${HOME}/dev/ITK-build"
if [ ! -f "${PREFIX}/lib/cmake/ITK-5.4/ITKConfig.cmake" ]; then
    echo "=== Building ITK 5.4 ==="
    [ -d "${ITK_SRC}/.git" ] || git clone --depth 1 --branch v5.4.2 \
        https://github.com/InsightSoftwareConsortium/ITK.git "${ITK_SRC}"
    mkdir -p "${ITK_BUILD}" && cd "${ITK_BUILD}"
    cmake "${ITK_SRC}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
        -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF \
        -DITK_USE_SYSTEM_EIGEN=ON
    make -j${NPROC} && make install
fi

# 5. QUIT
echo "=== Building QUIT ==="
QUIT_SRC="${HOME}/dev/QUIT"
[ -d "${QUIT_SRC}/.git" ] || git clone --depth 1 \
    https://github.com/spinicist/QUIT.git "${QUIT_SRC}"
mkdir -p "${QUIT_SRC}/build" && cd "${QUIT_SRC}/build" && rm -rf *
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
    -DCMAKE_PREFIX_PATH="${PREFIX};${PREFIX}/lib/cmake;${PREFIX}/share/cmake" \
    -DITK_DIR="${PREFIX}/lib/cmake/ITK-5.4" \
    -Dfmt_DIR="${PREFIX}/lib/cmake/fmt"
make -j${NPROC} && make install

echo ""
echo "=== QUIT installed ==="
echo "Binary: ${PREFIX}/bin/qi"
echo "Add to shell: export LD_LIBRARY_PATH=${PREFIX}/lib:\$LD_LIBRARY_PATH"
${PREFIX}/bin/qi --version 2>/dev/null || ${PREFIX}/bin/qi --help 2>&1 | head -3
