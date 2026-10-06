# Oracle container for osl-dynamics baseline comparison.
#
# Generates osl-dynamics HMM/DyNeMo baselines and saves .npy for parity tests
# against the JAX reimplementation (src/neurojax/models/tests/test_*oracle*.py).
#
# CPU-only by design: osl-dynamics needs TensorFlow, which has no aarch64 GPU
# build, and a baseline generator needs no GPU on x86 either — so this uses a
# slim Python base, not a multi-GB NVIDIA image.
#
# Versions are pinned from containers/oracle-osl-requirements.txt — the EXACT
# set verified to import + run on aarch64 (2026-06-23). The previous loose
# ranges break today: scikit-image pulls numpy>=2 (incompatible with TF 2.17 /
# numba), and osl-dynamics eagerly imports PyYAML + mat73, which are absent from
# its wheel metadata.
#
# Build:
#   docker build -f containers/oracle-osl.Dockerfile -t neurojax/oracle-osl .
#
# Generate the HMM baseline fixtures (200 epochs → means/states converge):
#   docker run --rm -e DATA_DIR=/data \
#     -e N_SAMPLES=12000 -e N_STATES=4 -e N_CHANNELS=8 -e N_EPOCHS=200 \
#     -v $(pwd)/tests/data/oracle_osl/hmm:/data \
#     neurojax/oracle-osl \
#     sh -c "python /scripts/generate_synthetic.py && python /scripts/run_hmm_baseline.py"
#
FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive

# libgomp1: required by numba + TensorFlow CPU at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY containers/oracle-osl-requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY containers/scripts/ /scripts/
WORKDIR /data
