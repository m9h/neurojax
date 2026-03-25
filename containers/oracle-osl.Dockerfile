# Oracle container for osl-dynamics baseline comparison.
#
# Provides a self-contained environment for running osl-dynamics HMM/DyNeMo
# pipelines and saving results as .npy files for comparison with neurojax.
#
# Build:
#   docker build -f containers/oracle-osl.Dockerfile -t neurojax/oracle-osl .
#
# Run (mount data volume):
#   docker run --gpus all --ipc=host \
#     -v $(pwd)/data:/data \
#     neurojax/oracle-osl \
#     python /scripts/run_hmm_baseline.py
#
FROM nvcr.io/nvidia/pytorch:26.02-py3

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# osl-dynamics pulls in TF, nibabel, nilearn, numba, etc.
# Install osl-dynamics with ALL transitive deps it eagerly imports.
# Pin numpy<2 for TF 2.17 compat (this is why we containerise).
RUN pip install --no-cache-dir \
    "numpy<2" \
    "osl-dynamics>=3.0,<4" \
    "tensorflow>=2.15,<2.18" \
    "tensorflow-probability[tf]>=0.23,<0.25" \
    "tf-keras>=2.15,<2.18" \
    "mne>=1.6" \
    "nibabel>=5" \
    "nilearn>=0.13" \
    "numba>=0.60" \
    "scikit-image>=0.22" \
    "pqdm>=0.2" \
    "scikit-learn>=1.4" \
    "seaborn>=0.13"

# Copy oracle scripts into the image
COPY containers/scripts/ /scripts/

WORKDIR /data
