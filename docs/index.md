# NeuroJAX Documentation

Welcome to the NeuroJAX documentation.

## Overview

NeuroJAX is a JAX-accelerated toolkit for MEG and EEG analysis.  It provides
GPU-native implementations of preprocessing (ASR, ICA, filtering), source
reconstruction (LCMV, DICS, SAM, MNE/dSPM/sLORETA/eLORETA), spectral
analysis (SpecParam, multitaper, Morlet wavelets), statistical inference
(GLM with permutation testing), and functional connectivity.

## Installation

We recommend using [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
uv sync
```

Or using pip:

```bash
pip install neurojax
```

## Tutorials

```{toctree}
:maxdepth: 2
:caption: Tutorials

tutorials/meg_pipeline
tutorials/mrs_mega_press
tutorials/sindy_dynamics
tutorials/koopman_operator
tutorials/ica_source_separation
tutorials/preprocessing_asr
tutorials/beamforming
tutorials/glm_inference
```

## API Reference

```{toctree}
:maxdepth: 2
:caption: API Reference

reference/neurojax
```
