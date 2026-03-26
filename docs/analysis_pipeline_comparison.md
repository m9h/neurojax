# Analysis Pipeline Comparison: NeuroJAX vs osl-dynamics vs SPM

*For lab meeting review — comparing analysis pipelines stage by stage*

## Overview

Three ecosystems for MEG/EEG dynamic brain state analysis, applied to the same data (WAND resting-state MEG, sub-08033, CTF 275ch, 10 min).

| | **osl-dynamics** (Oxford) | **NeuroJAX** (this project) | **SPM/DCM** (UCL) |
|---|---|---|---|
| Language | Python + TensorFlow | Python + JAX | MATLAB/Python (spm-python) |
| Approach | Data-driven (HMM, DyNeMo) | Data-driven + physics-informed | Model-based Bayesian |
| Differentiable | Partial (TF graph) | End-to-end (JAX autodiff) | No (EM/VB) |
| GPU | TF GPU | JAX GPU/TPU | No |

---

## Stage-by-Stage Pipeline Comparison

### Stage 1: Preprocessing

| Step | osl-dynamics (osl-ephys) | NeuroJAX | SPM |
|---|---|---|---|
| **Load** | `osl.source_recon.run_src_batch` | `WANDMEGLoader.load_raw()` (MNE CTF) | `spm_eeg_convert` |
| **Filter** | `mne.filter()` via osl config | `WANDMEGLoader.preprocess()` 1-45Hz | `spm_eeg_filter` |
| **Bad channels** | `osl.preprocessing` | MNE `find_bad_channels_maxwell` | `spm_eeg_review` (manual) |
| **Downsample** | `mne.resample()` | 250 Hz default | `spm_eeg_downsample` |
| **ICA artifact removal** | Optional (osl-ephys) | Not yet (use MNE ICA) | `spm_eeg_spatial_confounds` |

### Stage 2: Source Reconstruction

| Step | osl-dynamics (osl-ephys) | NeuroJAX | SPM |
|---|---|---|---|
| **Co-registration** | `osl.source_recon.rhino` (custom) | Headshape + fsaverage template | `spm_eeg_inv_datareg` |
| **Forward model** | Single-shell (via FSL BET) | Single-sphere (from sensor positions) | Single-shell BEM |
| **Source space** | Cortical surface (fsaverage) | `oct5` or `oct6` (MNE) | Cortical mesh (canonical) |
| **Inverse method** | LCMV beamformer (default) | sLORETA / dSPM / MNE (configurable) | MSP / IID / LCMV |
| **Parcellation** | DK (68) or custom (e.g., Glasser 360) | `aparc` (68) / `aparc.a2009s` (148) | AAL / DK / custom |
| **Sign flip** | `osl.source_recon.find_flips` | `mean_flip` mode in MNE | Not standard |

**Sensitivity concern:** Source imaging choices (forward model, inverse method, parcellation) directly affect the dynamics discovered downstream. NeuroJAX includes a `--sensitivity` flag to systematically compare.

### Stage 3: Data Preparation

| Step | osl-dynamics | NeuroJAX | SPM/DCM |
|---|---|---|---|
| **Time-delay embedding** | `Data.prepare({"tde": {"n_embeddings": 15}})` | `prepare_tde(n_embeddings=15)` | Not used |
| **PCA** | `Data.prepare({"pca": {"n_pca_components": 80}})` | `prepare_pca(n_pca_components=80)` | Not used (DCM uses parcels directly) |
| **TDE+PCA combined** | `{"tde_pca": {...}}` | Same API, drop-in compatible | N/A |
| **TuckerTDE** (tensor) | Not available | `TuckerTDE(rank_channels=10, rank_lags=8)` | N/A |
| **Amplitude envelope** | `{"amplitude_envelope": {}}` | `{"amplitude_envelope": {}}` | Hilbert envelope in SPM |
| **Standardize** | `{"standardize": {}}` | `{"standardize": {}}` | Implicit in DCM |

**NeuroJAX addition:** `TuckerTDE` preserves multilinear structure (channel × lag) instead of flattening, avoiding the information loss in TDE+PCA.

### Stage 4: Dynamic State Inference

| Method | osl-dynamics | NeuroJAX | SPM/DCM |
|---|---|---|---|
| **HMM** | `models.hmm.Config + Model` (TF Keras) | `GaussianHMM` (pure JAX Baum-Welch) | Not available |
| **DyNeMo** | `models.dynemo.Config + Model` (TF) | `DyNeMo` (equinox/optax VAE) | Not available |
| **M-DyNeMo** | Available | Not yet | N/A |
| **SINDy** | Not available | `windowed_sindy()` (jaxctrl) | N/A |
| **Log-Signatures** | Not available | `windowed_signatures()` (signax) | N/A |
| **Koopman/DMD** | Not available | `windowed_dmd()` (jaxctrl) | N/A |
| **DCM** | Not available | Not yet (planned via SPM-Python oracle) | `spm_dcm_erp` / `spm_dcm_csd` |

**Key difference:** osl-dynamics and NeuroJAX are data-driven (discover states from data). SPM/DCM is model-driven (specify a neural mass model, infer its parameters). The comparison asks whether they agree.

### Stage 5: Post-hoc Analysis

| Analysis | osl-dynamics | NeuroJAX | SPM |
|---|---|---|---|
| **Fractional occupancy** | `analysis.modes.fractional_occupancy` | `summary_stats.fractional_occupancy` | N/A |
| **Mean lifetime** | `analysis.modes.mean_lifetimes` | `summary_stats.mean_lifetime` | N/A |
| **Switching rate** | `analysis.modes.switching_rates` | `summary_stats.switching_rate` | N/A |
| **State spectra** | `analysis.spectral.get_spectra` | `state_spectra.get_state_spectra` | `spm_dcm_csd` (model spectra) |
| **State coherence** | `analysis.spectral.get_covariance_matrix_from_spectra` | `state_spectra.coherence_from_cpsd` | DCM A-matrix |
| **Multitaper PSD** | Not built-in (uses scipy) | `multitaper.multitaper_psd` (pure JAX) | `spm_eeg_tf` |
| **Regression spectra** | `analysis.spectral.regression_spectra` | `regression_spectra.compute_regression_spectra` | N/A |
| **NNMF separation** | Not built-in | `nnmf.spectral_nnmf` | N/A |
| **Static baseline** | Manual | `static.static_summary` | Standard SPM results |
| **Functional networks** | Not built-in | `funcnet` (MI, graph measures) | DCM effective connectivity |
| **Recurrence** | Not built-in | `recurrence.recurrence_matrix` | N/A |
| **Surrogates** | Not built-in | `surrogates.phase_randomize` | Not standard |

### Stage 6: Visualization & Reporting

| | osl-dynamics | NeuroJAX | SPM |
|---|---|---|---|
| **Brain maps** | `analysis.power.save` (nilearn) | MNE + nilearn (via parcellation) | SPM glass brain |
| **State networks** | `utils.plotting.plot_connections` | matplotlib (from funcnet adjacency) | DCM network diagram |
| **HTML report** | Not built-in | `reporting.html.HTMLReport` | SPM batch report |

---

## What NeuroJAX Adds Beyond osl-dynamics

1. **5-method comparison** — HMM + DyNeMo + SINDy + Log-Signatures + Koopman on the same data
2. **TuckerTDE** — tensor decomposition alternative to flatten+PCA
3. **Source imaging sensitivity** — systematic comparison across parcellation/inverse/spacing choices
4. **Physics-informed dynamics** — SINDy discovers governing equations, not just state labels
5. **Interpretable signatures** — log-signatures detect change points without assuming a model
6. **Static baseline** — quantifies whether dynamics add value over time-averaged analysis
7. **End-to-end differentiability** — JAX autodiff from sensor error to biophysical parameters
8. **Containerized oracles** — osl-dynamics and (planned) SPM run in isolated containers

---

## Data: WAND MEG sub-08033

| Property | Value |
|---|---|
| System | CTF DSQ-3000, 275 channels |
| Site | CUBRIC, Cardiff |
| Task | 10-minute eyes-open resting state |
| Sampling rate | 1200 Hz (raw) → 250 Hz (after resampling) |
| Other modalities | DWI, fMRI, MRS, qMT, TMS (8 sessions total) |
| Multi-modal potential | Structural connectivity (DWI) constrains DCM, qMT for tissue properties |

---

## Planned Additions

- [ ] SPM-Python DCM oracle container
- [ ] M-DyNeMo (multi-scale DyNeMo)
- [ ] SINDy+Signature hybrid (interpretable DyNeMo replacement)
- [ ] Group-level comparison (multiple WAND subjects)
- [ ] Cam-CAN / OMEGA as additional datasets
- [ ] Neurodesk submission for osl-dynamics container
