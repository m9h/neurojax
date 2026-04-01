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

## FreeSurfer Processing: WAND T1w/T2w Strategy

### Literature & Dataset Review (2026-03-29)

The WAND team (McNabb et al. 2025, *Scientific Data* 12:220) **does not distribute FreeSurfer derivatives**. Their provided derivatives are limited to HD-BET brain extraction, eddy QC, and MRIQC. Only one published study (Stylianopoulou et al. 2025, *Scientific Reports*) has run FreeSurfer on WAND data (126 subjects, Desikan-Killiany atlas), but does not specify which T1w session was used.

### The Two T1w Acquisitions

| Parameter | ses-02 (Connectom 3T, 300mT/m) | ses-03 (Prisma 3T) |
|---|---|---|
| Matrix | 192 × 256 × 256 | 176 × 256 × 256 |
| FOV | 256 × 256 × 192 mm | 256 × 288 × 176 mm |
| Voxel size | 1 × 1 × 1 mm | 1 × 1 × 1 mm |
| TR / TE / TI | 2300 / 2 / 857 ms | 2250 / 3.06 / 850 ms |
| Flip angle | 9° | 9° |
| Data type | FLOAT | SHORT |

**Key findings:**
- Different scanners, different TR/TE/TI → cannot average via `mri_robust_template`
- Different matrix sizes → recon-all rejects multiple `-i` inputs ("mismatched dimensions")
- ses-03 T2w (1×1×2 mm, 256×256×88) is not co-registered with either T1w; recon-all `-T2pial` handles registration internally via BBR
- ses-03 Prisma protocol is closest to HCP/ADNI standard MPRAGE
- ses-03 has 288mm FOV on one axis → requires `-cw256` flag
- ses-02 Connectom T1w is co-registered with DWI session (important for TRACULA)

### Processing Strategy

Run **two separate recon-all** pipelines per subject to compare and serve different downstream needs:

1. **`sub-XXXXX_ses-02`** — Connectom 3T T1w only
   - Best for DWI integration (TRACULA, tractography, connectomics)
   - Same-session coregistration with AxCaliber/CHARMED diffusion data
2. **`sub-XXXXX_ses-03`** — Prisma 3T T1w + gradient-corrected T2w (`-T2pial -cw256`)
   - Best for cortical morphometry (T2w improves pial surface accuracy)
   - Standard MPRAGE protocol, comparable to HCP/ADNI
   - T1w/T2w ratio available for myelin mapping (Glasser & Van Essen 2011)
3. **`sub-XXXXX_ses-04`** — 7T 0.7mm iso T1w MPRAGE (`-hires`)
   - Sub-millimeter cortical surface reconstruction
   - Best for laminar/columnar analysis, fine-grained parcellation
   - 7T also has MP2RAGE (ses-06, 0.7mm) and multi-echo GRE (0.67mm) for quantitative mapping

Both runs include advanced FreeSurfer 8.2.0 segmentations:
- Hippocampal subfields + amygdala nuclei
- Thalamic nuclei
- Hypothalamic subunits
- Brainstem substructures
- SynthSeg contrast-invariant parcellation

If longitudinal-style within-subject template is needed later, use `recon-all -base` + `recon-all -long` across the two sessions.

### Thalamic Connectivity-Based Segmentation (2026-03-31)

Implemented Johansen-Berg & Behrens probtrackx2 thalamic parcellation using CUBRIC-preprocessed CHARMED data + bedpostx_gpu + FreeSurfer aparc+aseg cortical targets.

**Method**: Seed from thalamus (aseg labels 10/49), 7 cortical targets per hemisphere (prefrontal, premotor, M1, somatosensory, posterior parietal, temporal, occipital). Hybrid target definition: Desikan-Killiany for most regions, BA4/BA6 exvivo labels for motor/premotor split. 5000 samples per voxel via probtrackx2_gpu, winner-takes-all classification via find_the_biggest.

**Containers**: freesurfer-tracula (FS 7.4.1) for TRACULA/probtrackx2, freesurfer (FS 8.2.0) for recon-all.

**Results (sub-08033, lh/rh voxel counts at 2mm):**

| Region | lh | rh | Expected Nucleus |
|---|---|---|---|
| Somatosensory | 471 | 475 | VPL/VPM |
| Occipital | 213 | 129 | LGN |
| Premotor | 185 | 171 | VA |
| Primary Motor | 152 | 195 | VL |
| Post. Parietal | 121 | 147 | Pulvinar/LP |
| Temporal | 52 | 21 | MGN |
| Prefrontal | 20 | 34 | MD |

**7T Validation — T2* Iron Mapping (ses-06 multi-echo GRE, 0.67mm)**

Cross-session registration chain: ses-06 MEGRE → ses-04 7T T1w (rigid, same scanner) → ses-02 3T T1w (rigid, cross-scanner). T2* fitted from 7 echoes (5-35ms) via log-linear monoexponential.

| Region | lh T2* (ms) | rh T2* (ms) | Iron interpretation |
|---|---|---|---|
| Prefrontal | 24.2 ± 5.9 | 25.0 ± 5.4 | Highest iron |
| Premotor | 24.3 ± 7.1 | 25.2 ± 5.7 | High |
| Primary Motor | 25.1 ± 4.2 | 25.1 ± 3.7 | Moderate |
| Somatosensory | 26.0 ± 3.6 | 24.8 ± 3.2 | Variable |
| Post. Parietal | 26.4 ± 3.6 | 25.2 ± 3.4 | Lower |
| Temporal | 27.0 ± 4.9 | 25.6 ± 3.9 | Low |
| Occipital | 27.9 ± 3.2 | 24.6 ± 2.8 | Lowest (lh) |

T2* range is narrow (24-28ms). Anterior-to-posterior gradient broadly consistent with known iron distribution but the prefrontal (MD) showing shortest T2* is unexpected — may reflect registration imprecision or spatial blur from the 2mm→1mm warping chain. Laterality differences in occipital/somatosensory warrant investigation.

**TODO**: Improve registration quality (see below), repeat with subject-specific FNIRT nonlinear warp, sample T2* in native 7T space using inverse transforms.

### References
- McNabb CB et al. (2025) WAND: A multi-modal dataset. *Scientific Data* 12:220. DOI:10.1038/s41597-024-04154-7
- Stylianopoulou et al. (2025) Decoding brain structure-function dynamics. *Scientific Reports*. DOI:10.1038/s41598-025-24232-z
- Behrens et al. (2003) Non-invasive mapping of connections between human thalamus and cortex. *Nature Neuroscience* 6(7):750-757
- Johansen-Berg et al. (2005) Functional-anatomical validation of diffusion tractography-based segmentation of the human thalamus. *Cerebral Cortex* 15(1):31-39
- WAND GIN repository: https://gin.g-node.org/CUBRIC/WAND

---

## Planned Additions

- [ ] SPM-Python DCM oracle container
- [ ] M-DyNeMo (multi-scale DyNeMo)
- [ ] SINDy+Signature hybrid (interpretable DyNeMo replacement)
- [ ] Group-level comparison (multiple WAND subjects)
- [ ] Cam-CAN / OMEGA as additional datasets
- [ ] Neurodesk submission for osl-dynamics container
