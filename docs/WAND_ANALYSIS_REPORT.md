# Every Variable in the Model Has an Independent Measurement: Validating and Constraining Neurovascular Coupling with the WAND Multi-Modal Dataset

## Dataset

**WAND** (Welsh Advanced Neuroimaging Database) — 170 healthy volunteers, CUBRIC Cardiff.
Ultra-strong gradient (300 mT/m) Connectom scanner + CTF 275-channel MEG + TMS-EEG.

**Reference subject:** sub-08033 (all modalities confirmed, all raw data at `/data/raw/wand/`)
**Full dataset:** 170 subjects, all sessions, available at `/data/raw/wand/` on DGX Spark
**CUBRIC pre-computed derivatives:** eddy_qc (169 subjects), mriqc (~1783 scans), hd-bet (~1304 files)

| Session | Modalities | Key Data |
|---------|-----------|----------|
| ses-01 | MEG | CTF 275ch: resting, auditory-motor, MMN, simon, visual + headshape |
| ses-02 | DWI + anat | AxCaliber (4 shells, b=0–15500), CHARMED (AP+PA), T1w, VFA (SPGR/SSFP), QMT (16 volumes) |
| ses-03 | fMRI + anat | T1w, T2w (gradient-corrected), fieldmaps, perfusion (ASL) |
| ses-04 | MRS + anat | Magnetic resonance spectroscopy, T1w, fieldmaps |
| ses-05 | MRS + anat | Second MRS session, T1w |
| ses-06 | fMRI + anat | MP2RAGE, multi-echo GRE (7 echoes), T2w, fieldmaps |
| ses-08 | TMS | TMS-EEG SICI protocol (3 runs, .mat format) |

---

## Flagship Study: Unified TMS Intervention

### Core Thesis
Can we predict an individual's TMS response from their structural, functional, and metabolic brain architecture?

### Phase 1: Individual Brain Models
Build a digital twin per subject using all available modalities.

**Structural connectome:**
- FreeSurfer recon-all → cortical surfaces, parcellation
- DWI preprocessing (topup + eddy) → bedpostx → probtrackx2 → Cmat (streamline counts) + Lmat (fiber lengths)
- sbi4dwi AxCaliber/SANDI → axon diameter maps → ConnectomeMapper → microstructure-informed Dmat (conduction delays)

**Functional dynamics:**
- Resting MEG → source reconstruction → parcellated timeseries
- TDE + PCA (or TuckerTDE) → HMM/DyNeMo → baseline brain states
- State spectra → per-state power maps + coherence networks
- Summary statistics → fractional occupancy, lifetime, switching rate

**Metabolic context:**
- MRS → GABA and glutamate concentrations
- QMT → bound pool fraction (myelin content)
- T1w/T2w ratio → myelination proxy (validated against QMT)

**Status:** Processing running on DGX Spark for sub-08033:
- FreeSurfer recon-all: **Complete** (both T1ws + T2w)
- T1Prep (v0.3.1): **Complete** ses-02 (10m52s) + ses-03 (11m35s) — 3 upstream bugs found and patched
- fsl_anat: **Complete** ses-02, ses-03, ses-06
- bedpostx_gpu (3 fibers): **Complete** on CUBRIC CHARMED
- dtifit: **Complete** on CUBRIC eddy output
- MMORF T1w → Oxford-MM-1: **Complete** (147s)
- MMORF T1w+DTI → Oxford-MM-1: **Complete** (923s) — flagship multimodal registration
- QSMxT (MEGRE 6-echo): **Complete** — QSM Chimap, T2*/R2* maps at 0.67mm
- T2* fitting (7-echo GRE): **Complete**
- MRS sLASER fitting (fsl_mrs): **Complete** — 4 VOIs, NAA=15.8mM CRLB 2.1%
- MRS MEGA-PRESS editing: **Complete** — 4 VOIs, GABA/NAA 0.73-1.30
- DWI eddy (AxCaliber, local): Running

### Phase 2: TMS-EEG Fitting
Fit whole-brain model to each subject's TMS-evoked potentials.

- TMSProtocol → stimulus injection into JR/RWW neural mass model
- Balloon-Windkessel BOLD hemodynamics
- Leadfield forward projection → sensor-space TEP
- Multi-modal loss: FC + FCD + sensor MSE + TEP waveform + GFP
- Gradient-based optimization (optax Adam through full simulation)
- Per-region heterogeneous parameters (RegionalParameterSpace)

**Fitted parameters per subject:** global coupling G, per-region excitability, conduction velocity scaling

**Status:** Code complete. Need WAND TMS-EEG loader and subject-specific BEM.

### Phase 3: Virtual Lesion Analysis
Decompose TMS-evoked responses into local vs. network contributions (Momi et al. 2023).

- For each region: zero out connectome connections → re-simulate → measure TEP change
- Contribution matrix: (n_regions, n_timepoints) — time-resolved impact of each region
- Local-to-network transition time: when does network feedback exceed local reverberation?
- Momi's finding: ~100ms post-TMS

**Key question:** Does the transition time vary across subjects? Does it correlate with structural connectivity strength, conduction velocity, or GABA concentration?

**Status:** Code complete (`virtual_lesion.py`). Ready to run on fitted models.

### Phase 4: Predictive Modeling
Cross-validated prediction of TMS outcomes from pre-TMS features.

**Features (per subject):**
- Structural: connectome density, clustering, small-world index, hub topology
- Microstructural: mean axon diameter, conduction velocity distribution, g-ratio
- Functional: HMM state occupancies, lifetimes, switching rates, state spectra
- Metabolic: MRS GABA, glutamate, QMT myelin

**Targets:**
- TEP amplitude (peak-to-peak GFP)
- TEP complexity (perturbational complexity index)
- Local/network transition time
- Fitted model parameters (G, regional excitability)

**Methods:** Ridge regression, K-fold CV, permutation-based feature importance

**Status:** Code complete (`prediction.py`). Need features from processed data.

### Phase 5: Publication Outputs
- Fitted vs. empirical TEP waveforms (R² across subjects)
- Virtual lesion contribution matrix (group heatmap)
- Local→network transition time vs. structural/metabolic predictors
- Cross-modal prediction accuracy (which features matter?)
- T1w/T2w vs. QMT myelin validation

---

## Additional Research Topics

### 1. Microstructure → Oscillation Frequency
**Hypothesis:** Individual alpha frequency correlates with mean thalamocortical conduction velocity (from AxCaliber diameter).

The Valdes-Sosa et al. (2026, NSR) ξ-αNET paper shows conduction delays shape alpha across the lifespan using T1w/T2w as a myelin proxy. WAND can do this with *actual* microstructure from AxCaliber.

**Data:** ses-02 AxCaliber → diameter → Hursh-Rushton velocity; ses-01 MEG → individual alpha frequency

### 2. MRS-Constrained Neural Mass Models
**Instead of fitting J_I and J_NMDA freely, constrain them from MRS:**
- J_I ∝ [GABA] (from ses-04/05 MRS)
- J_NMDA ∝ [Glutamate] (from ses-04/05 MRS)

Does this improve model fit? Reduce parameter degeneracy?

### 3. T1w/T2w Ratio Validation
**Question:** How good is the T1w/T2w proxy for myelin content?

WAND provides ground truth from QMT (bound pool fraction), multi-echo GRE T2*, and VFA T1 mapping. The comparison script (`08_advanced_freesurfer.sh`) computes the vertex-wise correlation between T1w/T2w and QMT across the cortical surface.

**Expected:** Strong correlation in cortex, significant bias in regions with high iron content (basal ganglia) where T2* confounds the ratio.

### 4. Tucker-TDE vs. Standard TDE+PCA
**Methods contribution:** Compare tensor decomposition (TuckerTDE) against flatten+PCA for HMM/DyNeMo brain state analysis on the same 170 subjects.

### 5. SINDy/Koopman vs. HMM/DyNeMo
**Cross-method comparison:** Do data-driven dynamical systems (SINDy eigenvalues, DMD frequencies, log-signatures) find the same brain states as generative models (HMM/DyNeMo)?

### 6. Disconnectome Simulation
**Clinical translation:** Given a hypothetical lesion, predict the functional deficit by modifying the connectome (using DeepDisconnectome or tractography-based disconnection) and simulating the consequence with neurojax.

### 7. 4-Way Microstructure Comparison
**Benchmarking:** DIPY vs. FSL vs. sbi4dwi vs. DMI.jl on WAND AxCaliber data. Which method best estimates axon diameter on ultra-strong gradient data?

### 8. FEM Head Mesh Comparison: brain2mesh vs SimNIBS charm vs MNE BEM
**Three-way comparison** of tetrahedral head meshing for TMS E-field and MEG/EEG forward modeling:

| Method | Tool | Tissues | Input | Strengths |
|---|---|---|---|---|
| brain2mesh | `pip install iso2mesh` (Fang, v0.5.5) | 5 | FreeSurfer segmentation → tissue prob maps | Customizable, open source, EEG 10-20 landmarks |
| SimNIBS charm | SimNIBS 4.x | 5 | T1w + T2w (raw) | DL-based, TMS-optimized, no FreeSurfer needed |
| MNE BEM | MNE-Python | 3 | FreeSurfer watershed surfaces | Standard for MEG/EEG, fast, well-validated |

Additional brain-tissue segmentation tools for comparison:

| Tool | Method | Tissues | Speed | Where to run | Install |
|---|---|---|---|---|---|
| **T1Prep** (Gaser) | AMAP + DeepMriPrep | GM/WM/CSF + thickness + lesions | Fast (DL) | **DGX Spark** (GPU) | GitHub, Python+C |
| **GRACE** | MONAI U-Net | 11 tissues incl. cortical/cancellous bone | Fast (DL) | **DGX Spark** (GPU, MONAI) | GitHub |
| **FastSurfer** | FastSurferCNN | 95 brain classes | < 1 min GPU | **DGX Spark** (GPU) | GitHub, PyTorch |
| **SynthSeg** | Contrast-agnostic CNN | 40 brain structures | 6s GPU / 2min CPU | Local or DGX | FreeSurfer 8.2.0 |
| **deepmriprep** | Neural networks | GM/WM/CSF | 37x faster than CAT12 | **DGX Spark** (GPU) | `pip install deepmriprep` |
| **FSL FAST** | Hidden Markov RF + EM | GM/WM/CSF | Minutes | Local (CPU) | FSL 6.0.7 |
| **SimNIBS charm** | DL atlas | 10 tissues incl. compact/spongy bone | ~30 min | Local or DGX | `pip install simnibs` |
| **brain2mesh** | iso2mesh/TetGen | 5 tissues (approx skull) | ~10 min | Local (CPU) | `pip install iso2mesh` |
| **MNE BEM** | FreeSurfer watershed | 3 shells | ~5 min | Local (CPU) | MNE-Python |
| **FSL BET+betsurf** | BET mesh + betsurf | 4 surfaces (brain, inner/outer skull, scalp) | ~5 min | Local (CPU) | FSL 6.0.7 (`bet2 -e -A`) |

**DGX Spark workload** (batch 170 subjects): T1Prep, GRACE, FastSurfer, deepmriprep, bedpostx_gpu, DyNeMo training, TMS fitting.
**Local workload** (single subject validation): FSL FAST, SynthSeg, charm, brain2mesh, MNE BEM, recon-all.

**T1Prep** (github.com/ChristianGaser/T1Prep) is particularly notable: it's CAT12 rewritten in Python by the same author (Christian Gaser), using DeepMriPrep for DL-accelerated processing with AMAP tissue segmentation. BIDS-native output, Apache 2.0 license. Provides cortical thickness maps and lesion detection alongside standard tissue probability maps.

**GRACE** is the only alternative to SimNIBS charm that segments cortical + cancellous bone separately — critical for accurate TMS E-field modeling through the skull.

WAND provides ground truth for validating tissue boundaries: QMT (WM/GM contrast), T2* (CSF boundaries), T1w+T2w (skull). Script: `17_brain2mesh_comparison.sh`.

**Note on iso2mesh**: The Python package (`pip install iso2mesh`, `NeuroJSON/pyiso2mesh`) is a native Python reimplementation (not MATLAB wrapper) but still auto-downloads external CGAL and TetGen binaries. CGAL also available via `brew install cgal`. The `brain2mesh()` function expects 5 tissue probability maps (CSF, GM, WM, bone, scalp) — FreeSurfer `aparc+aseg` needs conversion to these, with approximate skull/scalp from morphological dilation.

### 9. Pseudo-CT Validation Against Quantitative MRI
**Question:** Can WAND's quantitative structural data validate (or replace) pseudo-CT inference for skull conductivity modeling?

**The pseudo-CT problem:** TMS/tDCS E-field accuracy depends on skull conductivity, which requires CT-like bone density estimates. The Plymouth tool (`sitiny/mr-to-pct`, Yaakub et al. 2023, Brain Stimulation) uses a MONAI U-Net to predict pseudo-CT from T1w MRI. SimNIBS charm uses DL tissue segmentation to distinguish compact/spongy bone without generating CT. BabelBrain uses atlas-based pseudo-CT. sbi4dwi already has a `PseudoCTMapper` module implementing the Plymouth and BabelBrain approaches with Archie's Law for conductivity estimation.

**What WAND provides as ground truth:**

| qMRI acquisition | Measures | Pseudo-CT proxy for |
|---|---|---|
| QMT bound pool fraction (ses-02) | Macromolecular content in bone | Bone mineral density → HU |
| VFA quantitative T1 (ses-02) | T1 relaxation in skull | Bone density (shorter T1 = denser) |
| Multi-echo GRE R2* (ses-06) | Susceptibility/mineral content | Cortical bone density |
| MP2RAGE quantitative T1 (ses-06) | Independent T1 estimate | Cross-validates VFA in bone |

**Analyses:**
1. Generate pseudo-CT from T1w using Plymouth DL model → compare predicted HU against QMT/VFA-derived bone density
2. Compare charm's compact/spongy bone segmentation boundaries against QMT-derived density gradient
3. Calibrate sbi4dwi `PseudoCTMapper` Archie's Law parameters (porosity exponent, brine conductivity) using QMT porosity estimates
4. Test: **Can qMRI replace pseudo-CT entirely?** If QMT gives bone density directly, the DL prediction step is unnecessary
5. Compare E-field simulations using each skull model: charm segmentation vs Plymouth pseudo-CT vs qMRI-derived conductivity

**Potential finding:** qMRI-based skull conductivity may be more accurate than pseudo-CT because it measures the actual physical property rather than predicting it from T1 contrast. This would establish WAND-type acquisitions as the gold standard for TMS planning.

### 10. Simulation-Based MRS Lipid Contamination Modeling
**Question:** Can subject-specific head geometry + MR physics simulation predict and correct lipid contamination in WAND's sLASER MRS?

**The problem:** SVS voxels near cortex (especially sensorimotor, auditory VOIs) are contaminated by scalp fat and skull marrow fat via chemical shift displacement error (CSDE) and imperfect slice profiles. This biases GABA/glutamate estimates — the metabolites we need for constraining neural mass models.

**WAND provides the geometry and tissue properties:**
- `betsurf` / `charm`: scalp fat and skull marrow boundaries
- QMT: macromolecular content in bone (fat fraction)
- VFA T1: tissue-specific T1 (fat ~300ms vs muscle ~1400ms)
- Multi-echo GRE: fat/water separation via chemical shift effects
- `svs_segment` voxel mask: exact voxel geometry in T1 space

**Simulation approach (building on prior MIDAS/MCMRSimulator work):**
1. Build subject-specific multi-compartment phantom from charm/betsurf segmentation with tissue-specific T1/T2/chemical shift from qMRI
2. Simulate sLASER sequence spatial response using MCMRSimulator.jl (~/dev/mcmrsimulator.jl) or KomaMRI.jl for Bloch-based slice profile + CSDE
3. Use fsl_mrs_sim for metabolite + lipid basis spectra (J-coupling, chemical shifts)
4. Predict lipid contamination per VOI per subject
5. Correct fsl_mrs fitting results → improved GABA/Glu quantification
6. Test: does correction improve ses-04 ↔ ses-05 test-retest reliability?

**Tools:** MCMRSimulator.jl (spatial/Bloch), fsl_mrs_sim (spectral basis), KomaMRI.jl (GPU-accelerated alternative), FID-A/Spinach (J-coupling if needed).

**Acquisition-side reality:** Even with sLASER (best SVS slice profiles via adiabatic refocusing), fat suppression quality varies per scan due to coil loading, B0 shim quality, voxel placement distance from scalp, VAPOR water suppression interaction with short-T1 lipids, and operator-dependent OVS band placement. WAND's 4 VOIs (sensorimotor, auditory near skull; occipital deeper; ACC midline) will show different contamination profiles. The simulation predicts which VOIs to trust and quantifies repositioning-driven artifacts between ses-04 and ses-05.

#### Architecture: "POSSUM for MRS"

No existing tool does the full pipeline. The architecture combines three layers:

```
Layer 1: Spatial Response Engine (MCMRSimulator.jl or KomaMRI.jl)
  Input: sLASER RF pulse waveforms + gradient timings
  Physics: 3D Bloch equations
  Output: S(x,y,z,Δf) — spatial response per chemical shift offset
  → Captures: slice profiles, CSDE, OVS band effects

Layer 2: Spectral Basis Engine (fsl_mrs_sim or FID-A or VESPA)
  Input: metabolite spin systems + pulse sequence
  Physics: density matrix with J-coupling
  Output: B_m(t) — basis spectrum per metabolite per spatial position
  → Captures: J-evolution, sequence-specific phase/amplitude

Layer 3: Phantom + Integration Engine (THE MISSING PIECE)
  Input: segmented head (charm/betsurf) + qMRI tissue properties
  Integration:
    Signal(t) = Σ_xyz [ S(x,y,z) · ρ(x,y,z) ·
                  Σ_m C_m(x,y,z) · B_m(t,x,y,z) ·
                  exp(-t/T2*(x,y,z)) · exp(i·Δω(x,y,z)·t) ]
  Output: NIfTI-MRS format realistic spectrum
  → Captures: lipid contamination, partial volume, B0 inhomogeneity
```

#### MRS Simulation Tool Landscape

| Tool | Language | Spatial | J-coupling | Multi-tissue phantom | Role |
|---|---|---|---|---|---|
| **FID-A** | MATLAB | 1D-3D | Yes | No | Basis spectra (gold standard) |
| **fsl_mrs_sim** | Python | Limited | Yes | No | Basis spectra (in our pipeline) |
| **Spinach** | MATLAB | Yes | Yes (best) | Partial | Complex spin systems (GABA) |
| **VESPA/MARSS** | Python | Yes | Yes | No | Python basis spectra |
| **MCMRSimulator.jl** | Julia | Yes (3D MC) | No | Yes (mesh) | Spatial engine (at ~/dev/mcmrsimulator.jl) |
| **KomaMRI.jl** | Julia | Yes (3D GPU) | No | Yes | Spatial engine (GPU-accelerated) |
| **FSL-MRS** | Python | Fitting only | N/A | Yes (fitting) | Fitting / validation |
| **OSPREY** | MATLAB | Fitting only | N/A | Yes (fitting) | Fitting / validation |
| **POSSUM** | C++ (FSL) | Yes (imaging) | No | Yes | Architectural model (fMRI, not MRS) |

**Primary gap:** The integration layer (Layer 3) that loads a segmented head, assigns tissue-specific metabolite concentrations + lipid resonances (0.9, 1.3 ppm) at anatomically correct locations, and performs spatially-weighted spectral integration. This is novel work — a JAX implementation would enable GPU acceleration and differentiability (backpropagate through simulation for fitting).

**Connection to MIDAS:** The whole-brain MRSI approach (Rivera et al. 2024, PMC11614968) processed 118,922 voxels through MIDAS. The lipid ring contamination in EPSI is even more severe than in SVS. A simulation-based correction calibrated on WAND's qMRI ground truth could improve metabolite quantification across the entire MRSI volume.

### 11. Perfusion Imaging Suite (pCASL, TRUST, Angiography)
WAND ses-03 includes a complete cerebrovascular imaging battery:

| Acquisition | Measures | Data | Processing |
|---|---|---|---|
| **pCASL** | Cerebral blood flow (CBF, ml/100g/min) | 64×64×22, 110 vols, PLD=2.0s | FSL `oxford_asl` / BASIL |
| **TRUST** | Venous oxygen saturation (SvO₂) via sagittal sinus blood T2 | 64×64×1, 24 vols, TI=1.02s | Custom T2 fitting (Lu et al. 2008) |
| **Inversion Recovery** | Blood T1 for absolute CBF calibration | 128×128×1, 960 TIs | Multi-TI T1 fitting |
| **Phase-contrast angiography** | Feeding artery flow at pCASL labeling plane | 384×512×60, z=[-105,-28]mm (neck) | Total CBF from ICA+VA flow; labeling efficiency QC |
| **M0 scan** (AP direction) | Equilibrium magnetization for ASL quantification | 64×64×22, 3 vols | Part of oxford_asl pipeline |

**Relevance to TMS study:**
- **CBF at M1** as predictor of TMS excitability (higher perfusion → more metabolically active cortex)
- **SvO₂ → OEF → CMRO₂** (metabolic rate = CBF × OEF × CaO₂) connects to MRS metabolites
- **Vascular anatomy** informs neurovascular coupling model (Riera et al. 2006/2007)
- **CBF + MRS + fMRI BOLD** provides three independent hemodynamic/metabolic measures of the same tissue

**Processing:** FSL `oxford_asl` / BASIL: pCASL + M0 + structural T1 → absolute CBF map. InvRec data (16 blocks × 60 TIs) provides subject-specific blood T1 via block-averaged IR fitting. TRUST gives global OEF via structural-guided sagittal sinus ROI (T1w brain surface projection) and per-voxel T2 quality control. Phase-contrast angiography is acquired at the pCASL labeling plane (neck, z≈-60mm) to (a) plan/verify labeling slab position over the feeding arteries, (b) compute total cerebral blood flow from ICA+VA flow velocities × cross-sectional area as an independent CBF cross-validation, and (c) assess labeling efficiency by comparing PC-derived total flow with ASL-derived whole-brain CBF.

**CMRO₂ hierarchy (vpjax):**

| Level | OEF source | Spatial resolution | Method |
|---|---|---|---|
| Global | TRUST (sagittal sinus T2) | Whole-brain average | Lu 2008 |
| **Regional** | **qBOLD (7-echo GRE, ses-06)** | **Per-voxel** | **He & Yablonskiy 2007, Bulte (Oxford)** |
| Dynamic | Riera neurovascular model | Time-varying | Riera 2006/2007 |

The critical upgrade: WAND's 7-echo GRE enables **quantitative BOLD (qBOLD)** — fitting a biophysical model to separate R2' (reversible, from deoxygenated blood) from R2 (irreversible, from tissue), giving per-voxel OEF and deoxygenated blood volume (DBV). Combined with pCASL CBF:

`CMRO₂(regional) = CBF(regional) × OEF(regional) × CaO₂`

This replaces the uniform global OEF from TRUST with spatially resolved oxygen extraction — a significant accuracy improvement, especially for cortical regions near large veins where OEF varies substantially. Implementation in vpjax (`vpjax/qbold/`).

### 12. Cortical Layer Analysis + QSM (sub-mm WAND data)
**Discovery:** WAND ses-06 structural data is **sub-millimeter** — MEGRE at 0.67mm, MP2RAGE at 0.7mm, T2w at 0.21mm in-plane. This enables cortical depth-dependent analysis via LAYNII or Nighres (~2-3 depth bins across the cortical ribbon).

**Untapped resource:** The 7-echo GRE **phase data** (`*_part-phase_MEGRE.nii.gz`) is currently unused. Processing with QSMxT (BIDS-aware, Neurodesk) gives **Quantitative Susceptibility Mapping (QSM)** — separating paramagnetic iron from diamagnetic myelin at each voxel.

**Layer-resolved analyses enabled:**

| Map | Acquisition | Per-layer measurement |
|---|---|---|
| T1 | MP2RAGE (0.7mm) | Myelination gradient (short T1 = more myelin) |
| R2* | MEGRE magnitude (0.67mm) | Combined iron + myelin |
| **QSM** | **MEGRE phase (0.67mm)** | **Iron (+) vs myelin (−) SEPARATED** |
| BPF | QMT (ses-02) | Direct myelin content |
| T1w/T2w | ses-03 | Proxy — validate per layer vs QMT |

**Iron-myelin decomposition per cortical layer** from R2* + QSM + BPF — no single contrast can achieve this. This connects to the feedforward (deep layers) / feedback (superficial layers) hierarchy in Valdes-Sosa's ξ-αNET and the geodesic cortical flow (Liu et al. 2026).

**Tools:** LAYNII (Huber et al. 2021, volume-based layers), Nighres (Python, level-set layers), QSMxT (BIDS QSM pipeline). All via Neurodesk. Implementation in vpjax (`vpjax/layers/`, `vpjax/qsm/`).

### 13. MEG Preprocessing via osl-ephys (Oxford Pipeline)

**osl-ephys** (`github.com/OHBA-analysis/osl-ephys`) is the preprocessing and source reconstruction tool that feeds into osl-dynamics (HMM/DyNeMo). It provides the full MEG pipeline from raw data to parcellated source timeseries:

```
WAND CTF .ds → osl-ephys
  ├── preprocessing: ICA artifact rejection (ica_label), filtering, downsampling
  ├── source_recon: RHINO coregistration → beamformer or minimum_norm
  ├── parcellation: project to atlas + sign_flipping across subjects
  └── report: HTML QC per subject
→ parcellated .npy → neurojax Data() → TDE+PCA → HMM/DyNeMo
```

Key modules: `rhino/` (Registration of Head Images to mNe Objects), `beamforming.py`, `minimum_norm.py`, `parcellation.py`, `sign_flipping.py`, `ica_label.py`, `batch.py` (multi-subject pipeline), `report/` (HTML QC).

For WAND: MaxFilter not needed (CTF system, not Elekta). Install via `pip install osl-ephys` or from source. Needs FreeSurfer SUBJECTS_DIR (done: sub-08033 recon-all complete) + MNE-Python.

**Key findings from the osl-ephys paper (van Es, Gohil, Quinn, Woolrich — Frontiers in Neuroscience 2025):**

- **Volumetric source reconstruction** (FSL-based): Unique to osl-ephys, not available in standard MNE-Python. Uses FSL's volumetric parcellation rather than FreeSurfer surfaces. Since we have FSL installed, this is a low-friction option alongside our neurojax surface-based beamformers.
- **Config-based pipeline API**: Processing workflows specified as Python dictionaries rather than custom scripts. Standardizes parameter documentation and enables pipeline sharing. Model for our WAND scripts.
- **Dask parallelization**: 175 min → 21 min on 16 cores for batch processing. Critical for scaling to 170 WAND subjects on DGX.
- **Sign ambiguity resolution**: `sign_flipping.py` resolves the arbitrary sign of beamformer/MNE source amplitudes across subjects — prerequisite for group-level HMM/DyNeMo.
- **Parcel time-course orthogonalization**: Colclough et al. method to remove zero-lag correlations from spatial leakage, ensuring connectivity estimates reflect genuine coupling rather than source reconstruction artifacts.
- **Generalized ESD test**: Automated statistical outlier detection for bad channels/segments, more principled than fixed-threshold approaches.

| Feature | osl-ephys | MNE-Python | neurojax |
|---------|-----------|-----------|----------|
| Config-based pipeline | Yes | No | Scripts |
| Volumetric source recon | FSL-based | No | No |
| Surface source recon | FreeSurfer | FreeSurfer | JAX native |
| Parallel processing | Dask | Not built-in | jax.vmap |
| Sign flipping | Yes | No | Needed |
| HTML reports | Automatic | Manual | fsreport |

osl-ephys provides the validated Oxford defaults for the same analysis neurojax reimplements in JAX — use as reference/oracle pipeline for comparison, like qMRLab for qMRI fitting.

### 14. FOOOF/specparam Aperiodic + Periodic Decomposition
**Per-subject spectral parameterization** using FOOOF (Fitting Oscillations & One Over F) on MEG power spectra:

- **Aperiodic component** (1/f slope + offset): reflects excitation-inhibition balance. Steeper slope → more inhibition-dominated. Connects directly to MRS GABA/glutamate concentrations (ses-04/05).
- **Periodic peaks** (centre frequency, power, bandwidth): individual alpha frequency (IAF), beta peak, theta peak per region. IAF connects to AxCaliber conduction velocity (Valdes-Sosa ξ-αNET).
- **Per-state FOOOF**: Run FOOOF on the state-specific power spectra from HMM/DyNeMo. Each brain state gets its own aperiodic slope + peak parameters. This reveals: "Does state 3 have a steeper 1/f slope (more inhibition)? Is its alpha peak shifted?"

**Pipeline:**
```
Resting MEG → source reconstruction → parcellated (68 or 148 regions)
  → Per-region PSD (multitaper)
  → FOOOF: aperiodic (offset, exponent) + peaks (frequency, power, bandwidth)
  → Per-state: state_spectra.py PSD → FOOOF per state
  → Features: IAF, aperiodic exponent, peak power per band per region
  → Prediction model (Phase 4): do FOOOF features predict TMS response?
```

**Cross-modal validation:**
- Aperiodic exponent vs. MRS GABA/glutamate ratio (E/I balance)
- IAF vs. AxCaliber conduction velocity (speed → resonance frequency)
- Aperiodic exponent vs. cortical thickness (thicker cortex → more layers → steeper 1/f)
- Peak power vs. QMT myelin (more myelin → faster oscillations → different peak structure)

**Implementation:** `specparam` (formerly `fooof`) Python package, or native JAX reimplementation for differentiability. Per-region, per-state, per-subject → feature matrix for prediction.

**Alternative: ξ-π nonparametric spectral decomposition** (Hu et al., IEEE TBME 2024): A nonparametric alternative to FOOOF that uses penalized Whittle likelihood + shape language modeling in an EM framework. Avoids FOOOF's parametric assumptions about 1/f shape. Validated on sleep EEG and large-sample iEEG with significantly lower MSE and more accurate peak detection than FOOOF/IRASA/BOSC. Open-source: `github.com/ShiangHu/Xi-Pi`. Should run both FOOOF and ξ-π on WAND data and compare — particularly interesting for state-specific spectra where the aperiodic shape may deviate from power-law.

### 15. HIGGS: One-Step Source + Connectivity Estimation

**Hidden Gaussian Graphical Spectral (HIGGS) model** (Valdes-Sosa et al., Sci. Reports 2023): A one-step Bayesian approach that simultaneously estimates source activity and functional connectivity from M/EEG, avoiding the circularity of traditional two-step methods (source recon → connectivity).

**Why this matters:** Standard pipelines run source imaging (MNE/beamformer) then estimate connectivity from the source timeseries. HIGGS shows this two-step approach introduces systematic errors because the source reconstruction is uninformed about the actual connectivity structure. HIGGS jointly estimates both, achieving <2% error vs ~20% for state-of-the-art two-step methods.

**Technical approach:**
- Models brain oscillations as complex-valued Gaussian graphical model with Hermitian precision matrix Θ(ιι,f)
- Integrates electromagnetic forward model (leadfield L) directly into the estimation
- EM algorithm alternates between updating source estimates and connectivity
- Hermitian graphical LASSO (hgLASSO) for sparse connectivity with statistical guarantees
- Unbiased estimation via debiasing procedure with Rayleigh-distributed test statistics

**Validation:** Macaque simultaneous EEG/ECoG recordings provide experimental confirmation with 1/3 times larger congruence (Riemannian distance) than multistep methods.

**Connection to neurojax:** HIGGS is the natural evolution of VARETA (already implemented in `source/vareta.py`). Both are from Valdes-Sosa's group. HIGGS adds the Hermitian GGS connectivity model on top of the source localization. Implementation target for neurojax — would use our existing leadfield module + JAX sparse solvers.

### 16. Dynamic Mode Time-Frequency Representation (DTFR)

(Hu et al., IEEE TBME 2025): Treats multivariate signals as a dynamical system, identifies fundamental dynamic modes with mutual orthogonality, then projects data onto modes for time-frequency representation. Outperforms traditional TF approaches (wavelets, STFT, Hilbert) for EEG disease discrimination.

**Connection to neurojax:** Complements our TDE (time-delay embedding) approach. Where TDE captures temporal structure via delay-embedded covariance, DTFR captures it via dynamic mode decomposition — related to Koopman operator theory (which we already have in jaxctrl). The dynamic modes could serve as a basis for state identification, complementing HMM/DyNeMo.

### 17. Automatic Lead Field QC (LF-AQI)

(Valdes-Sosa et al., NeuroImage 2023): Automated quality control index for EEG lead fields computed from individual MRIs. Compares realistic head model lead fields against reference models (homogeneous/spherical) via correlation-based similarity. Validated on 1,251 subjects (Child Mind Institute). Cohen's d = 1.3.

**Connection to neurojax:** Our `monitors/leadfield.py` and `source/` modules compute forward models — LF-AQI provides automatic validation. Should implement as a QC step before source reconstruction. Threshold: 90th percentile cutoff (-0.9755) for flagging problematic lead fields.

---

## Key Design Decisions

### 1. Spatial Anchor: ses-02 T1w
All sessions register TO the ses-02 T1w because that is the DWI session. The structural connectome (Cmat/Dmat) defines the spatial reference frame for whole-brain simulation — everything else aligns to it. Cross-session registration uses FLIRT 6-DOF rigid body (same subject, different days).

### 2. Native Space First
All processing stays in native (subject) space as long as possible. Standard space transforms are computed and stored but not applied until needed for group analysis. This avoids interpolation artifacts in microstructure maps, preserves individual anatomical detail for source reconstruction, and keeps the connectome in the space where it was estimated.

### 3. Registration: SynthMorph (DL) + FNIRT (classic) + FSL DL Reg (future)
**FreeSurfer `mri_synthmorph`** (SynthMorph, Hoffmann et al. 2022) is now the primary registration tool for cross-session and cross-contrast alignment. Unlike FLIRT/FNIRT which assume similar contrast between images, SynthMorph is **contrast-agnostic** — trained on synthetic images to learn anatomy, not intensity. This is critical for WAND where each session has different contrasts:
- ses-02 T1w ↔ ses-03 T1w: trivial (same contrast)
- ses-02 QMT ↔ ses-03 T1w: **different contrast** — FLIRT fails, SynthMorph handles it
- ses-06 MP2RAGE ↔ ses-02 T1w: **very different contrast** — SynthMorph handles it
- ses-06 multi-echo GRE ↔ ses-02 T1w: **different contrast** — SynthMorph handles it

**Registration modes available:**
- `joint`: affine + deformable in one symmetric pass (default, best quality)
- `affine`: linear alignment with `-t aff.lta` output
- `rigid`: 6-DOF rigid body (cross-session same-subject)
- `deform`: nonlinear only (after affine initialization)

**FSL MMORF** (MultiModal Registration Framework, FSL 6.0.7+) is the new nonlinear registration tool that **simultaneously aligns multiple modalities** — critically, it handles **DTI tensor registration** alongside scalar images. Uses SPRED regularization (log Jacobian singular values) and bias field estimation. This is the right tool for registering AxCaliber DWI to T1w while preserving tensor orientation — neither FNIRT nor SynthMorph can do this.

**Five-way registration comparison:**

| Tool | Method | Best for | Install |
|---|---|---|---|
| **ANTs SyN** | Diffeomorphic (CC/MI metric) | Gold standard accuracy, cortical alignment | `brew install ants` or `pip install antspyx` |
| **SynthMorph** | DL inference (contrast-agnostic) | Cross-contrast (QMT↔T1w, MP2RAGE↔T1w), fast | FreeSurfer 8.2.0 |
| **MMORF** | Iterative multimodal (SPRED) | DWI tensor → T1w simultaneous alignment | FSL 6.0.7 |
| **FLIRT+FNIRT** | Iterative (bending energy) | Standard baseline, widely validated | FSL 6.0.7 |
| **FSL DL Reg + OM-1** | DL (planned) | Standard space with new template | When available |

Each tool has a distinct strength for WAND:
- **ANTs SyN**: best cortical registration accuracy (proven in many benchmarks). MI metric handles cross-contrast. Diffeomorphic guarantees topology preservation. ANTsPy for Python integration.
- **SynthMorph**: fastest (~seconds), no preprocessing, any contrast pair. Best for quick cross-session alignment.
- **MMORF**: unique tensor registration capability for DWI AxCaliber → T1w. Simultaneous multimodal alignment preserves tensor orientation.
- **FLIRT+FNIRT**: reproducible baseline. Widely used, easy to compare against published results.

All transforms stored in `fsl-reg/` for comparison. The registration comparison across 170 subjects (which tool gives best cortical alignment? best subcortical? best white matter tract overlap?) is itself a methods contribution.

### 4. MRS Quantification via fsl_mrs + mrs-jax

WAND MRS: sLASER (ses-04, 7T TE=78ms) + MEGA-PRESS (ses-05, 3T TE=68ms) with water references in 4 brain regions per session:
- Anterior cingulate cortex (ACC) — prefrontal GABA/glutamate
- Occipital cortex — visual area baseline
- Right auditory cortex — sensory processing
- Left sensorimotor cortex — motor excitability

**Pipeline (validated on DGX Spark for sub-08033):**

1. **TWIX → NIfTI-MRS**: `spec2nii` (PatientName bug fixed upstream) or native `mrs-jax.read_twix()`
2. **Preprocessing**: SVD coil combination (32 coils → 1), spectral registration alignment, MAD outlier rejection
3. **sLASER fitting** (ses-04): `fsl_mrs` with WIN-MRS-Basis-Sets sLASER basis (GOIA-WURST, TE=78ms, 19 metabolites simulated via `fsl_mrs_sim`)
4. **MEGA-PRESS editing** (ses-05): `mrs-jax` MEGA-PRESS pipeline — paired FPC alignment, edit-ON/OFF subtraction, GABA Gaussian fitting at 3.0 ppm
5. **Quantification**: Water-referenced tissue-corrected concentrations (Gasparovic 2006)

**Results (sub-08033):**

| Session | Sequence | VOI | NAA (mM) | GABA | GABA/NAA | CRLB |
|---------|----------|-----|----------|------|----------|------|
| ses-04 | sLASER 7T | ACC | 15.8 | 3.7 mM | — | 2.1% |
| ses-04 | sLASER 7T | Occ | 26.6 | 24.3 mM* | — | 1.3% |
| ses-05 | MEGA 3T | ACC | — | — | 0.73 | — |
| ses-05 | MEGA 3T | Occ | — | — | 1.30 | — |
| ses-05 | MEGA 3T | Aud | — | — | 1.08 | — |
| ses-05 | MEGA 3T | SM | — | — | 0.94 | — |

*sLASER absolute concentrations for non-ACC VOIs need sub-echo timing calibration.

**mrs-jax** (github.com/m9h/mrs-jax): Standalone package extracted from neurojax — 97 tests, validated on Big GABA (12 Siemens subjects), ISMRM Fitting Challenge (28 spectra), NIfTI-MRS standard, and WAND. Features: MEGA-PRESS + HERMES editing, JAX backend (jit/vmap/grad), Philips SDAT + LCModel I/O, QC reports, and a differentiable MRSI simulator for whole-brain metabolic mapping.

Two sessions (ses-04, ses-05) enable test-retest reliability of metabolite quantification.

### 5. Microstructure-Informed Connectome (Not Just Streamline Counts)
Standard connectomes use streamline count for weights and fiber_length/7 m/s for delays. WAND enables a better approach:
- **Cmat weights**: ConnectomeMapper samples microstructural metrics (ICVF, axon diameter) along each streamline
- **Dmat delays**: Hursh-Rushton velocity from AxCaliber diameter per tract segment: `delay = Σ(segment_length / local_velocity)`
- This gives connection-specific conduction delays rather than a single global velocity

### 6. T1w/T2w Ratio as Myelin Proxy — Validated Against Ground Truth
The Valdes-Sosa ξ-αNET paper uses T1w/T2w ratio to infer cortical hierarchy and constrain conduction delays. WAND uniquely allows validating this proxy against:
- **QMT bound pool fraction** (ses-02) — direct measurement of macromolecular (myelin) content
- **Multi-echo GRE T2*** (ses-06) — sensitive to myelin and iron
- **VFA T1 mapping** (ses-02) — quantitative T1 relaxation
- **AxCaliber g-ratio** (from diameter + myelin thickness via sbi4dwi)

This comparison is a standalone publication: "Validating the T1w/T2w myelin proxy against quantitative MRI in a large healthy cohort."

### 7. Four-Way Microstructure Comparison
The AxCaliber data (4 shells, b=0-15500 s/mm², 264 volumes on 300 mT/m) is processed through four independent estimation pipelines:
- **DIPY**: DTI, DKI, MAPMRI, Free Water DTI (Python community standard)
- **FSL**: dtifit + NODDI via AMICO
- **sbi4dwi/dmipy-jax**: JAX-accelerated cylinder models + simulation-based inference
- **DMI.jl**: Julia SANDI/AxCaliber

This benchmarks methods on data that was specifically acquired for microstructure estimation, unlike most comparisons that use standard clinical DWI.

### 8. Advanced FreeSurfer Beyond recon-all
FreeSurfer 8.2.0 provides tools that most labs don't use:
- **SynthSeg**: DL segmentation that works on any contrast — enables consistent parcellation across all WAND sessions despite different acquisition protocols
- **Thalamic nuclei with DTI**: Uses DWI to improve thalamic subnucleus boundaries — critical for thalamocortical conduction velocity estimation
- **SAMSEG**: Jointly segments T1+T2 (or any combination of contrasts) — better than running FAST on T1 alone
- **SynthStrip**: DL skull stripping — more robust than BET for the diverse contrasts in WAND

### 9. Neurodesk Containers for All Processing
All neuroimaging tools run via **Neurodesk containers with explicit version pinning** on all systems (local Mac, DGX Spark, collaborator machines). This ensures:
- **Reproducibility**: exact same tool version and dependencies everywhere
- **No build headaches**: no compiling C++ (QUIT), no MATLAB licenses (qMRLab via Octave), no dependency conflicts
- **Version tracking**: container tags recorded in `dataset_description.json` per derivatives directory
- **Portability**: same pipeline runs on laptop, HPC, and cloud

Key containers for WAND processing:

| Tool | Container | Use |
|---|---|---|
| FSL | `fsl/6.0.7` | DWI preproc, bedpostx, xtract, fsl_mrs, MMORF |
| FreeSurfer | `freesurfer/8.2.0` | recon-all, SynthSeg, SynthMorph, thalamic nuclei |
| ANTs | `ants/2.5.x` | SyN registration, DiReCT cortical thickness |
| QUIT | `quit/3.4` | All qMRI fitting (T1, T2, MWF, BPF, T2*, MP2RAGE) |
| qMRLab | `qmrlab/2.5.x` | QMT two-pool fitting (Sled-Pike model) |
| hMRI | `hmri/x.x` | R2* with B1/impurity correction |
| SimNIBS | `simnibs/4.x` | charm head model, TMS E-field simulation |
| FastSurfer | `fastsurfer/2.x` | Fast DL segmentation (GPU) |
| CAT12/T1Prep | `cat12/x.x` | AMAP segmentation, cortical thickness |
| DIPY | `dipy/1.x` | Diffusion processing comparison |
| MRtrix3 | `mrtrix3/3.x` | CSD tractography, tckgen, connectome |
| iso2mesh | `iso2mesh/x.x` | brain2mesh FEM head modeling |

Processing scripts call containers with data mounted from BIDS directories. The container version is the single source of truth — no local installs needed except neurojax/sbi4dwi (JAX ecosystem, developed locally).

### 10. QSIPrep + fMRIPrep for Structure-Function Coupling
**QSIPrep** and **fMRIPrep** (both via Neurodesk) provide BIDS-structured parcellated outputs across multiple atlases, enabling systematic structure-function coupling analysis:

| Tool | Produces | Atlases |
|---|---|---|
| **QSIPrep** | Preprocessed DWI + structural connectivity matrices | Schaefer (100-1000), Glasser, Gordon, Brainnetome, AAL, DK, Destrieux |
| **fMRIPrep** | Preprocessed BOLD + parcellated timeseries + confounds | Same atlases via TemplateFlow |

**Analysis:** For each atlas, compute structural connectivity (QSIPrep Cmat) and functional connectivity (fMRIPrep parcellated BOLD → correlation/partial correlation/tangent), then:
1. Edge-wise structure-function correlation: where does anatomy predict function?
2. Regions where SC-FC coupling is weak → dynamics (HMM state-switching) explains the gap
3. Multiple parcellation resolutions (100→1000) reveal scale-dependence of coupling
4. Seed-based connectivity from fMRI compared against tractography-based connectivity from DWI

Both tools standardize the parcellation step — same atlas, same registration — so the comparison is clean. This is more systematic than running MELODIC + probtrackx2 separately with different parcellations.

**WAND advantage:** ses-03 resting fMRI (3T) + ses-06 resting fMRI (7T) enables test-retest of FC, while ses-02 DWI (300 mT/m AxCaliber) provides the highest-quality structural connectivity available.

### 11. BIDS Derivatives Compliance
Each processing stage produces its own derivatives directory with `dataset_description.json` recording tool versions and provenance. Outputs follow BIDS naming conventions so neurojax loaders (BIDSConnectomeLoader, WANDMEGLoader) can discover them automatically. Full specification in `docs/BIDS_DERIVATIVES_STRUCTURE.md`.

### 10. Parallel Processing Strategy (DGX Spark)

Processing validated on DGX Spark (Grace ARM64 + GB10 GPU, 20 CPUs, 120GB unified memory). Jobs managed via SLURM (`sbatch`) with named log files in `derivatives/logs/`.

**Measured runtimes for sub-08033:**

| Pipeline | GPU? | Time | Notes |
|----------|------|------|-------|
| fsl_anat (per session) | No | ~10 min | Bias corr + FAST + MNI reg |
| T1Prep (per session) | Yes | ~11 min | AMAP seg + surfaces + thickness (NGC PyTorch 26.03 container or uv venv) |
| FreeSurfer recon-all | No | ~8 h | Both T1ws + T2w pial refinement |
| BET | No | 4.5 min | Quick skull strip |
| FLIRT (per pair) | No | 20 s | 6-DOF rigid |
| dtifit | No | 25 s | On CUBRIC eddy output |
| bedpostx_gpu (3 fibers) | Yes | ~20 min | CUBRIC CHARMED, fsl_sub shell fallback |
| MMORF T1w → Oxford-MM-1 | Yes | 147 s | Unimodal baseline |
| MMORF T1w+DTI → Oxford-MM-1 | Yes | 923 s | Flagship multimodal with tensor reorientation |
| QSMxT (6-echo MEGRE) | No | ~50 min | ROMEO + PDF + TGV (Julia, 3 ARM64 bugs fixed) |
| T2* fitting (7-echo GRE) | No | 1.5 min | Log-linear least squares |
| MRS sLASER fitting | No | ~2 min | spec2nii + fsl_mrs (per VOI) |
| MRS MEGA-PRESS editing | No | ~3 min | mrs-jax pipeline (per VOI) |

**Dependency chain:**
```
Independent (run simultaneously):
  fsl_anat ─────────────────────────── ✓
  FreeSurfer recon-all ─────────────── ✓
  T1Prep ───────────────────────────── ✓ (needs NGC container or uv venv)
  BET ──────────────────────────────── ✓
  DWI eddy (AxCaliber, local) ──────── running
  MRS spec2nii + fitting ───────────── ✓

After fsl_anat:
  FLIRT cross-session ──────────────── ✓
  FLIRT struct → Oxford-MM-1 ───────── ✓

After eddy (CUBRIC CHARMED):
  dtifit ───────────────────────────── ✓
  bedpostx_gpu ─────────────────────── ✓
  → probtrackx2 / TRACULA

After FLIRT + dtifit:
  MMORF T1w → Oxford-MM-1 ─────────── ✓
  MMORF T1w+DTI → Oxford-MM-1 ──────── ✓

After QSMxT:
  QSM + R2* + T2* maps ────────────── ✓
  → Cortical layer analysis (LAYNII)

After bedpostx + FreeSurfer:
  → xtract, connectome, MEG source recon, TMS fitting
```

**Issues encountered and resolved:**
- T1Prep: 3 bugs patched (deterministic CUDA, version string, case-sensitive atlas) — bug report at `docs/t1prep_bug_report.md`
- QSMxT/Julia: 3 ARM64 bugs (FFTW FakeLazyLibrary, Polyester cfunction, threadlocal in LSMR) — fixed in forked QSM.jl
- bedpostx: fsl_sub parallel env config for single-node SLURM — `map_ram: false` fix
- spec2nii: PatientName numeric → string cast for WAND TWIX data
- fsl_sub SLURM config: added `parallel_envs` and `large_job_split_pe` for map_ram queues

---

## Processing Pipeline

All scripts in `scripts/wand_processing/`:

| Script | Stage | Runtime | Status |
|--------|-------|---------|--------|
| `00_setup_env.sh` | Environment (FSL, paths for DGX Spark) | — | Updated for DGX |
| `01_freesurfer_recon.sh` | recon-all + T2 pial refinement | ~6-8h | ✓ sub-08033 |
| `02_dwi_preproc.sh` | topup + eddy for AxCaliber/CHARMED | ~24h (AxCaliber) | Running |
| `09_submit_sub08033.sh` | sbatch parallel submission for sub-08033 | — | ✓ |
| `21_dtifit.sh` | dtifit on CUBRIC CHARMED | 25s | ✓ |
| `22_flirt_diff2struct.sh` | DWI→T1w affine | 9s | ✓ |
| `23_flirt_struct2omm1.sh` | T1w→Oxford-MM-1 affine init | 34s | ✓ |
| `24_mmorf_t1w.sh` | MMORF T1w-only → Oxford-MM-1 | 147s | ✓ |
| `25_mmorf_t1w_dti.sh` | MMORF T1w+DTI → Oxford-MM-1 (flagship) | 923s | ✓ |
| `36_mrs_fslmrs.sh` | spec2nii + fsl_mrs sLASER fitting | ~2min/VOI | ✓ |
| `39_mrs_validation.py` | MRS validation (ISMRM + Big GABA + WAND) | ~1min | ✓ |
| `run_t1prep.sh` | T1Prep via uv venv on DGX | ~11min | ✓ |
| `03_bedpostx_xtract.sh` | Crossing fibers + xtract tract segmentation | ~20min GPU | ✓ (bedpostx) |
| `04_tracula.sh` | FreeSurfer+FSL tract-specific analysis | ~2-4h | Needs FreeSurfer |
| `05_connectome.sh` | probtrackx2 → Cmat/Lmat CSV | ~4-8h | After bedpostx |
| `07_microstructure_comparison.sh` | DIPY/FSL/sbi4dwi/DMI.jl AxCaliber | varies | After AxCaliber eddy |
| `08_advanced_freesurfer.sh` | T1w/T2w, SAMSEG, thalamic nuclei, SynthSeg | ~2-4h | Needs FreeSurfer |

---

## neurojax Module Inventory (Built This Session)

### Whole-Brain Modeling (`bench/`)
| Module | Purpose |
|--------|---------|
| `models/rww.py` | Reduced Wong-Wang neural mass model |
| `monitors/bold.py` | Balloon-Windkessel BOLD hemodynamics |
| `monitors/leadfield.py` | Source→sensor forward projection |
| `monitors/tep.py` | TEP extraction + waveform/GFP loss |
| `adapters/regional.py` | Per-region heterogeneous parameters |
| `adapters/vbjax_adapter.py` | Integrated adapter (BOLD, leadfield, regional, TMS, multi-modal loss, vmap) |
| `stimuli/tms.py` | TMS protocol + stimulus train generation |
| `optimizers/gradient.py` | Multi-modal gradient optimizer |

### Data Loading (`io/`)
| Module | Purpose |
|--------|---------|
| `connectome.py` | BIDS connectome loader (QSIRecon/dmipy-jax) |
| `wand_meg.py` | WAND CTF MEG loader |

### Analysis (`analysis/`)
| Module | Purpose |
|--------|---------|
| `state_spectra.py` | HMM/DyNeMo state → PSD + coherence (inverse TDE-PCA) |
| `summary_stats.py` | Fractional occupancy, lifetime, switching rate |
| `nnmf.py` | Spectral band separation (alpha/beta/gamma) |
| `multitaper.py` | DPSS tapers, multitaper PSD/CPSD/coherence |
| `regression_spectra.py` | DyNeMo GLM-spectrum approach |
| `static.py` | Time-averaged baseline (is dynamics needed?) |
| `tensor_tde.py` | Tucker decomposition alternative to TDE+PCA |
| `virtual_lesion.py` | Disconnection simulation (Momi 2023) |
| `prediction.py` | Cross-validated multi-modal prediction |
| `recurrence.py` | Recurrence plots + RQA measures |
| `visibility.py` | Visibility graphs (time series → network) |
| `funcnet.py` | Mutual information, graph measures, small-world |
| `surrogates.py` | Phase randomization, AAFT, significance testing |

### Models (`models/`)
| Module | Purpose |
|--------|---------|
| `hmm.py` | Gaussian HMM with Baum-Welch EM (osl-dynamics compatible) |
| `dynemo.py` | DyNeMo VAE (equinox, fully native JAX) |

### Other
| Module | Purpose |
|--------|---------|
| `preprocessing/filter.py` | IIR filter via jax.lax.scan (fixed lfilter bug) |
| `geometry/surface.py` | FreeSurfer surface IO |
| `geometry/source_space.py` | Icosahedron source space decimation |
| `dynamics/windowed.py` | Windowed SINDy/DMD/signatures |

---

## Test Coverage

**Total tests written this session: ~380+**

All modules have TDD test coverage. Key test files:
- `test_rww.py` (23), `test_leadfield.py` (25), `test_regional.py` (23)
- `test_integration.py` (34 — wired-up adapter tests)
- `test_tms.py` (27), `test_hmm.py` (22), `test_dynemo.py` (41)
- `test_windowed.py` (28), `test_connectome.py` (22), `test_filter.py` (10)
- `test_state_spectra.py` (23), `test_summary_stats.py` (21), `test_nnmf.py` (11)
- `test_multitaper.py` (15), `test_regression_spectra.py` (7), `test_static.py` (10)
- `test_tensor_tde.py` (13), `test_virtual_lesion.py` (17), `test_prediction.py` (21)
- `test_recurrence.py` (21), `test_visibility.py` (14), `test_funcnet.py` (17), `test_surrogates.py` (15)
- `test_surface.py` (8), `test_source_space.py` (11)

**Bugs found and fixed: ~12** (RWW gamma units, Viterbi T=1, filter lfilter, BOLD normalization, static connectivity normalization, Kronecker covariance symmetry, etc.)

---

## WhoBPyT Feature Parity

neurojax reimplements WhoBPyT (Griffiths Lab) in JAX rather than PyTorch, achieving full feature parity:

| WhoBPyT Capability | neurojax Module | Status |
|---|---|---|
| Reduced Wong-Wang model | `bench/models/rww.py` | Complete (Deco 2013, gamma units fixed) |
| Jansen-Rit model | via vbjax (`vbjax.jr_dfun`) | Complete |
| Balloon-Windkessel BOLD | `bench/monitors/bold.py` | Complete (sigmoid normalization for JR PSP) |
| Leadfield/gain matrix | `bench/monitors/leadfield.py` | Complete (EEG avg ref, MEG, sEEG) |
| TMS stimulus injection | `bench/stimuli/tms.py` | Complete (monophasic/biphasic/square, spatial spread) |
| TEP observation model | `bench/monitors/tep.py` | Complete (waveform MSE + GFP loss) |
| FC / FCD loss | `bench/monitors/fc.py`, `fcd.py` | Complete (differentiable KS distance) |
| Per-region parameters | `bench/adapters/regional.py` | Complete (global + per-region flat array) |
| Gradient-based fitting | `bench/optimizers/gradient.py` | Complete (optax Adam, multi-modal loss) |
| Multi-modal loss | `bench/adapters/vbjax_adapter.py` | Complete (FC + FCD + sensor + TEP with weights) |
| vmap batch evaluation | `bench/adapters/vbjax_adapter.py` | Complete |
| Subject-specific connectome | `io/connectome.py` | Complete (BIDS loader) |

**JAX advantage over PyTorch:** end-to-end `jax.grad` through simulation + hemodynamics + forward projection + loss, `jax.vmap` for batch subjects, `jax.jit` compilation. No tracing issues.

---

## osl-dynamics Reimplementation

Both HMM and DyNeMo are fully reimplemented in JAX, not wrapped as oracles:

**HMM** (`models/hmm.py`, 619 lines): Baum-Welch EM with forward/backward in log-space via `jax.lax.scan`. Viterbi decoding. Multiple random initializations with warm-up. Standardization. osl-dynamics-compatible API (`fit()`, `infer()`, `decode()`).

**DyNeMo** (`models/dynemo.py`, 945 lines): Full variational autoencoder in equinox. BiGRU inference network → reparameterized theta → softmax alpha. GRU temporal prior. Cholesky-parameterized covariances with softplus diagonal. KL annealing. Gradient clipping. osl-dynamics-compatible API.

**Analysis pipeline parity with osl-dynamics:**

| osl-dynamics Feature | neurojax Module | Notes |
|---|---|---|
| TDE + PCA preparation | `data/loading.py` | Matching API |
| State spectral decomposition | `analysis/state_spectra.py` | InvertiblePCA → autocov → FFT → PSD/coherence |
| Summary statistics | `analysis/summary_stats.py` | FO, lifetime, interval, switching rate, binarize |
| NNMF spectral separation | `analysis/nnmf.py` | Multiplicative updates (Lee & Seung 2001) |
| Regression spectra | `analysis/regression_spectra.py` | DyNeMo GLM approach |
| Multitaper PSD | `analysis/multitaper.py` | DPSS tapers, CPSD, coherence |
| Static baseline | `analysis/static.py` | "Is dynamics needed?" comparison |

**Novel addition:** `analysis/tensor_tde.py` — Tucker decomposition as alternative to TDE+PCA, preserving multilinear structure. Factor matrices are directly interpretable as spatial patterns (U_channels) and temporal basis functions (U_lags).

---

## Disconnectome and Virtual Lesion Integration

Three related approaches, all addressable with WAND + neurojax + sbi4dwi:

### De Schotten's Disconnectome (Anatomical)
Maps lesion location → which white matter tracts are severed, using normative tractography atlases. BCBtoolkit provides probabilistic disconnection maps. **Data-driven, structural only.**

### DeepDisconnectome (Matsulevits 2024)
3D U-Net trained on BCBtoolkit outputs: lesion mask → disconnection probability map, 720x faster than tractography. **Could be ported to sbi4dwi as a JAX/equinox model** — sbi4dwi already has 2D U-Net patterns (`models/deepfcd.py`), training pipelines, and differentiable tractography.

### Momi Virtual Lesion (Functional, implemented in neurojax)
Selectively disconnect regions in a fitted whole-brain model and re-simulate to measure functional consequence. **Model-driven, predicts the electrophysiological effect.**

**The unique combination:**
```
Lesion location
  → De Schotten/DeepDisconnectome: which tracts are damaged
  → sbi4dwi ConnectomeMapper: microstructural properties of those tracts
  → Modified Cmat (reduced weights) + Modified Dmat (increased delays / severed)
  → neurojax: simulate with lesioned connectome → predicted TEP/FC/dynamics change
  → Predicted functional deficit
```

This is **end-to-end from lesion to functional prediction** — anatomical disconnection through biophysical modeling to simulated outcome. Nobody has the full chain.

---

## sbi4dwi ↔ neurojax Bridge

The two codebases connect through the Cmat/Dmat interface:

```
sbi4dwi (~/dev/sbi4dwi)                    neurojax (~/dev/neurojax)
─────────────────────                       ─────────────────────────
dMRI → AxCaliber → diameter map
     → tractography → streamlines
     → ConnectomeMapper
       ├── .map_microstructure_to_weights()
       │     → Cmat (n_regions, n_regions)  ──→ BIDSConnectomeLoader
       └── .map_microstructure_to_velocity()       ↓
             → Dmat (n_regions, n_regions)  ──→ ConnectomeData.for_adapter()
                                                   ↓
     biophysics/velocity.py                   VbjaxFitnessAdapter(weights, delays)
       hursh_rushton_velocity()                    ↓
       calculate_latency_matrix()              simulate → fit → predict
```

**ConnectomeMapper** (`sbi4dwi/dmipy_jax/biophysics/network/connectome_mapper.py`):
- `map_microstructure_to_weights()`: samples any voxel-level metric along streamlines → region-level Cmat
- `map_microstructure_to_velocity()`: samples axon diameter along streamlines → per-segment delay via Hursh-Rushton → total conduction delay per connection

Currently numpy-based with python loops (slow for 100K+ streamlines). Future: JAX-ify with vmap for GPU acceleration.

---

## Implementation Status (sub-08033 validation)

### qMRI Relaxometry Pipeline

| Fit | Tool | Result | Status |
|-----|------|--------|--------|
| VFA T1 (DESPOT1) | Python linear + QUIT NLLS | WM 906ms (Python), 1131ms (QUIT) | Validated, both agree |
| DESPOT1-HIFI (T1+B1) | JAX + QUIT | B1=1.31 (both agree), T1 corrected +277ms | B1 field confirmed |
| QMT BPF | Python delta-MTR proxy | 0.12 (semi-quantitative) | Needs pulse param calibration for absolute |
| QMT BPF (Ramani) | JAX (qMRLab port) | Model works, BPF=0.24 (high — pulse params need CUBRIC calibration) | Architecture ready |
| Multi-echo T2*/R2* | Python WLS + QUIT | 32.6ms / 32.2ms (0.4ms agreement) | Validated |
| MP2RAGE | QUIT | UNI image produced | Quantitative T1 needs correct sequence timing from Siemens headers |
| mcDESPOT MWF | JAX (classical) | MWF=0.30, 26% valid — degenerate | Known limitation |
| mcDESPOT MWF | JAX MultiCompartmentNODE | Recovers MWF with DESPOT1-initialized start | Mixing net must be frozen |

### Perfusion Pipeline

| Measure | Tool | Result | Status |
|---------|------|--------|--------|
| Blood T1 | Block-averaged IR (16 blocks × 60 TIs) | 1548ms fitted, 1594ms null-point | Fixed (was 460ms/default) |
| CBF | FSL oxford_asl | 32.3 ml/100g/min | Working |
| TRUST OEF | Structural-guided SSS ROI + voxel-wise T2 QC | T2=76ms, OEF=14% | Real measurement, protocol sensitivity limited |
| CMRO₂ | CBF × OEF × CaO₂ | 41 µmol/100g/min (low, follows OEF) | Working |
| qBOLD OEF | JAX R2/R2' separation | R2=23.8Hz, R2'=10.1Hz | Fitting works, DBV calibration needed |

### PINN / Neural ODE

| Module | Result | Status |
|--------|--------|--------|
| RelaxometryPINN (B1) | Spatial B1 extremely smooth (nn-diff std=0.0006) | Works, but CUBRIC small-FA protocol prevents M0/B1 disentanglement |
| BlochNeuralODE | Forward pass + gradients verified | 7/7 tests pass |
| MultiCompartmentNODE | bSSFP forward + joint loss + MWF recovery | 6/6 tests pass |
| qBOLD module | Signal model + R2/R2' fitting + OEF conversion | 10/10 tests pass |
| Ramani QMT (qMRLab port) | Sf table + Bloch ODE + Ramani signal + fitting | 10/10 tests pass |

### Vascular Modeling (VMTK)

| Step | Tool | Result |
|------|------|--------|
| Frangi vesselness | scikit-image | 693K vessel voxels from 384×512×60 angio |
| Surface extraction | VTK marching cubes | 9.8K points after cleanup |
| Centerlines | VMTK Voronoi-based | 3 paths, vessel radius 0.21-0.94mm |
| VMTK on ARM64 | Built from source (VTK 9.3 + ITK 5.4) | All 7 Python modules working |

### Tools on DGX Spark (ARM64)

| Tool | Status | Install |
|------|--------|---------|
| QUIT | Built from source, working | `~/.local/bin/qi` |
| qMRLab | Loads in Octave, too slow for production | Native Octave 8.4 |
| VMTK | Built from source, Python wrapping working | Fork m9h/vmtk |
| FSL 6.0.7 | Pre-installed | oxford_asl, bet, flirt |
| FreeSurfer | Available for ses-03 | recon-all outputs |
| neurojax | JAX 0.4 (CPU fallback) | `.venv/` |

### Key Findings

1. **B1 field at CUBRIC 3T:** median B1=1.31 across the brain. The CUBRIC mcDESPOT protocol (FA 2-18°, IR FA=5°) has weak B1 sensitivity — S ∝ M0×B1 in the small-angle regime, making M0 and B1 degenerate. Both JAX HIFI and QUIT HIFI agree on B1=1.31.

2. **InvRec is multi-block:** 16 discrete IR blocks of 60 readouts (not continuous Look-Locker). Block-averaged fitting gives T1_blood=1548ms with null-point cross-validation at 1594ms.

3. **Phase-contrast angio at labeling plane:** z=[-105,-28]mm covers the neck (feeding arteries), not the TRUST/InvRec slice (z≈+10mm). Used for vessel modeling and total CBF validation, not SSS ROI guidance.

4. **qMRLab models are better but slow:** The Ramani QMT model with full pulse shape characterization (Gauss-Hanning, super-Lorentzian lineshape) is ported to JAX (`neurojax.qmri.qmt_ramani`) but needs CUBRIC-specific pulse parameter calibration for absolute BPF.

---

## Quantitative MRI Available in WAND

WAND ses-02 and ses-06 contain rich qMRI data beyond standard T1w:

| Acquisition | Session | What It Measures | Relevance |
|---|---|---|---|
| VFA (SPGR + SPGR-IR + SSFP) | ses-02 | Quantitative T1 mapping | Tissue composition |
| QMT (16 MT-weighted volumes) | ses-02 | Bound pool fraction, exchange rate | **Direct myelin content** |
| T1w + T2w | ses-03 | T1w/T2w ratio | Myelin proxy (Glasser 2011) |
| MP2RAGE (PSIR, 2 inversions) | ses-06 | Quantitative T1 mapping | Tissue composition (7T-like contrast at 3T) |
| Multi-echo GRE (7 echoes) | ses-06 | T2* mapping, R2* | **Iron + myelin sensitivity** |

The QMT bound pool fraction is the gold standard for in-vivo myelin quantification. Comparing it against the cheap T1w/T2w ratio tests an assumption underlying the Valdes-Sosa ξ-αNET model and many other connectome-constrained models that use T1w/T2w as a cortical hierarchy proxy.

---

## End-to-End Data Flow (Single Subject)

```
WAND sub-08033 (10.24 GB, 7 sessions, 108 files)
│
├─ ses-02/dwi (AxCaliber + CHARMED)
│   → topup + eddy → bedpostx → probtrackx2
│   → sbi4dwi AxCaliber → diameter map → ConnectomeMapper
│   → Cmat (streamline weights) + Dmat (microstructure-informed delays)
│
├─ ses-02/anat (QMT + VFA)
│   → QMT fitting → bound pool fraction (myelin ground truth)
│   → VFA fitting → T1 map
│
├─ ses-03/anat (T1w + T2w)
│   → FreeSurfer recon-all → surfaces, parcellation
│   → T1w/T2w ratio → myelin proxy → validate against QMT
│   → BEM forward model → leadfield matrix
│
├─ ses-01/meg (CTF 275ch, 5 tasks)
│   → Source reconstruction (beamformer/CHAMPAGNE) → parcellated
│   → TDE + PCA (or TuckerTDE) → HMM/DyNeMo
│   → State spectra, summary stats, NNMF
│   → Individual alpha frequency
│
├─ ses-04+05/mrs (sLASER, 4 VOIs × 2 sessions)
│   → fsl_mrs → GABA, glutamate, NAA concentrations
│   → Tissue-corrected quantification
│   → Constrain RWW J_I (∝ GABA) and J_NMDA (∝ Glu)
│
├─ ses-06/anat (MP2RAGE + multi-echo GRE)
│   → T1 mapping, T2* mapping → validate myelin measures
│
└─ ses-08/tms (SICI, 3 runs)
    → Extract TEP windows
    → VbjaxFitnessAdapter: fit RWW model to empirical TEP
    → Virtual lesion analysis → contribution matrix
    → Local→network transition time

All features → cross_validated_predict → "What predicts TMS response?"
```

---

## FreeSurfer 8.2.0 Opportunities Beyond recon-all

Most neuroimaging pipelines use FreeSurfer only for `recon-all`. The 8.x series includes powerful tools that are underutilized:

### Deep Learning Segmentation (No recon-all prerequisite)
- **`mri_synthseg`**: Segments any brain image regardless of contrast, resolution, or orientation. On WAND, this enables consistent parcellation across ses-02 (T1w), ses-03 (T1w+T2w), ses-04 (T1w), ses-06 (MP2RAGE, multi-echo GRE) without session-specific recon-all. Outputs volume estimates + QC scores.
- **`mri_synthstrip`**: DL skull stripping that outperforms BET on non-standard contrasts. Critical for WAND's QMT and VFA volumes where BET frequently fails.
- **`mri_synthsr`**: Super-resolution synthesis — could upsample lower-resolution functional data for improved registration.
- **`mri_synthmorph` / `fs-synthmorph-reg`**: DL contrast-agnostic registration. **Now the primary cross-session registration tool** — handles T1w↔QMT↔MP2RAGE↔multi-echo GRE without intensity-based assumptions. Modes: joint (affine+deform), affine, rigid, deform. Symmetric, anatomy-aware, no skull stripping needed. 24GB model.
- **`mri_WMHsynthseg`**: White matter hyperintensity segmentation — relevant for aging subjects in the WAND cohort (ages 18-63).

### Subcortical Nuclei Segmentation
- **`segmentThalamicNuclei.sh`**: Segments 25+ thalamic subnuclei from T1. Critical for thalamocortical conduction velocity estimation (Valdes-Sosa ξ-αNET). The thalamus is the relay station — knowing which subnuclei connect to which cortical regions via which tracts determines the delay structure.
- **`segmentThalamicNuclei_DTI.sh`**: Enhanced version using DWI — WAND's AxCaliber data improves boundaries where T1 contrast is insufficient.
- **`segmentHA_T1.sh` / `segmentHA_T2.sh`**: Hippocampal subfields + amygdala nuclei. With T2w from ses-03, subfield boundaries are more accurate. Relevant for memory network analysis.
- **`mri_segment_hypothalamic_subunits`**: Hypothalamic segmentation — relevant for autonomic/stress response studies.
- **`mri_sclimbic_seg`**: Subcortical limbic structures (nucleus accumbens, ventral pallidum, etc.).

### Multi-Modal Segmentation
- **`run_samseg`**: Sequence-Adaptive Multimodal Segmentation — uses ALL available contrasts simultaneously (T1+T2+FLAIR+etc.) for a single segmentation that leverages multi-contrast information. For WAND, feeding it T1w + T2w + QMT produces better tissue boundaries than any single contrast.
- **`run_samseg_long`**: Longitudinal version — could track within-subject changes across WAND sessions.

### Surface-Based Analysis
- **`pctsurfcon`**: Percent surface contrast — another myelin-sensitive measure for the T1w/T2w comparison.
- **`mri_vol2surf`**: Project any volumetric map onto cortical surface — used for surface-level myelin mapping, QMT visualization, AxCaliber diameter projected to cortex.
- **`mris_anatomical_stats`**: Per-region cortical thickness, surface area, volume, curvature — features for the prediction model (Phase 4).

### Quantitative Tissue Properties
- **`mri_gtmpvc`**: Geometric Transfer Matrix partial volume correction — designed for PET but applicable to MRS voxel analysis (corrects for CSF/WM contamination of grey matter signal).

---

## Valdes-Sosa ξ-αNET: Implications for WAND Analysis

**Paper:** Valdes-Sosa et al. (2026, National Science Review). "Lifespan development of EEG alpha and aperiodic component sources is shaped by the connectome and axonal delays."

### Key Claims
1. EEG alpha rhythm and aperiodic (1/f) component arise from distinct cortical networks with **opposite directional organization**: alpha = feedback (posterior→anterior), aperiodic = feedforward (anterior→posterior).
2. Both components follow a **nonlinear inverted-U lifespan trajectory** (increasing until ~30-40 years, then declining).
3. **Global conduction delays correlate negatively with alpha frequency** — faster conduction → higher alpha peak.
4. The cortical hierarchy is inferred from the **T1w/T2w myelination map** (Glasser & Van Essen 2011), used as a proxy for feedforward/feedback organization.

### What WAND Enables Beyond This Paper
The paper used T1w/T2w as a myelin proxy. WAND allows testing whether the relationship holds with **ground-truth myelin** from QMT:

| Valdes-Sosa used | WAND has | Improvement |
|---|---|---|
| T1w/T2w ratio → cortical hierarchy | QMT bound pool fraction → actual myelin | Direct measurement, not proxy |
| Assumed conduction velocity from myelination | AxCaliber axon diameter → Hursh-Rushton velocity | Per-tract measurement, not assumed |
| Population-level connectome | Individual structural connectome (probtrackx2) | Subject-specific, not template |
| EEG (lower spatial resolution) | MEG (275-channel CTF, better spatial) | Superior source localization |

### Specific Analyses Enabled
1. **Replicate ξ-αNET with MEG instead of EEG** — MEG has better spatial resolution for separating alpha and aperiodic sources
2. **Replace T1w/T2w with QMT** — does the cortical hierarchy change? Does the alpha-delay correlation improve?
3. **Individual conduction delays from AxCaliber** — instead of a global delay parameter, use per-tract delays from actual microstructure
4. **Test at single-subject level** — the paper worked at population level; WAND's multi-modal data enables individual-level model fitting
5. **Add MRS constraints** — GABA/glutamate concentrations constrain the excitation-inhibition balance that determines aperiodic slope (1/f exponent)

### Connection to Whole-Brain Modeling
The ξ-αNET model decomposes spectral Granger causality — it's a frequency-domain model of effective connectivity. neurojax's time-domain models (RWW, JR) with the same connectome and delays should produce the same alpha peak frequency if the physics is right. This provides a **cross-validation**: does the time-domain simulation predict the same alpha frequency as the spectral Granger model?

---

## Geodesic Cortical Flow (Liu, Wiesman & Baillet 2026)

**Paper:** "Hierarchical Flows of Human Cortical Activity" — bioRxiv 2026.03.19.712872. Montreal Neurological Institute, McGill.

### Method
Introduces **geodesic cortical flow**: surface-based optical flow that estimates millisecond-resolved propagation vectors **tangent to the cortical surface** from source-imaged MEG. Unlike connectivity (which pair of regions interact) or states (which state is the brain in), this captures the actual *direction and speed* of activity propagating across the cortical sheet.

### Key Findings (608 healthy adults, resting-state MEG)
1. **Slow activity (1-13 Hz) propagates upstream** — sensory → association cortex (feedforward direction along the functional gradient)
2. **Beta activity (13-30 Hz) propagates downstream** — association → sensory cortex (feedback)
3. **Aging shifts the balance**: weaker upstream slow flow, stronger downstream beta flow
4. **Kinetic energy** of cortical flow follows a posterior→anterior gradient; higher frontoparietal kinetic energy predicts better fluid intelligence
5. **Dwell times** of stable kinetic-energy states track regional neuronal timescales

### Integration with WAND Analysis

This approach is directly complementary to our existing pipeline:

| Our approach | Their approach | Combined |
|---|---|---|
| HMM/DyNeMo: *which state* | Geodesic flow: *which direction* | Which direction during each state |
| State lifetimes | Kinetic-energy dwell times | Cross-validate: do they agree? |
| AxCaliber conduction velocity | Surface propagation speed | Microstructure predicts flow speed |
| Parcellated timeseries | Vertex-level surface activity | Flow computed before parcellation |

### Specific Opportunities for WAND
1. **Flow × microstructure**: AxCaliber conduction velocity should predict surface propagation speed. Thick-axon tracts → faster flow.
2. **Flow × TMS**: After TMS, does the evoked flow follow the structural connectome? Virtual lesion: disconnect a region and predict how flow patterns change.
3. **Flow × brain states**: Compute geodesic flow within each HMM state window. Do different states have different dominant flow directions?
4. **Flow kinetic energy as prediction feature**: Add to Phase 4 — does flow KE predict TMS response better than static connectivity measures?
5. **Frequency-specific flow × Valdes-Sosa ξ-αNET**: The alpha upstream/beta downstream organization maps onto ξ-αNET's feedforward/feedback hierarchy. WAND can test whether subject-specific conduction delays (from AxCaliber) predict the flow direction bias.

### Implementation Requirements
A `geodesic_cortical_flow` module would need:
- Source-space MEG on cortical surface (vertices × time) — from FreeSurfer + MNE source reconstruction
- Surface mesh with geodesic distance computation — `geometry/surface.py` provides the mesh IO
- Optical flow estimation on surface manifold — Horn-Schunck or Lucas-Kanade adapted to triangulated surfaces
- Frequency-band filtering → per-band flow statistics
- Kinetic energy timecourse → dwell time analysis (reuse `summary_stats.py` machinery)

This is implementable in JAX: the surface mesh defines a sparse adjacency, the optical flow becomes a regularized least-squares problem solvable with `jax.scipy.sparse.linalg`, and the temporal iteration uses `jax.lax.scan`.

---

## Key References

1. Momi D et al. (2023). TMS-evoked responses are driven by recurrent large-scale network dynamics. *eLife* 12:e83232.
2. Valdes-Sosa PA et al. (2026). Lifespan development of EEG alpha and aperiodic component sources is shaped by the connectome and axonal delays. *National Science Review*.
3. Matsulevits A et al. (2024). Deep learning disconnectomes to accelerate long-term post-stroke predictions. *Brain Communications* 6(5):fcae338.
4. Griffiths JD et al. (2022). Whole-brain modelling: past, present, and future. *Computational Modelling of the Brain*.
5. Gohil C et al. (2024). osl-dynamics: A toolbox for modelling fast dynamic brain activity. *eLife* 13:e91949.
6. Liu X, Wiesman AI, Baillet S (2026). Hierarchical flows of human cortical activity. *bioRxiv* 2026.03.19.712872.
