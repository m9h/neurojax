# MEGA-PRESS GABA Quantification with NeuroJAX

```{admonition} Tutorial overview
:class: tip

This tutorial walks through the complete MEGA-PRESS processing pipeline in
NeuroJAX, from loading raw Siemens TWIX data to reporting GABA concentrations
with quality control metrics. Each step maps to a function in the
`neurojax.analysis` and `neurojax.qmri` subpackages.
```

## Introduction

### What is MEGA-PRESS?

MEGA-PRESS (**ME**scher-**GA**rwood **P**oint-**RES**olved **S**pectroscopy)
is a J-difference editing technique for detecting low-concentration
metabolites --- most commonly $\gamma$-aminobutyric acid (GABA) --- whose
resonances overlap with much larger signals in a conventional
$^{1}\mathrm{H}$ MRS spectrum.

The experiment interleaves two acquisition conditions:

- **Edit-ON**: A frequency-selective inversion pulse is applied at 1.9 ppm,
  which refocuses the J-coupled GABA multiplet at 3.0 ppm.
- **Edit-OFF**: The inversion pulse is applied at a symmetric control
  frequency (e.g., 7.5 ppm), leaving the 3.0 ppm region unperturbed.

Subtracting the averaged edit-OFF from the averaged edit-ON spectrum
(**difference spectrum**) cancels overlapping creatine and other singlet
resonances, isolating the edited GABA signal at ~3.0 ppm
{cite:p}`mescher1998simultaneous,edden2014gannet`.

### Why JAX acceleration matters for MRS

Single-voxel MRS datasets are modest in size, but modern analysis
workflows involve computationally intensive steps that benefit from
hardware acceleration:

- **Spectral registration** aligns hundreds of individual transients via
  iterative optimisation ({cite:t}`near2015frequency`).
- **Tensor decomposition** of multi-coil, multi-average data scales
  cubically with the number of coils.
- **Batch processing** across large multi-site cohorts (e.g., Big GABA with
  24 sites and >250 datasets; {cite:t}`mikkelsen2017big`) is
  embarrassingly parallel but still time-consuming on a single CPU.

NeuroJAX provides NumPy-based implementations that can be JIT-compiled
through JAX for GPU/TPU acceleration, and the tensor decomposition
routines in `neurojax.qmri.mrs_tensor` are designed with this in mind.


## Pipeline overview

The NeuroJAX MEGA-PRESS pipeline consists of the following stages, each
handled by a dedicated module:

| Stage | Module | Key function |
|---|---|---|
| 1. Data loading | `neurojax.analysis.mrs_io` | `read_twix()` |
| 2. Preprocessing | `neurojax.analysis.mrs_preproc` | `exponential_apodization()`, `eddy_current_correction()` |
| 3. MEGA-PRESS processing | `neurojax.analysis.mega_press` | `process_mega_press()` |
| 4. Phase correction | `neurojax.analysis.mrs_phase` | `zero_order_phase_correction()`, `first_order_phase_correction()` |
| 5. Quantification | `neurojax.analysis.mrs_quantify` | `quantify_mega_press()` |
| 6. Quality control | `neurojax.analysis.mrs_qc` | `generate_qc_report()` |

For most use cases, `quantify_mega_press()` wraps stages 3--5 in a
single call. The sections below explain each stage individually so you
can customise or inspect intermediate results.


## 1. Data loading

### Loading Siemens TWIX data

NeuroJAX reads Siemens TWIX `.dat` files natively through the
[mapVBVD](https://github.com/pehses/mapVBVD) Python package, without
requiring `spec2nii` or FSL-MRS as a dependency.

```python
from neurojax.analysis.mrs_io import read_twix, MRSData

# Load a MEGA-PRESS TWIX file (e.g., from the Big GABA dataset)
mrs = read_twix(
    "/data/datasets/big_gaba/S5/S5_MP/S01/S01_GABA_68.dat",
    load_water_ref=True,
)

print(type(mrs))          # <class 'neurojax.analysis.mrs_io.MRSData'>
print(mrs.data.shape)     # (4096, 32, 2, 160)  — typical 3T Siemens MEGA-PRESS
print(mrs.dwell_time)     # 2.5e-04  (seconds)
print(mrs.centre_freq)    # 123253000.0  (Hz, ~3T proton)
print(mrs.te)             # 68.0  (ms)
print(mrs.tr)             # 2000.0  (ms)
print(mrs.field_strength) # 2.89  (T)
print(mrs.n_coils)        # 32
print(mrs.n_averages)     # 160
```

### Understanding the `MRSData` container

The {class}`~neurojax.analysis.mrs_io.MRSData` dataclass standardises MRS
data from any vendor format into a consistent layout:

```{list-table} MRSData attributes
:header-rows: 1

* - Attribute
  - Type
  - Description
* - `data`
  - `np.ndarray`
  - Complex FID array. Shape is `(n_spec, n_coils, n_dyn)` for standard
    acquisitions, or `(n_spec, n_coils, n_edit, n_dyn)` for edited
    sequences (MEGA-PRESS, HERMES).
* - `dwell_time`
  - `float`
  - Dwell time in seconds (= 1 / spectral bandwidth).
* - `centre_freq`
  - `float`
  - Spectrometer centre frequency in Hz.
* - `te`, `tr`
  - `float`
  - Echo time and repetition time in milliseconds.
* - `field_strength`
  - `float`
  - Static magnetic field strength in Tesla.
* - `n_coils`
  - `int`
  - Number of receive coils.
* - `n_averages`
  - `int`
  - Number of dynamic repetitions.
* - `dim_info`
  - `dict`
  - Mapping of dimension names to axis indices, e.g.,
    `{'spec': 0, 'coil': 1, 'edit': 2, 'dyn': 3}`.
* - `water_ref`
  - `np.ndarray` or `None`
  - Unsuppressed water reference FID, if available.
```

For MEGA-PRESS data the `dim_info` dictionary will contain an `'edit'`
key whose value is `2`, indicating two editing conditions (ON and OFF).

### Data shape conventions

The raw FID array follows the convention:

$$
\texttt{data.shape} = (N_{\text{spec}},\; N_{\text{coils}},\; N_{\text{edit}},\; N_{\text{dyn}})
$$

where:
- $N_{\text{spec}}$ is the number of spectral (time-domain) points,
- $N_{\text{coils}}$ is the number of receive coil elements,
- $N_{\text{edit}} = 2$ for MEGA-PRESS (ON / OFF),
- $N_{\text{dyn}}$ is the number of dynamic averages per edit condition.


## 2. Preprocessing

Before MEGA-PRESS-specific processing, standard preprocessing steps
improve spectral quality.

### Apodization (line broadening)

Exponential apodization applies a matched filter that improves the
signal-to-noise ratio at the expense of spectral resolution
({cite:t}`degraafinvivo`):

```python
from neurojax.analysis.mrs_preproc import (
    exponential_apodization,
    gaussian_apodization,
)

# Apply 3 Hz exponential (Lorentzian) line broadening
fid_apod = exponential_apodization(
    mrs.data,
    dwell_time=mrs.dwell_time,
    broadening_hz=3.0,
)

# Alternatively, Gaussian apodization for resolution enhancement
fid_gauss = gaussian_apodization(
    mrs.data,
    dwell_time=mrs.dwell_time,
    broadening_hz=5.0,
)
```

```{note}
Apodization is applied along axis 0 (the spectral/time dimension) and
broadcasts over coil, edit, and dynamic dimensions automatically.
```

### Eddy current correction

If an unsuppressed water reference is available, the Klose method removes
time-dependent phase distortions caused by eddy currents
({cite:t}`klose1990invivo`):

```python
from neurojax.analysis.mrs_preproc import eddy_current_correction

if mrs.water_ref is not None:
    fid_ecc = eddy_current_correction(
        mrs.data,
        water_ref=mrs.water_ref,
    )
```

The correction subtracts the instantaneous phase of the water FID from the
metabolite FID point-by-point, exploiting the fact that water and
metabolite signals experience the same eddy-current-induced phase errors.

### Frequency referencing

Frequency referencing shifts the spectrum so that a known metabolite peak
(typically NAA at 2.01 ppm or creatine at 3.03 ppm) lands at its
canonical chemical shift:

```python
from neurojax.analysis.mrs_preproc import frequency_reference

# Single-FID frequency referencing (e.g., after coil combination)
fid_ref = frequency_reference(
    fid=averaged_fid,
    dwell_time=mrs.dwell_time,
    centre_freq=mrs.centre_freq,
    target_ppm=2.01,          # where we want NAA to end up
    target_peak_ppm=2.01,     # approximate current NAA position
    search_window_ppm=0.3,    # search +/- 0.3 ppm
)
```


## 3. MEGA-PRESS processing

The core of the pipeline is {func}`~neurojax.analysis.mega_press.process_mega_press`,
which performs coil combination, per-transient alignment, outlier rejection,
and edit-ON/OFF subtraction in a single call.

### Full processing in one call

```python
from neurojax.analysis.mega_press import process_mega_press, MegaPressResult

result = process_mega_press(
    data=mrs.data,                # (n_spec, n_coils, 2, n_dyn)
    dwell_time=mrs.dwell_time,
    centre_freq=mrs.centre_freq,
    align=True,                   # spectral registration
    reject=True,                  # outlier rejection
    reject_threshold=3.0,         # z-score threshold
    paired_alignment=True,        # recommended: paired FPC
)

# result is a MegaPressResult named tuple
print(type(result))               # <class 'MegaPressResult'>
print(result.diff.shape)          # (4096,) — difference FID
print(result.edit_on.shape)       # (4096,) — averaged edit-ON FID
print(result.edit_off.shape)      # (4096,) — averaged edit-OFF FID
print(result.n_averages)          # e.g., 156 (after rejecting 4 outliers)
```

The returned {class}`~neurojax.analysis.mega_press.MegaPressResult` contains:

| Field | Shape | Description |
|---|---|---|
| `diff` | `(n_spec,)` | Difference FID (edit-ON minus edit-OFF) |
| `edit_on` | `(n_spec,)` | Averaged edit-ON FID |
| `edit_off` | `(n_spec,)` | Averaged edit-OFF FID |
| `sum_spec` | `(n_spec,)` | Sum spectrum (edit-ON plus edit-OFF) |
| `freq_shifts` | `(2*n_dyn,)` | Per-transient frequency corrections (Hz) |
| `phase_shifts` | `(2*n_dyn,)` | Per-transient phase corrections (rad) |
| `rejected` | `(2*n_dyn,)` | Boolean mask of rejected transients |
| `n_averages` | `int` | Number of averages used per condition |
| `dwell_time` | `float` | Dwell time (seconds) |
| `bandwidth` | `float` | Spectral bandwidth (Hz) |

### Step-by-step breakdown

Below we walk through each sub-step that `process_mega_press()` performs
internally, in case you need to customise individual stages.

#### Coil combination (SVD)

Multi-coil data is combined using the first singular vector of the coil
dimension ({cite:t}`near2021preprocessing`):

```python
from neurojax.analysis.mega_press import coil_combine_svd

# Input: (n_spec, n_coils, n_edit, n_dyn)
# Output: (n_spec, n_edit, n_dyn)
combined = coil_combine_svd(mrs.data)
print(combined.shape)  # (4096, 2, 160)
```

The SVD weights are estimated from the first transient and applied
uniformly to all dynamics. This provides optimal SNR coil combination
without requiring separate noise covariance data.

```{admonition} Alternative coil combination methods
:class: seealso

The `neurojax.qmri.mrs` module offers two additional methods:
- {func}`~neurojax.qmri.mrs.sensitivity_weighted_combine` --- amplitude/phase
  weighting from the first FID point (standard FSL-MRS method).
- {func}`~neurojax.qmri.mrs.tucker_coil_combine` --- data-driven Tucker
  decomposition on the coil mode.
```

#### Spectral registration (frequency/phase alignment)

Individual transients are aligned to a reference (the mean edit-OFF FID)
using the spectral registration algorithm of {cite:t}`near2015frequency`:

```python
from neurojax.analysis.mega_press import spectral_registration, apply_correction

# Estimate frequency and phase offset of one transient
freq_shift_hz, phase_shift_rad = spectral_registration(
    fid=combined[:, 1, 0],          # first edit-OFF transient
    reference=combined[:, 1, :].mean(axis=1),  # mean edit-OFF
    dwell_time=mrs.dwell_time,
    freq_range=(1.8, 4.2),          # ppm range for alignment
    centre_freq=mrs.centre_freq,
)

# Apply the correction in the time domain
corrected = apply_correction(
    fid=combined[:, 1, 0],
    freq_shift=freq_shift_hz,
    phase_shift=phase_shift_rad,
    dwell_time=mrs.dwell_time,
)
```

#### Paired frequency-phase correction (FPC)

```{important}
For MEGA-PRESS data, **paired alignment** (`paired_alignment=True`) is
strongly recommended. Independent alignment of edit-ON and edit-OFF
transients can introduce differential phase errors that create subtraction
artifacts in the difference spectrum ({cite:t}`mikkelsen2018robust`).
```

Paired FPC estimates the correction from each edit-OFF transient (which
contains stable singlet peaks) and applies the *same* correction to both
the paired edit-ON and edit-OFF, preserving their relative phase for
clean subtraction:

```python
from neurojax.analysis.mega_press import align_edit_pairs

edit_on = combined[:, 0, :]   # (n_spec, n_dyn)
edit_off = combined[:, 1, :]  # (n_spec, n_dyn)

edit_on_aligned, edit_off_aligned, freq_shifts, phase_shifts = align_edit_pairs(
    edit_on=edit_on,
    edit_off=edit_off,
    dwell_time=mrs.dwell_time,
    centre_freq=mrs.centre_freq,
    freq_range=(1.8, 4.2),
)
```

#### Outlier rejection

Transients with abnormal residuals (e.g., from subject motion or hardware
glitches) are detected using a median-absolute-deviation-based z-score
and excluded from averaging:

```python
from neurojax.analysis.mega_press import reject_outliers

rejected_mask = reject_outliers(
    fids=edit_off_aligned,
    dwell_time=mrs.dwell_time,
    threshold=3.0,  # reject if |z-score| > 3
)

n_rejected = rejected_mask.sum()
print(f"Rejected {n_rejected} of {edit_off_aligned.shape[1]} transients")

# Apply mask
edit_off_clean = edit_off_aligned[:, ~rejected_mask]
```

#### Difference spectrum

After alignment and rejection, the edit-ON and edit-OFF spectra are
averaged separately, then subtracted to isolate the edited GABA signal:

```python
avg_on = edit_on_aligned[:, ~rejected_mask].mean(axis=1)
avg_off = edit_off_aligned[:, ~rejected_mask].mean(axis=1)

# Difference spectrum isolates GABA at ~3.0 ppm
diff_fid = avg_on - avg_off

# Sum spectrum retains all metabolites (useful for QC)
sum_fid = avg_on + avg_off
```


## 4. Phase correction

Automatic phase correction ensures that metabolite peaks appear as
positive, absorption-mode lineshapes in the real part of the spectrum.
This is essential for accurate Gaussian fitting and quantification.

### Zero-order phase correction

Zero-order correction applies a constant phase rotation that maximises
the integral of the real part of the spectrum:

```python
from neurojax.analysis.mrs_phase import zero_order_phase_correction

# Correct the difference spectrum
diff_phased, phi0 = zero_order_phase_correction(
    diff_fid,
    return_phase=True,
)

print(f"Applied zero-order phase: {np.degrees(phi0):.1f} degrees")
```

### First-order phase correction

First-order correction removes a linear phase gradient across the
spectrum (group delay artifact), which affects peaks at different
chemical shifts differently:

```python
from neurojax.analysis.mrs_phase import first_order_phase_correction

# Joint zero + first-order correction on the edit-OFF spectrum
edit_off_phased = first_order_phase_correction(
    fid=avg_off,
    dwell_time=mrs.dwell_time,
    ppm_range=(0.5, 4.2),
    cf=mrs.centre_freq,
)
```

```{tip}
For the MEGA-PRESS *difference* spectrum, zero-order correction alone is
usually sufficient because the subtraction cancels most first-order
phase errors. Apply first-order correction to the edit-OFF spectrum if
you need it for standard metabolite quantification (e.g., NAA, creatine).
```

### Phase correction in `neurojax.qmri.mrs`

The `neurojax.qmri.mrs` module provides additional phase correction
routines designed for use with the higher-level preprocessing pipeline:

```python
from neurojax.qmri.mrs import auto_phase_correct_0th, auto_phase_correct_1st, ppm_axis

# Compute ppm axis
ppm = ppm_axis(n_points=len(diff_fid), dwell=mrs.dwell_time,
               cf_mhz=mrs.centre_freq / 1e6)

# Zero-order on the spectrum directly
diff_spec = np.fft.fftshift(np.fft.fft(diff_fid))
diff_spec_phased, phase0_deg = auto_phase_correct_0th(diff_spec)

# Joint zero + first-order
spec_phased, phase0, phase1 = auto_phase_correct_1st(
    diff_spec, ppm, pivot_ppm=2.02,
)
```


## 5. Quantification

### GABA Gaussian fitting

The edited GABA+ peak at ~3.0 ppm in the difference spectrum is fitted
with a Gaussian model to extract peak area, centre, width, and fit
uncertainty:

```python
from neurojax.analysis.mrs_phase import fit_gaba_gaussian
import numpy as np

# Compute the phased difference spectrum
diff_spec = np.fft.fftshift(np.fft.fft(diff_phased))
diff_spec_real = np.real(diff_spec)

# Compute the ppm axis
n = len(diff_phased)
freq = np.fft.fftshift(np.fft.fftfreq(n, mrs.dwell_time))
ppm = freq / (mrs.centre_freq / 1e6) + 4.65

# Fit GABA Gaussian
gaba_fit = fit_gaba_gaussian(
    spectrum_real=diff_spec_real,
    ppm=ppm,
    fit_range=(2.7, 3.3),  # ppm range for fitting
)

print(f"GABA+ centre:    {gaba_fit['centre_ppm']:.3f} ppm")
print(f"GABA+ amplitude: {gaba_fit['amplitude']:.2f}")
print(f"GABA+ FWHM:      {gaba_fit['fwhm_ppm']:.3f} ppm")
print(f"GABA+ area:      {gaba_fit['area']:.4f}")
print(f"CRLB:            {gaba_fit['crlb_percent']:.1f} %")
print(f"Fit residual:    {gaba_fit['residual']:.6f}")
```

The returned dictionary contains:

| Key | Type | Description |
|---|---|---|
| `centre_ppm` | `float` | Fitted peak centre in ppm |
| `amplitude` | `float` | Gaussian amplitude |
| `sigma_ppm` | `float` | Gaussian standard deviation (ppm) |
| `fwhm_ppm` | `float` | Full width at half maximum (ppm) |
| `area` | `float` | Integrated peak area |
| `baseline` | `float` | Fitted baseline offset |
| `crlb_percent` | `float` | Cramer--Rao lower bound (%) |
| `residual` | `float` | RMS fit residual |

### Water-referenced absolute quantification

If an unsuppressed water reference is available, the GABA area can be
converted to an absolute concentration in millimolar (mM) using the
method of {cite:t}`gasparovic2006use`:

$$
[\text{GABA}] = \frac{S_{\text{GABA}}}{S_{\text{W}}} \cdot \frac{f_{\text{W}}}{R_{\text{GABA}}} \cdot \frac{[\text{W}]}{f_{\text{tissue}}}
$$

where $f_{\text{W}}$ accounts for tissue water content and relaxation,
$R_{\text{GABA}}$ is the metabolite relaxation attenuation, and
$f_{\text{tissue}}$ excludes CSF from the voxel:

```python
from neurojax.analysis.mrs_phase import water_referenced_quantification

# Compute water area from the unsuppressed water reference
water_spec = np.fft.fftshift(np.fft.fft(mrs.water_ref))
water_area = float(np.max(np.abs(water_spec)))

# Tissue fractions from voxel segmentation (e.g., via FSL FAST)
tissue_fracs = {
    'gm': 0.55,    # grey matter fraction
    'wm': 0.35,    # white matter fraction
    'csf': 0.10,   # CSF fraction
}

gaba_conc_mM = water_referenced_quantification(
    metab_area=gaba_fit['area'],
    water_area=water_area,
    tissue_fracs=tissue_fracs,
    te=mrs.te / 1000.0,       # convert ms -> s
    tr=mrs.tr / 1000.0,       # convert ms -> s
    metab_t1=1.3,              # GABA T1 at 3T (s)
    metab_t2=0.16,             # GABA T2 at 3T (s)
    field_strength=mrs.field_strength,
)

print(f"GABA concentration: {gaba_conc_mM:.2f} mM")
```

```{admonition} Tissue water relaxation values
:class: note

The quantification uses literature values for tissue water $T_1$ and
$T_2$ at 3T from {cite:t}`wansapura1999nmr` and
{cite:t}`stanisz2005t1`:

| Tissue | $T_1$ (s) | $T_2$ (s) | Water content |
|---|---|---|---|
| Grey matter | 1.33 | 0.110 | 0.78 |
| White matter | 0.83 | 0.080 | 0.65 |
| CSF | 4.16 | 1.650 | 0.97 |

At 7T, different values are used automatically when `field_strength >= 5`.
```

### GABA/NAA ratio (without water reference)

When no water reference is available, the GABA/NAA area ratio provides a
semi-quantitative alternative:

```python
# The edit-OFF spectrum contains NAA at ~2.01 ppm
off_spec = np.fft.fftshift(np.fft.fft(edit_off_phased))

# Fit NAA Gaussian in the 1.8--2.25 ppm range
naa_fit = fit_gaba_gaussian(
    np.real(off_spec), ppm, fit_range=(1.8, 2.25),
)

gaba_naa_ratio = gaba_fit['area'] / naa_fit['area']
print(f"GABA/NAA ratio: {gaba_naa_ratio:.4f}")
```

Normal GABA/NAA ratios in healthy adults are typically in the range
0.10--0.25 ({cite:t}`mikkelsen2017big`).


### End-to-end quantification

For convenience, {func}`~neurojax.analysis.mrs_quantify.quantify_mega_press`
wraps the entire chain --- MEGA-PRESS preprocessing, phase correction,
GABA fitting, NAA fitting, and optional water-referenced quantification
--- in a single function call:

```python
from neurojax.analysis.mrs_quantify import quantify_mega_press

results = quantify_mega_press(
    data=mrs.data,
    dwell_time=mrs.dwell_time,
    centre_freq=mrs.centre_freq,
    water_ref=mrs.water_ref,
    tissue_fracs={'gm': 0.55, 'wm': 0.35, 'csf': 0.10},
    te=mrs.te / 1000.0,       # seconds
    tr=mrs.tr / 1000.0,       # seconds
    align=True,
    reject=True,
    reject_threshold=3.0,
    paired_alignment=True,     # recommended
    gaba_fit_range=(2.7, 3.3),
    metab_t1=1.3,
    metab_t2=0.16,
    field_strength=mrs.field_strength,
)

# Primary results
print(f"GABA concentration: {results['gaba_conc_mM']:.2f} mM")
print(f"GABA/NAA ratio:     {results['gaba_naa_ratio']:.4f}")
print(f"GABA SNR:           {results['snr']:.1f}")
print(f"GABA CRLB:          {results['crlb_percent']:.1f} %")
print(f"Averages used:      {results['n_averages']}")
print(f"Transients rejected: {results['rejected'].sum()}")
```

The returned dictionary contains all quantification metrics, fit
parameters, and intermediate FIDs for downstream QC reporting.


## 6. Quality control

### QC report generation

NeuroJAX generates a self-contained HTML quality control report with
inline base64-encoded plots, following the recommendations of
{cite:t}`near2021preprocessing` for MRS reporting standards:

```python
from neurojax.analysis.mrs_qc import generate_qc_report

# Build the input dict from quantify_mega_press results
qc_input = {
    'diff':         results['diff_fid'],
    'edit_on':      results['edit_on_fid'],
    'edit_off':     results['edit_off_fid'],
    'sum_spec':     results['sum_spec_fid'],
    'freq_shifts':  results['freq_shifts'],
    'phase_shifts': results['phase_shifts'],
    'rejected':     results['rejected'],
    'n_averages':   results['n_averages'],
    'dwell_time':   mrs.dwell_time,
    'bandwidth':    1.0 / mrs.dwell_time,
    'centre_freq':  mrs.centre_freq,
}

# Metabolite concentrations for the report table
fitting_results = {
    'GABA+': {
        'concentration_mM': results['gaba_conc_mM'],
        'crlb_percent': results['crlb_percent'],
    },
}

html = generate_qc_report(
    qc_input,
    fitting_results=fitting_results,
    title="MEGA-PRESS QC Report — Subject 01",
)

# Write to file
from pathlib import Path
Path("qc_report.html").write_text(html)
print("QC report written to qc_report.html")
```

The report includes:

- **Spectra plot** --- Edit-ON, edit-OFF, and difference spectra
  (0.5--4.5 ppm range) for visual inspection of spectral quality.
- **Alignment metrics** --- Per-transient frequency and phase shifts,
  showing drift patterns and the effectiveness of spectral registration.
- **Outlier rejection summary** --- Number and percentage of rejected
  transients, with specific indices.
- **Metabolite concentration table** --- Fitted concentrations and CRLBs
  for all quantified metabolites.
- **Acquisition summary** --- Dwell time, bandwidth, spectral points,
  and total transient count.

### Key QC metrics to check

Following the consensus recommendations of {cite:t}`near2021preprocessing`,
inspect these metrics when evaluating data quality:

```{list-table} QC acceptance criteria
:header-rows: 1

* - Metric
  - Acceptable range
  - Notes
* - GABA+ SNR
  - > 10
  - Lower SNR increases CRLB; consider longer acquisitions
* - CRLB
  - < 20 %
  - Datasets with CRLB > 20% are unreliable for group analyses
* - Frequency drift (std)
  - < 5 Hz
  - Large drift indicates subject motion or B0 instability
* - Rejection rate
  - < 20 %
  - Higher rates suggest excessive motion or hardware problems
* - GABA+ linewidth (FWHM)
  - 0.07--0.15 ppm
  - Broader peaks may indicate shimming problems
* - GABA centre
  - 2.95--3.05 ppm
  - Off-centre peaks suggest frequency referencing errors
```


## 7. Tensor decomposition for multi-voxel MRS (advanced)

For multi-voxel or multi-coil MRS data, tensor decomposition provides a
principled framework for simultaneous coil combination, artifact
rejection, and spectral unmixing. The `neurojax.qmri.mrs_tensor` module
implements several methods from the chemometrics literature.

### Tucker / HOSVD decomposition

The Higher-Order SVD decomposes the 3-way MRS tensor
(spectral $\times$ coils $\times$ averages) into orthonormal subspaces
per mode and a dense core tensor ({cite:t}`delathauwer2000multilinear`):

```python
from neurojax.qmri.mrs_tensor import mrs_tucker_decomposition

# Shape: (n_spectral, n_coils, n_averages)
data_3way = mrs.data[:, :, 0, :]  # take edit-ON condition

tucker = mrs_tucker_decomposition(
    data_3way,
    ranks=(50, 4, 32),  # truncation per mode
)

print(f"Core tensor shape: {tucker['core'].shape}")
print(f"Explained variance: {tucker['explained_variance']:.4f}")

# Factor matrices: spectral, coil, and average subspaces
spectral_basis = tucker['factors'][0]   # (n_spec, 50)
coil_weights = tucker['factors'][1]     # (n_coils, 4)
temporal_basis = tucker['factors'][2]   # (n_averages, 32)
```

### PARAFAC / CP decomposition

PARAFAC decomposes the tensor into a sum of rank-1 components, each
representing an independent source signal with its coil sensitivity
profile and temporal evolution ({cite:t}`bro1997parafac`):

```python
from neurojax.qmri.mrs_tensor import mrs_parafac

cp = mrs_parafac(
    data_3way,
    n_components=5,
    n_iter=200,
    tol=1e-7,
)

print(f"Component weights: {cp['weights']}")
print(f"Iterations: {cp['n_iter']}")
print(f"Reconstruction error: {cp['rec_error']:.6f}")

# Factor matrices
spectral_profiles = cp['factors'][0]  # (n_spec, 5)
coil_profiles = cp['factors'][1]      # (n_coils, 5)
temporal_profiles = cp['factors'][2]  # (n_averages, 5)
```

### PARAFAC-based artifact rejection

Dominant PARAFAC components correspond to genuine metabolite signals,
while minor components capture artifacts (motion, lipid contamination,
hardware instabilities). Reconstructing from only the dominant
components provides data-driven artifact rejection:

```python
from neurojax.qmri.mrs_tensor import artifact_rejection_parafac

cleaned = artifact_rejection_parafac(
    data_3way,
    n_signal=3,    # keep 3 signal components
    n_total=5,     # fit 5 total components
    n_iter=200,
)

print(f"Cleaned data shape: {cleaned.shape}")
```

### Optimal coil weights from Tucker decomposition

Instead of using the first FID point for coil weighting, the rank-1
Tucker decomposition on the coil mode provides data-driven optimal
weights:

```python
from neurojax.qmri.mrs_tensor import optimal_coil_weights_from_tucker

weights = optimal_coil_weights_from_tucker(data_3way)
print(f"Coil weights shape: {weights.shape}")  # (n_coils,)

# Apply coil combination
combined = np.tensordot(weights.conj(), data_3way, axes=([0], [1]))
```

### MCR-ALS for spectral unmixing

Multivariate Curve Resolution resolves overlapping metabolite spectra
from a set of observations ({cite:t}`tauler1995multivariate`):

```python
from neurojax.qmri.mrs_tensor import mrs_mcr_als

# Input: (n_observations, n_spectral_points)
# e.g., spectra from multiple voxels or time points
spectra_matrix = np.real(
    np.fft.fftshift(np.fft.fft(data_3way[:, 0, :], axis=0), axes=0)
).T  # (n_averages, n_spec)

mcr = mrs_mcr_als(
    spectra_matrix,
    n_components=3,    # e.g., NAA, Cr, GABA
    n_iter=200,
    non_negative_spectra=True,
)

print(f"Resolved spectra shape: {mcr['spectra'].shape}")
print(f"Concentration profiles: {mcr['concentrations'].shape}")
print(f"Residual: {mcr['residual']:.6f}")
```


## Complete example: Big GABA dataset

The following script processes a single subject from the Big GABA
multi-site dataset ({cite:t}`mikkelsen2017big`) end-to-end:

```python
"""Complete MEGA-PRESS pipeline for Big GABA data."""
import numpy as np
from pathlib import Path

from neurojax.analysis.mrs_io import read_twix
from neurojax.analysis.mrs_quantify import quantify_mega_press
from neurojax.analysis.mrs_qc import generate_qc_report

# --- 1. Load data ---
mrs = read_twix(
    "/data/datasets/big_gaba/S5/S5_MP/S01/S01_GABA_68.dat",
    load_water_ref=True,
)
print(f"Loaded: {mrs.data.shape}, {mrs.n_coils} coils, "
      f"{mrs.n_averages} averages, TE={mrs.te} ms")

# --- 2. End-to-end quantification ---
results = quantify_mega_press(
    data=mrs.data,
    dwell_time=mrs.dwell_time,
    centre_freq=mrs.centre_freq,
    water_ref=mrs.water_ref,
    tissue_fracs={'gm': 0.55, 'wm': 0.35, 'csf': 0.10},
    te=mrs.te / 1000.0,
    tr=mrs.tr / 1000.0,
    align=True,
    reject=True,
    paired_alignment=True,
    field_strength=mrs.field_strength,
)

# --- 3. Report ---
print(f"\n--- GABA Quantification Results ---")
print(f"GABA concentration:   {results['gaba_conc_mM']:.2f} mM")
print(f"GABA/NAA ratio:       {results['gaba_naa_ratio']:.4f}")
print(f"GABA SNR:             {results['snr']:.1f}")
print(f"GABA CRLB:            {results['crlb_percent']:.1f} %")
print(f"GABA peak centre:     {results['gaba_centre_ppm']:.3f} ppm")
print(f"GABA FWHM:            {results['gaba_fwhm_ppm']:.3f} ppm")
print(f"Phase correction:     {np.degrees(results['phase_correction_rad']):.1f} deg")
print(f"Averages used:        {results['n_averages']}")
print(f"Transients rejected:  {results['rejected'].sum()}")
print(f"Mean freq drift:      {np.mean(results['freq_shifts']):.2f} Hz")

# --- 4. QC report ---
qc_input = {
    'diff':         results['diff_fid'],
    'edit_on':      results['edit_on_fid'],
    'edit_off':     results['edit_off_fid'],
    'sum_spec':     results['sum_spec_fid'],
    'freq_shifts':  results['freq_shifts'],
    'phase_shifts': results['phase_shifts'],
    'rejected':     results['rejected'],
    'n_averages':   results['n_averages'],
    'dwell_time':   mrs.dwell_time,
    'bandwidth':    1.0 / mrs.dwell_time,
    'centre_freq':  mrs.centre_freq,
}

fitting = {
    'GABA+': {
        'concentration_mM': results['gaba_conc_mM'],
        'crlb_percent': results['crlb_percent'],
    },
}

html = generate_qc_report(qc_input, fitting_results=fitting,
                           title="Big GABA S01 — MEGA-PRESS QC")
Path("big_gaba_s01_qc.html").write_text(html)
print("\nQC report saved to big_gaba_s01_qc.html")
```


## References

```{bibliography}
:filter: docname in docnames
```
