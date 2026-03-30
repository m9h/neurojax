# FSL-MRS Upstream Issues and Feature Requests

Discovered during WAND dataset processing with FSL-MRS 2.4.12.
Filed from neurojax project (github.com/m9h/neurojax).

Contact: Morgan Hough

---

## Bug Reports

### 1. spec2nii: PatientName validation crash on numeric TWIX IDs

**Severity:** Blocker — prevents TWIX conversion for WAND dataset

**Reproduction:**
```bash
spec2nii twix -e image -o output/ input.dat
```

**Error:**
```
nifti_mrs.validator.headerExtensionError: PatientName must be a (<class 'str'>,).
PatientName is a <class 'float'>, with value 31408033.0.
```

**Cause:** WAND Siemens TWIX files store PatientName as a numeric float (the subject ID). The `nifti_mrs` validator enforces `str` type before `spec2nii` can apply `--anon`.

**Workaround:** Monkey-patch `nifti_mrs.validator.validate_hdr_ext` to coerce PatientName to str before validation:
```python
import nifti_mrs.validator as validator
orig = validator.validate_hdr_ext
def patched(hdr_ext, *a, **kw):
    if 'PatientName' in hdr_ext:
        hdr_ext['PatientName'] = str(hdr_ext['PatientName'])
    return orig(hdr_ext, *a, **kw)
validator.validate_hdr_ext = patched
```

**Suggested fix:** In `spec2nii` or `nifti_mrs`, coerce PatientName to str automatically, or apply `--anon` before validation.

**Affected versions:** spec2nii 0.8.7, nifti_mrs (bundled with FSL-MRS 2.4.12)

---

### 2. fsl_mrs_sim: simseq RFUnits='deg' not supported

**Severity:** Documentation gap / usability

**Reproduction:**
```python
from fsl_mrs.denmatsim.simseq import simseq
params = {..., 'RFUnits': 'deg', ...}
simseq(spinsys, params)
```

**Error:**
```
ValueError: Unknown RFUnits deg.
```

**Expected behaviour:** `deg` (degrees) should be accepted for ideal hard pulses, converting internally via `amp_Hz = flip_deg / (360 * pulse_duration)`.

**Actual behaviour:** Only accepts 'Hz', 'T', 'mT', 'uT'. Users must manually convert flip angles to Hz, which requires knowing the pulse duration — unintuitive for ideal pulse simulations.

**Suggested fix:** Add `deg` as an accepted RFUnits value with automatic conversion using the pulse `time` parameter:
```python
elif RFUnits.lower() == 'deg':
    amp_hz = amp / (360.0 * pulse_time)
```

---

### 3. fsl_mrs_sim: simseq underdocumented parameterdict format

**Severity:** Major usability gap

**Description:** The `simseq()` function requires a complex parameterdict with keys like `RF`, `delays`, `rephaseAreas`, `CoherenceFilter`, but there are:
- No example JSON files shipped with FSL-MRS
- No documentation of the expected structure for common sequences (PRESS, STEAM, sLASER)
- No helper functions to construct a parameterdict from high-level parameters (TE, sequence_type)

The `grumble()` validation function provides some hints but the expected shapes and types for `RF` list-of-dicts, `CoherenceFilter` nested lists, and gradient units are not documented.

**Suggested fix:**
1. Ship example sequence JSON files: `press_ideal.json`, `steam_ideal.json`, `slaser_ideal.json`
2. Add a helper: `make_press_params(TE, B0, BW, npts)` that constructs a valid parameterdict
3. Document the parameterdict schema in the FSL-MRS docs

---

### 4. fsl_mrs_sim: spinSystems.json incompatible with simseq()

**Severity:** Integration bug

**Reproduction:**
```python
import json
from fsl_mrs.denmatsim.simseq import simseq

with open('.../spinSystems.json') as f:
    ss = json.load(f)

naa = ss['sysNAA']  # Returns a list of dicts
simseq(naa[0], params)  # Fails with type errors
```

**Error:** Various — `unsupported operand type(s) for /: 'list' and 'int'`, shape mismatches

**Cause:** The `spinSystems.json` shipped with FSL-MRS stores spin systems as lists of sub-system dicts (e.g., NAA has acetyl + aspartyl), but `simseq()` expects a single spin system dict with specific key formatting. The conversion between the JSON format and what `simseq()` actually consumes is not documented or automated.

**Suggested fix:** Add a loader function:
```python
def load_spin_system(name):
    """Load spin system from built-in database, ready for simseq()."""
```

---

### 5. basis_tools: JMRUI .txt format parsing failure

**Severity:** Minor

**Reproduction:**
```bash
fsl_mrs --basis /path/to/basisset_JMRUI/ ...
```

**Error:**
```
ValueError: need at least one array to concatenate
```
in `readjMRUItxt()` when reading the JMRUI-format basis files shipped in `pkg_data/mrs_fitting_challenge/basisset_JMRUI/`.

**Cause:** The built-in JMRUI basis files appear to be in a format incompatible with the current reader in FSL-MRS 2.4.12.

---

### 6. fsl_mrs: BasisHasInsufficientCoverage with matched dwell times

**Severity:** Medium

**Reproduction:** Create basis JSON files with dwell time matching data exactly (from NIfTI-MRS header), then run fsl_mrs.

**Error:**
```
BasisHasInsufficientCoverage: The basis spectra covers too little time.
```

**Cause:** Floating-point precision mismatch between data dwell (166700ns → 0.0001667000s) and basis dwell (1/6000 = 0.00016666...s). The comparison uses strict equality or tight tolerance.

**Workaround:** Generate basis with exactly the same dwell time extracted from the NIfTI-MRS data object:
```python
dwell = 1.0 / data.bandwidth  # Use data's own bandwidth
```

**Suggested fix:** Use a tolerance of ~0.1% when comparing dwell times between data and basis.

---

## Feature Requests

### 7. fsl_mrs_sim: High-level sequence simulation functions

**Request:** Add convenience functions for common sequences:
```python
from fsl_mrs.denmatsim import simulate_press, simulate_slaser, simulate_mega

basis = simulate_press(TE=0.035, B0=3.0, metabolites='default', BW=2000, npts=2048)
basis.save('press_3t_te35.basis')
```

This would make basis set generation accessible without understanding the low-level parameterdict format.

---

### 8. spec2nii: Automatic PatientName sanitisation

**Request:** Before NIfTI-MRS header validation, automatically coerce all header fields to their expected types (str for PatientName, float for numeric fields, etc.) rather than raising a validation error. Alternatively, apply `--anon` processing before validation.

---

### 9. fsl_mrs Python API: Programmatic fitting example

**Request:** The CLI tools are well-documented but the Python API (`fsl_mrs.core.MRS`, `fsl_mrs.utils.fitting`) lacks examples for programmatic use. A Jupyter notebook showing:
1. Load NIfTI-MRS data
2. Create/load basis set
3. Run fitting programmatically
4. Extract concentrations and CRLBs

would significantly improve usability for pipeline integration.

---

## Environment

- FSL-MRS: 2.4.12
- spec2nii: 0.8.7
- FSL: 6.0.7.22
- Python: 3.12
- Platform: macOS Darwin 25.3.0 x86_64
- Data: WAND (Welsh Advanced Neuroimaging Database), Siemens Connectom 3T / Prisma 3T
