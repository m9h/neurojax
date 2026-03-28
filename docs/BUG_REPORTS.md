# Bug Reports — Issues Found During WAND Processing

Tracking bugs found in external tools during WAND pipeline development.
File upstream when ready, or fix locally as needed.

---

## 1. FreeSurfer 8.2.0 — qatools.py screenshots crash (scipy sparse indexing)

**Tool:** FreeSurfer 8.2.0 `qatools.py` (`--screenshots` and `--fornix` flags)
**File:** `/Applications/freesurfer/8.2.0/python/packages/qatoolspython/qatoolspythonUtils.py` line 119
**Error:**
```
File "qatoolspythonUtils.py", line 119, in levelsetsTria
    if np.count_nonzero(A[t[n[i], oi], t[n[i], oix[0]]]) == 0:
File "scipy/sparse/_lil.py", line 287, in _get_row_ranges
    new = self._lil_container((len(rows), nj), dtype=self.dtype)
TypeError: len() of unsized object
```
**Cause:** scipy sparse matrix API change — scalar index produces 0-d array instead of scalar. `len()` fails on 0-d array.
**Affects:** `--screenshots` and `--fornix` flags (both call `createScreenshots`)
**Workaround:** Run qatools without `--screenshots --fornix` — metrics still work correctly.
**Fix needed:** Cast scalar index to int before sparse matrix indexing in `levelsetsTria()`.
**Upstream:** https://github.com/Deep-MI/qatools-python (or FreeSurfer Jira)
**Status:** Not yet reported
**Severity:** Medium — metrics work, only screenshot generation broken

---

## 2. FreeSurfer 8.2.0 — qatools.py pandas deprecation (warn_bad_lines)

**Tool:** FreeSurfer 8.2.0 `qatools.py` (`--screenshots` flag)
**File:** `/Applications/freesurfer/8.2.0/python/packages/qatoolspython/createScreenshots.py` line 113-114
**Error:**
```
TypeError: read_csv() got an unexpected keyword argument 'warn_bad_lines'
```
**Cause:** pandas removed `warn_bad_lines` parameter (deprecated since pandas 1.3, removed in 1.4+). The file already has `on_bad_lines='skip'` on line 113 but also had the old `warn_bad_lines=True` on line 114.
**Fix applied locally:** `sudo sed -i '' '114d' .../createScreenshots.py` (removed duplicate line)
**Note:** This bug is hit before the scipy bug (#1), so both must be fixed for screenshots to work.
**Upstream:** Same as #1
**Status:** Fixed locally, not yet reported upstream
**Severity:** Low — easy one-line fix

---

## 3. neurojax — RWW model gamma_E/gamma_I units (FIXED)

**Tool:** neurojax `bench/models/rww.py`
**Bug:** `gamma_E = 0.641/1000` and `gamma_I = 1.0/1000` — the `/1000` was a unit confusion. With tau in seconds, gamma should also be in 1/s units, not 1/ms.
**Symptom:** Single-node RWW model converged to S_E ≈ 0.0002 instead of expected ~0.164 (Deco 2013 fixed point).
**Fix:** Changed to `gamma_E = 0.641` and `gamma_I = 1.0`. Fixed point now matches literature.
**Status:** Fixed in commit `49acac2`
**Severity:** High — incorrect model dynamics

---

## 4. neurojax — HMM Viterbi T=1 crash (FIXED)

**Tool:** neurojax `models/hmm.py`
**Bug:** Viterbi decoding with T=1 (single timepoint) crashed because `jax.lax.scan` on empty `log_B[1:]` produced empty arrays, then `log_deltas[-1]` indexed out of bounds.
**Fix:** Added early return before scan: `if T == 1: return jnp.array([jnp.argmax(log_delta_0)])`
**Status:** Fixed in commit `284cd67`
**Severity:** Medium — edge case crash

---

## 5. neurojax — filter.py used non-existent jax.scipy.signal.lfilter (FIXED)

**Tool:** neurojax `preprocessing/filter.py`
**Bug:** Called `jax.scipy.signal.lfilter` which does not exist in JAX.
**Fix:** Replaced with a `jax.lax.scan`-based Direct-Form II Transposed IIR filter. Also fixed zero-order edge case (identity filter with empty state vector).
**Status:** Fixed in commit `284cd67`
**Severity:** High — module was completely non-functional

---

## 6. QUIT v3.4 — bundled scipy/pandas compatibility (POTENTIAL)

**Tool:** QUIT v3.4 (if using Python wrappers `qipype`)
**Note:** QUIT itself is C++ and works fine. But the Python nipype wrappers may have compatibility issues with newer Python environments. Not yet confirmed — noting for reference.
**Status:** Not yet encountered
**Severity:** Unknown

---

## 7. sbi4dwi — jaxlib platform mismatch on macOS x86_64

**Tool:** sbi4dwi `uv.lock` pins `jaxlib==0.8.1` which has no macOS x86_64 wheel
**Error:** `Distribution jaxlib==0.8.1 can't be installed because it doesn't have a wheel for the current platform`
**Workaround:** Run sbi4dwi tests from the neurojax environment with `PYTHONPATH` pointing to sbi4dwi.
**Fix needed:** Update `pyproject.toml` to allow jaxlib version range compatible with macOS x86_64, or add platform to `tool.uv.required-environments`.
**Status:** Known, not yet fixed
**Severity:** Medium — blocks direct `uv run` in sbi4dwi on this Mac

---

## Reporting Template

When filing upstream:

```
Title: [Tool] Brief description
Version: X.Y.Z
Platform: macOS 26.3 x86_64 / Linux
Steps to reproduce:
  1. ...
  2. ...
Expected: ...
Actual: ...
Traceback: ...
Fix (if known): ...
```
