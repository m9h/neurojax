---
name: neurojax_asr_validation
description: Validate Artifact Subspace Reconstruction (ASR) using the SSVEP with Artifact Trials dataset (ds004745).
---

# ASR Validation Skill: SSVEP & Artifacts

## Objective
Implement and validate **Artifact Subspace Reconstruction (ASR)** (specifically the `clean_rawdata` algorithm) using the **8-Channel SSVEP EEG Dataset with Artifact Trials** (`ds004745`).

The core goal is to demonstrate that ASR can remove large-amplitude muscle and motion artifacts while **preserving the underlying steady-state visually evoked potentials (SSVEP)** at 2 Hz, 4 Hz, and 8 Hz.

## Dataset Context
- **Dataset ID:** `ds004745`
- **Features:**
    - **SSVEP Task:** Users watch flickering lights at 2, 4, 8 Hz.
    - **Artifact Trials:** Users intentionally perform artifacts (head movement, jaw clench, eye blink) *during* the task.
    - **Channels:** 8 (Low channel count is a stress test for ASR).

## Workflow Instructions

### 1. Data Ingestion
1.  **Download:** Use `datalad` to download `ds004745`.
    ```bash
    datalad install https://github.com/OpenNeuroDatasets/ds004745.git
    cd ds004745
    datalad get sub-001/ses-01/eeg/*
    ```
2.  **Load:** Load the raw EEG data (`.set` / `.eeg`). Note that some OpenNeuro datasets use EEGLAB format.
3.  **Locate Artifacts:** Identify the trials/events marked as "Artifact" or "Movement". If explicit events are missing, inspect the time series for gross deviations (> 100 µV).

### 2. Implementation: JAX ASR
Implement the `clean_rawdata` algorithm in JAX using `neurojax.preprocessing`. If not present, you must implement:
1.  **Calibration:** Calculate the geometric median and robust standard deviation (Huber/median absolute deviation) of the clean portions of the data (or a separate resting state file).
2.  **PCA Reconstruction:**
    -   Compute the sliding window PCA.
    -   Identify components with variance > `cutoff` standard deviations (typically 5-20).
    -   Reconstruct these high-variance components from the remaining subspace.

### 3. Verification & Validation (The "Test")
You must produce a `validation_report.md` containing:
1.  **Time-Domain Comparison:**
    -   Plot `Raw` vs `ASR-Cleaned` waveforms for an artifact segment.
    -   *Success Metric:* Amplitude of artifact segments should be reduced to within physiological range (< 50-100 µV).
2.  **Frequency-Domain Comparison:**
    -   Compute the Power Spectral Density (PSD) for `Raw` vs `ASR-Cleaned` data.
    -   *Success Metric:* The SSVEP peaks at **2 Hz, 4 Hz, and 8 Hz** must remain visible and distinct in the cleaned data. They should NOT be attenuated significantly compared to non-artifact periods.
3.  **Quantitative Metric:**
    -   **Signal-to-Noise Ratio (SNR):** Calculate SNR of the SSVEP peaks before and after ASR.
    -   *Goal:* SNR should increase (or at least decrease minimally) after cleaning.

## Constraints
- **Low Channel Count:** The dataset has only 8 channels. ASR typically relies on spatial redundancy. You may need to tune the `cutoff` parameter (e.g., set it looser, around 20-30 SD) to avoid deleting real signal.
- **JAX Compliance:** All computations must be JAX-compatible for GPU acceleration.
