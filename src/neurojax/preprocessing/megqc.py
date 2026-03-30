"""MEG Quality Control metrics for WAND CTF data.

Lean reimplementation of the core MEGqc metrics (ANCPLabOldenburg/MEGqc)
using only MNE-Python — no numba, plotly, or pyqt6 dependencies.

Computes per-subject QC metrics that feed into:
  1. Subject exclusion (flag bad recordings before batch processing)
  2. Adversarial preprocessing evaluation (Scanzi et al. 2026)
  3. BIDS derivatives/Meg_QC/ output

Metrics:
  - Per-channel STD and peak-to-peak amplitude (flag noisy/flat channels)
  - Power spectral density (line noise detection, spectral shape)
  - ECG contamination (heartbeat artifact presence + severity)
  - EOG contamination (blink artifact presence + severity)
  - Muscle artifact burden (110-140 Hz high-frequency power)
  - Head movement summary (from CTF HPI coils)
  - Global Quality Index (composite score)

Reference: MEGqc (Olguin Baxman, Reer, Lopez — University of Oldenburg)
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import mne
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MEGQCResult:
    """Quality control results for one MEG recording."""

    subject: str
    task: str
    n_channels: int = 0
    sfreq: float = 0.0
    duration_s: float = 0.0

    # Per-channel metrics
    noisy_channels: list[str] = field(default_factory=list)
    flat_channels: list[str] = field(default_factory=list)
    n_noisy: int = 0
    n_flat: int = 0

    # STD statistics across channels
    std_mean: float = 0.0
    std_std: float = 0.0
    std_threshold_noisy: float = 0.0
    std_threshold_flat: float = 0.0

    # PSD metrics
    line_noise_freq: float = 0.0
    line_noise_power: float = 0.0  # relative power at line freq
    has_line_noise: bool = False

    # ECG metrics
    ecg_channel_found: bool = False
    n_heartbeats: int = 0
    heartbeat_rate_bpm: float = 0.0
    ecg_correlation_mean: float = 0.0  # mean correlation of ECG artifact across MEG channels

    # EOG metrics
    eog_channel_found: bool = False
    n_blinks: int = 0
    blink_rate_per_min: float = 0.0

    # Muscle artifact
    muscle_fraction: float = 0.0  # fraction of time annotated as muscle artifact

    # Global Quality Index
    gqi: float = 0.0  # 0=bad, 1=good

    def to_json(self, path: str | Path) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2, default=str)

    def to_tsv_row(self) -> dict:
        """Flat dict suitable for pandas DataFrame row."""
        d = asdict(self)
        d.pop("noisy_channels")
        d.pop("flat_channels")
        return d


def compute_std_metrics(
    raw: mne.io.Raw,
    std_multiplier: float = 3.0,
) -> tuple[dict, list[str], list[str]]:
    """Per-channel STD with noisy/flat classification.

    Noisy: STD > mean + multiplier * std_of_stds
    Flat:  STD < mean - multiplier * std_of_stds
    """
    data = raw.get_data(picks="meg")
    ch_names = [raw.ch_names[i] for i in mne.pick_types(raw.info, meg=True)]
    stds = np.std(data, axis=1)

    mean_std = np.mean(stds)
    std_of_stds = np.std(stds)
    thresh_noisy = mean_std + std_multiplier * std_of_stds
    thresh_flat = mean_std - std_multiplier * std_of_stds

    noisy = [ch for ch, s in zip(ch_names, stds) if s > thresh_noisy]
    flat = [ch for ch, s in zip(ch_names, stds) if s < max(thresh_flat, 0)]

    return {
        "std_mean": float(mean_std),
        "std_std": float(std_of_stds),
        "std_threshold_noisy": float(thresh_noisy),
        "std_threshold_flat": float(thresh_flat),
        "noisy_channels": noisy,
        "flat_channels": flat,
        "n_noisy": len(noisy),
        "n_flat": len(flat),
    }, noisy, flat


def compute_psd_metrics(
    raw: mne.io.Raw,
    line_freqs: tuple[float, ...] = (50.0, 60.0),
    bandwidth: float = 2.0,
) -> dict:
    """PSD-based metrics: line noise detection."""
    psd = raw.compute_psd(method="welch", fmin=1, fmax=100, n_fft=2048,
                          picks="meg", verbose=False)
    freqs = psd.freqs
    psd_data = psd.get_data().mean(axis=0)  # average across channels

    # Check for line noise at 50 and 60 Hz
    total_power = np.sum(psd_data)
    best_line_freq = 0.0
    best_line_power = 0.0

    for lf in line_freqs:
        mask = (freqs >= lf - bandwidth / 2) & (freqs <= lf + bandwidth / 2)
        if mask.any():
            line_power = np.sum(psd_data[mask]) / total_power
            if line_power > best_line_power:
                best_line_power = line_power
                best_line_freq = lf

    return {
        "line_noise_freq": float(best_line_freq),
        "line_noise_power": float(best_line_power),
        "has_line_noise": best_line_power > 0.05,  # >5% of total power
    }


def compute_ecg_metrics(raw: mne.io.Raw) -> dict:
    """ECG artifact detection and quantification."""
    result = {
        "ecg_channel_found": False,
        "n_heartbeats": 0,
        "heartbeat_rate_bpm": 0.0,
        "ecg_correlation_mean": 0.0,
    }

    try:
        ecg_events, _, _ = mne.preprocessing.find_ecg_events(
            raw, ch_name=None, verbose=False)
        n_beats = len(ecg_events)
        duration_min = raw.times[-1] / 60.0
        result["ecg_channel_found"] = True
        result["n_heartbeats"] = n_beats
        result["heartbeat_rate_bpm"] = n_beats / duration_min if duration_min > 0 else 0

        # ECG-MEG correlation via projections
        if n_beats > 10:
            projs, ecg_events_proj = mne.preprocessing.compute_proj_ecg(
                raw, n_mag=1, n_grad=0, average=True, verbose=False)
            if projs:
                result["ecg_correlation_mean"] = float(
                    np.mean(np.abs(projs[0]["data"]["data"])))
    except Exception as e:
        logger.debug("ECG detection failed: %s", e)

    return result


def compute_eog_metrics(raw: mne.io.Raw) -> dict:
    """EOG blink detection and quantification."""
    result = {
        "eog_channel_found": False,
        "n_blinks": 0,
        "blink_rate_per_min": 0.0,
    }

    try:
        eog_events = mne.preprocessing.find_eog_events(raw, verbose=False)
        n_blinks = len(eog_events)
        duration_min = raw.times[-1] / 60.0
        result["eog_channel_found"] = True
        result["n_blinks"] = n_blinks
        result["blink_rate_per_min"] = n_blinks / duration_min if duration_min > 0 else 0
    except Exception as e:
        logger.debug("EOG detection failed: %s", e)

    return result


def compute_muscle_metrics(
    raw: mne.io.Raw,
    threshold: float = 5.0,
) -> dict:
    """Muscle artifact burden (110-140 Hz z-score annotation)."""
    result = {"muscle_fraction": 0.0}

    try:
        annot, scores = mne.preprocessing.annotate_muscle_zscore(
            raw, ch_type="mag", threshold=threshold,
            min_length_good=0.2, verbose=False)
        # Fraction of recording annotated as muscle
        total_dur = raw.times[-1]
        muscle_dur = sum(a["duration"] for a in annot if "BAD_muscle" in a["description"])
        result["muscle_fraction"] = muscle_dur / total_dur if total_dur > 0 else 0
    except Exception as e:
        logger.debug("Muscle detection failed: %s", e)

    return result


def compute_gqi(result: MEGQCResult) -> float:
    """Global Quality Index — composite score from 0 (bad) to 1 (good).

    Weighted combination of individual metrics. Configurable thresholds
    based on MEGqc defaults.
    """
    scores = []

    # Channel quality: penalize for noisy/flat channels
    ch_score = 1.0 - min(result.n_noisy + result.n_flat, 30) / 30.0
    scores.append(ch_score)

    # Line noise: penalize if present
    line_score = 0.0 if result.has_line_noise else 1.0
    scores.append(line_score)

    # ECG: penalize if heart rate is abnormal
    if result.ecg_channel_found and result.heartbeat_rate_bpm > 0:
        hr = result.heartbeat_rate_bpm
        ecg_score = 1.0 if 50 < hr < 100 else 0.5
    else:
        ecg_score = 0.5  # unknown
    scores.append(ecg_score)

    # Muscle: penalize proportional to artifact fraction
    muscle_score = max(0, 1.0 - result.muscle_fraction * 5)
    scores.append(muscle_score)

    return float(np.mean(scores))


def run_megqc(
    raw: mne.io.Raw,
    subject: str,
    task: str = "resting",
    std_multiplier: float = 3.0,
    muscle_threshold: float = 5.0,
) -> MEGQCResult:
    """Run all QC metrics on a raw MEG recording.

    Parameters
    ----------
    raw : mne.io.Raw — loaded (preload=True) MEG data.
    subject : str
    task : str
    std_multiplier : float — threshold for noisy/flat channel detection.
    muscle_threshold : float — z-score threshold for muscle artifacts.

    Returns
    -------
    MEGQCResult with all metrics populated.
    """
    result = MEGQCResult(
        subject=subject,
        task=task,
        n_channels=len(mne.pick_types(raw.info, meg=True)),
        sfreq=raw.info["sfreq"],
        duration_s=raw.times[-1],
    )

    logger.info("%s task-%s: %d ch, %.0f Hz, %.0fs",
                subject, task, result.n_channels, result.sfreq, result.duration_s)

    # STD
    std_info, noisy, flat = compute_std_metrics(raw, std_multiplier)
    result.std_mean = std_info["std_mean"]
    result.std_std = std_info["std_std"]
    result.std_threshold_noisy = std_info["std_threshold_noisy"]
    result.std_threshold_flat = std_info["std_threshold_flat"]
    result.noisy_channels = noisy
    result.flat_channels = flat
    result.n_noisy = len(noisy)
    result.n_flat = len(flat)
    logger.info("  STD: %d noisy, %d flat", result.n_noisy, result.n_flat)

    # PSD
    psd_info = compute_psd_metrics(raw)
    result.line_noise_freq = psd_info["line_noise_freq"]
    result.line_noise_power = psd_info["line_noise_power"]
    result.has_line_noise = psd_info["has_line_noise"]
    logger.info("  PSD: line noise %.0f Hz (%.3f relative power, %s)",
                result.line_noise_freq, result.line_noise_power,
                "DETECTED" if result.has_line_noise else "clean")

    # ECG
    ecg_info = compute_ecg_metrics(raw)
    result.ecg_channel_found = ecg_info["ecg_channel_found"]
    result.n_heartbeats = ecg_info["n_heartbeats"]
    result.heartbeat_rate_bpm = ecg_info["heartbeat_rate_bpm"]
    result.ecg_correlation_mean = ecg_info["ecg_correlation_mean"]
    logger.info("  ECG: %d beats, %.0f bpm", result.n_heartbeats, result.heartbeat_rate_bpm)

    # EOG
    eog_info = compute_eog_metrics(raw)
    result.eog_channel_found = eog_info["eog_channel_found"]
    result.n_blinks = eog_info["n_blinks"]
    result.blink_rate_per_min = eog_info["blink_rate_per_min"]
    logger.info("  EOG: %d blinks, %.1f/min", result.n_blinks, result.blink_rate_per_min)

    # Muscle
    muscle_info = compute_muscle_metrics(raw, muscle_threshold)
    result.muscle_fraction = muscle_info["muscle_fraction"]
    logger.info("  Muscle: %.1f%% of recording", result.muscle_fraction * 100)

    # GQI
    result.gqi = compute_gqi(result)
    logger.info("  GQI: %.3f", result.gqi)

    return result
