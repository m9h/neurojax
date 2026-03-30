"""Adversarial preprocessing pipeline evaluation for MEG.

Implements the Scanzi et al. (2026) approach: inject known ground truth
signals into real MEG data, then measure how well each preprocessing
pipeline preserves the injected signal while removing real artifacts.

This decouples pipeline selection from downstream analysis — the optimal
pipeline is the one that maximizes signal recovery, not the one that
produces the "best" HMM states or classification accuracy.

Workflow:
  1. MEGqc characterizes artifact profiles (STD, ECG, EOG, muscle, line noise)
  2. Generate realistic simulated neural signals (oscillatory, transient, ERP-like)
  3. Inject signals at known amplitudes into real contaminated MEG data
  4. Run candidate preprocessing pipelines
  5. Measure ground truth signal recovery (SNR, correlation, waveform distortion)
  6. Select pipeline that maximizes recovery per artifact profile

Reference: Scanzi et al. (2026) "An adversarial approach to guide the
selection of preprocessing pipelines for ERP studies" bioRxiv.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

import mne
import numpy as np

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------
# Ground truth signal generators
# -------------------------------------------------------------------------

def make_oscillatory_signal(
    n_times: int,
    sfreq: float,
    freq: float = 10.0,
    amplitude: float = 1e-13,
    phase: float = 0.0,
    envelope: str = "constant",
) -> np.ndarray:
    """Generate an oscillatory ground truth signal.

    Parameters
    ----------
    n_times : int
    sfreq : float — sampling frequency
    freq : float — oscillation frequency (Hz)
    amplitude : float — peak amplitude (Tesla for MEG)
    phase : float — initial phase (radians)
    envelope : str — 'constant', 'gaussian', or 'burst'

    Returns
    -------
    signal : (n_times,)
    """
    t = np.arange(n_times) / sfreq
    carrier = np.sin(2 * np.pi * freq * t + phase)

    if envelope == "constant":
        env = np.ones(n_times)
    elif envelope == "gaussian":
        center = n_times // 2
        sigma = n_times / 6
        env = np.exp(-0.5 * ((np.arange(n_times) - center) / sigma) ** 2)
    elif envelope == "burst":
        # Random bursts (3-5 per recording)
        env = np.zeros(n_times)
        n_bursts = np.random.randint(3, 6)
        burst_len = int(0.5 * sfreq)  # 500ms bursts
        for _ in range(n_bursts):
            start = np.random.randint(0, max(1, n_times - burst_len))
            burst_env = np.hanning(burst_len)
            end = min(start + burst_len, n_times)
            env[start:end] = burst_env[:end - start]
    else:
        env = np.ones(n_times)

    return amplitude * carrier * env


def make_erp_signal(
    n_times: int,
    sfreq: float,
    latency: float = 0.1,
    width: float = 0.05,
    amplitude: float = 1e-13,
) -> np.ndarray:
    """Generate an ERP-like transient signal.

    Parameters
    ----------
    n_times : int
    sfreq : float
    latency : float — peak latency in seconds
    width : float — Gaussian width in seconds
    amplitude : float — peak amplitude

    Returns
    -------
    signal : (n_times,)
    """
    t = np.arange(n_times) / sfreq
    return amplitude * np.exp(-0.5 * ((t - latency) / width) ** 2)


def make_multi_frequency_signal(
    n_times: int,
    sfreq: float,
    freqs: tuple[float, ...] = (4.0, 10.0, 22.0),
    amplitudes: Optional[tuple[float, ...]] = None,
) -> np.ndarray:
    """Generate a multi-frequency ground truth signal.

    Realistic: theta + alpha + beta components.
    """
    if amplitudes is None:
        amplitudes = tuple(1e-13 / (i + 1) for i in range(len(freqs)))
    t = np.arange(n_times) / sfreq
    signal = np.zeros(n_times)
    for freq, amp in zip(freqs, amplitudes):
        phase = np.random.uniform(0, 2 * np.pi)
        signal += amp * np.sin(2 * np.pi * freq * t + phase)
    return signal


# -------------------------------------------------------------------------
# Signal injection
# -------------------------------------------------------------------------

def inject_signal(
    raw: mne.io.Raw,
    signal: np.ndarray,
    channels: Optional[list[str]] = None,
    spatial_pattern: Optional[np.ndarray] = None,
) -> tuple[mne.io.Raw, np.ndarray]:
    """Inject a ground truth signal into real MEG data.

    Parameters
    ----------
    raw : mne.io.Raw — real MEG data (will be copied)
    signal : (n_times,) — ground truth time course
    channels : list of channel names to inject into (default: all MEG)
    spatial_pattern : (n_channels,) — per-channel amplitude weights.
        If None, uses a realistic dipolar pattern from the center of
        the sensor array.

    Returns
    -------
    raw_injected : mne.io.Raw — raw data with signal added
    injected_data : (n_channels, n_times) — the exact injected signal
        for recovery measurement
    """
    raw_out = raw.copy()
    meg_picks = mne.pick_types(raw.info, meg=True)

    if channels is not None:
        picks = mne.pick_channels(raw.ch_names, channels)
    else:
        picks = meg_picks

    n_channels = len(picks)
    n_times = min(len(signal), raw.n_times)

    if spatial_pattern is None:
        # Dipolar pattern: smooth spatial variation across sensors
        locs = np.array([raw.info["chs"][p]["loc"][:3] for p in picks])
        center = locs.mean(axis=0)
        dists = np.linalg.norm(locs - center, axis=1)
        if dists.max() > 1e-10:
            spatial_pattern = np.cos(np.pi * dists / (2 * dists.max()))
        else:
            # Fallback for synthetic data without real sensor positions
            spatial_pattern = np.ones(n_channels)

    spatial_pattern = spatial_pattern / (np.max(np.abs(spatial_pattern)) + 1e-20)

    # Build the injected data matrix
    injected = np.outer(spatial_pattern, signal[:n_times])  # (n_ch, n_times)

    # Build new RawArray with injected signal
    data = raw.get_data().copy()
    data[picks, :n_times] += injected
    raw_out = mne.io.RawArray(data, raw.info.copy(), verbose=False)

    return raw_out, injected


# -------------------------------------------------------------------------
# Pipeline evaluation
# -------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """A preprocessing pipeline configuration to evaluate."""
    name: str
    func: Callable[[mne.io.Raw], mne.io.Raw]
    description: str = ""


@dataclass
class RecoveryMetrics:
    """Signal recovery metrics for one pipeline × one signal type."""
    pipeline: str
    signal_type: str
    snr_input: float = 0.0     # SNR of injected signal in raw data
    snr_output: float = 0.0    # SNR of recovered signal after preprocessing
    snr_improvement: float = 0.0
    correlation: float = 0.0   # Pearson r between injected and recovered
    rmse: float = 0.0          # RMS error of recovered vs injected
    amplitude_ratio: float = 0.0  # recovered amplitude / injected amplitude
    waveform_distortion: float = 0.0  # 1 - correlation (0=perfect, 1=destroyed)


def measure_recovery(
    raw_clean: mne.io.Raw,
    raw_original: mne.io.Raw,
    injected: np.ndarray,
    picks: np.ndarray,
) -> dict:
    """Measure how well the preprocessing preserved the injected signal.

    Parameters
    ----------
    raw_clean : mne.io.Raw — preprocessed data (with injected signal)
    raw_original : mne.io.Raw — preprocessed data (without injected signal)
    injected : (n_channels, n_times) — the known injected signal
    picks : channel indices

    Returns
    -------
    dict with SNR, correlation, RMSE, amplitude ratio, distortion
    """
    n_times = injected.shape[1]

    # Recovered = (preprocessed with signal) - (preprocessed without signal)
    data_with = raw_clean.get_data()[picks, :n_times]
    data_without = raw_original.get_data()[picks, :n_times]
    recovered = data_with - data_without

    # Per-channel metrics, then average
    correlations = []
    for ch in range(len(picks)):
        inj = injected[ch]
        rec = recovered[ch]
        if np.std(inj) > 0 and np.std(rec) > 0:
            r = np.corrcoef(inj, rec)[0, 1]
            correlations.append(r)

    mean_corr = np.mean(correlations) if correlations else 0.0

    # Global metrics
    inj_power = np.mean(injected ** 2)
    rec_power = np.mean(recovered ** 2)
    noise_power = np.mean((recovered - injected) ** 2)

    snr_out = 10 * np.log10(inj_power / max(noise_power, 1e-30))
    rmse = np.sqrt(np.mean((recovered - injected) ** 2))
    amp_ratio = np.sqrt(rec_power / max(inj_power, 1e-30))

    return {
        "correlation": float(mean_corr),
        "snr_output_db": float(snr_out),
        "rmse": float(rmse),
        "amplitude_ratio": float(amp_ratio),
        "waveform_distortion": float(1 - mean_corr),
    }


def evaluate_pipelines(
    raw: mne.io.Raw,
    pipelines: list[PipelineConfig],
    signal_types: Optional[dict[str, np.ndarray]] = None,
    n_trials: int = 3,
    seed: int = 42,
) -> list[RecoveryMetrics]:
    """Evaluate multiple preprocessing pipelines via adversarial injection.

    For each pipeline × signal type × trial:
    1. Inject ground truth signal into raw data
    2. Run the pipeline on both injected and original data
    3. Measure signal recovery

    Parameters
    ----------
    raw : mne.io.Raw — real MEG data
    pipelines : list of PipelineConfig
    signal_types : dict mapping signal name → (n_times,) array.
        If None, generates default set (alpha, theta+alpha+beta, ERP, burst).
    n_trials : int — repetitions per signal type (different spatial patterns)
    seed : int

    Returns
    -------
    list of RecoveryMetrics — one per pipeline × signal type
    """
    rng = np.random.default_rng(seed)

    # Pick MEG channels and resample to target sfreq FIRST
    # so injection and comparison happen at the same sample rate/length.
    raw = raw.copy().pick(picks="meg", exclude="bads")
    target_sfreq = pipelines[0].func.__defaults__  # inspect isn't reliable, just resample
    raw.resample(250.0, verbose=False)  # standardize before injection
    n_times = raw.n_times
    sfreq = raw.info["sfreq"]
    n_channels = len(raw.ch_names)
    meg_picks = np.arange(n_channels)  # all channels are MEG now

    if signal_types is None:
        signal_types = {
            "alpha_10Hz": make_oscillatory_signal(n_times, sfreq, freq=10.0),
            "multi_freq": make_multi_frequency_signal(n_times, sfreq),
            "alpha_burst": make_oscillatory_signal(n_times, sfreq, freq=10.0, envelope="burst"),
        }

    # Preprocess the original (no injection) once per pipeline
    logger.info("Preprocessing original data for %d pipelines...", len(pipelines))
    originals = {}
    for pipeline in pipelines:
        try:
            originals[pipeline.name] = pipeline.func(raw.copy())
        except Exception as e:
            logger.warning("Pipeline '%s' failed on original: %s", pipeline.name, e)

    results = []

    for sig_name, signal in signal_types.items():
        logger.info("Signal type: %s", sig_name)

        for trial in range(n_trials):
            # Random spatial pattern per trial
            n_ch = len(meg_picks)
            spatial = rng.standard_normal(n_ch)
            spatial = spatial / np.max(np.abs(spatial))

            # Inject
            raw_injected, injected_data = inject_signal(
                raw, signal, spatial_pattern=spatial)

            # Input SNR
            orig_data = raw.get_data()[meg_picks]
            noise_power = np.mean(orig_data ** 2)
            signal_power = np.mean(injected_data ** 2)
            snr_input = 10 * np.log10(signal_power / max(noise_power, 1e-30))

            for pipeline in pipelines:
                if pipeline.name not in originals:
                    continue

                try:
                    raw_cleaned = pipeline.func(raw_injected.copy())

                    metrics = measure_recovery(
                        raw_cleaned, originals[pipeline.name],
                        injected_data, meg_picks)

                    rm = RecoveryMetrics(
                        pipeline=pipeline.name,
                        signal_type=sig_name,
                        snr_input=snr_input,
                        snr_output=metrics["snr_output_db"],
                        snr_improvement=metrics["snr_output_db"] - snr_input,
                        correlation=metrics["correlation"],
                        rmse=metrics["rmse"],
                        amplitude_ratio=metrics["amplitude_ratio"],
                        waveform_distortion=metrics["waveform_distortion"],
                    )
                    results.append(rm)
                    logger.info("  %s trial %d: corr=%.3f snr=%.1fdB distortion=%.3f",
                                pipeline.name, trial, rm.correlation,
                                rm.snr_output, rm.waveform_distortion)

                except Exception as e:
                    logger.warning("  %s trial %d failed: %s", pipeline.name, trial, e)

    return results


def evaluate_pipelines_snr_sweep(
    raw: mne.io.Raw,
    pipelines: list[PipelineConfig],
    snr_levels_db: list[float] = [-20, -10, -5, 0, 5, 10, 20],
    signal_freqs: tuple[float, ...] = (10.0,),
    n_trials: int = 3,
    seed: int = 42,
) -> list[RecoveryMetrics]:
    """Evaluate pipelines across a range of injection SNR levels.

    Scales signal amplitude relative to the actual data noise floor so that
    the injected signal has a specific SNR. At low SNR (e.g., -20 dB),
    only pipelines that effectively remove artifacts will recover the signal.

    Parameters
    ----------
    raw : mne.io.Raw
    pipelines : list of PipelineConfig
    snr_levels_db : list of float — target SNR in dB
    signal_freqs : frequencies for oscillatory test signals
    n_trials : int — repetitions per SNR level
    seed : int

    Returns
    -------
    list of RecoveryMetrics with snr_input set to the target SNR level
    """
    rng = np.random.default_rng(seed)

    # Pre-pick and resample
    raw = raw.copy().pick(picks="meg", exclude="bads")
    if raw.info["sfreq"] > 250.0:
        raw.resample(250.0, verbose=False)
    n_times = raw.n_times
    sfreq = raw.info["sfreq"]
    n_channels = len(raw.ch_names)
    meg_picks = np.arange(n_channels)

    # Measure noise floor (RMS of MEG data)
    noise_rms = np.sqrt(np.mean(raw.get_data() ** 2))

    # Preprocess originals
    logger.info("Preprocessing %d pipelines on original data...", len(pipelines))
    originals = {}
    for p in pipelines:
        try:
            originals[p.name] = p.func(raw.copy())
        except Exception as e:
            logger.warning("Pipeline '%s' failed on original: %s", p.name, e)

    results = []

    for snr_db in snr_levels_db:
        # amplitude = noise_rms * 10^(SNR_dB/20)
        amplitude = noise_rms * 10 ** (snr_db / 20.0)
        logger.info("SNR = %+.0f dB → amplitude = %.2e (noise_rms = %.2e)",
                     snr_db, amplitude, noise_rms)

        for freq in signal_freqs:
            signal = make_oscillatory_signal(
                n_times, sfreq, freq=freq, amplitude=1.0)  # unit amplitude

            for trial in range(n_trials):
                spatial = rng.standard_normal(n_channels)
                spatial = spatial / np.max(np.abs(spatial))

                # Scale signal to target amplitude
                scaled_signal = signal * amplitude

                raw_injected, injected_data = inject_signal(
                    raw, scaled_signal, spatial_pattern=spatial)

                for p in pipelines:
                    if p.name not in originals:
                        continue
                    try:
                        raw_cleaned = p.func(raw_injected.copy())
                        metrics = measure_recovery(
                            raw_cleaned, originals[p.name],
                            injected_data, meg_picks)

                        rm = RecoveryMetrics(
                            pipeline=p.name,
                            signal_type=f"{freq}Hz",
                            snr_input=snr_db,
                            snr_output=metrics["snr_output_db"],
                            snr_improvement=metrics["snr_output_db"] - snr_db,
                            correlation=metrics["correlation"],
                            rmse=metrics["rmse"],
                            amplitude_ratio=metrics["amplitude_ratio"],
                            waveform_distortion=metrics["waveform_distortion"],
                        )
                        results.append(rm)
                    except Exception as e:
                        logger.warning("  %s SNR=%+.0fdB trial %d: %s",
                                       p.name, snr_db, trial, e)

    return results


# -------------------------------------------------------------------------
# Default pipeline configurations
# -------------------------------------------------------------------------

def make_default_pipelines(sfreq: float = 250.0) -> list[PipelineConfig]:
    """Create a set of candidate preprocessing pipelines to compare.

    These cover the parameter space relevant for WAND CTF MEG data.
    """

    def _pipeline_minimal(raw):
        raw = raw.copy()
        # pick only if non-MEG channels remain
        non_meg = [ch for ch, t in zip(raw.ch_names, raw.get_channel_types()) if t not in ("mag", "grad")]
        if non_meg:
            raw.pick(picks="meg", exclude="bads")
        raw.filter(1.0, 45.0, verbose=False)
        # Resampling already done before injection; skip here
        return raw

    def _pipeline_wide(raw):
        raw = raw.copy()
        # pick only if non-MEG channels remain
        non_meg = [ch for ch, t in zip(raw.ch_names, raw.get_channel_types()) if t not in ("mag", "grad")]
        if non_meg:
            raw.pick(picks="meg", exclude="bads")
        raw.filter(0.1, 100.0, verbose=False)
        # Resampling already done before injection; skip here
        return raw

    def _pipeline_narrow(raw):
        raw = raw.copy()
        # pick only if non-MEG channels remain
        non_meg = [ch for ch, t in zip(raw.ch_names, raw.get_channel_types()) if t not in ("mag", "grad")]
        if non_meg:
            raw.pick(picks="meg", exclude="bads")
        raw.filter(1.0, 30.0, verbose=False)
        # Resampling already done before injection; skip here
        return raw

    def _pipeline_notch50(raw):
        raw = raw.copy()
        # pick only if non-MEG channels remain
        non_meg = [ch for ch, t in zip(raw.ch_names, raw.get_channel_types()) if t not in ("mag", "grad")]
        if non_meg:
            raw.pick(picks="meg", exclude="bads")
        raw.notch_filter([50.0, 100.0], verbose=False)
        raw.filter(1.0, 45.0, verbose=False)
        # Resampling already done before injection; skip here
        return raw

    def _pipeline_ssp_ecg(raw):
        raw = raw.copy()
        # pick only if non-MEG channels remain
        non_meg = [ch for ch, t in zip(raw.ch_names, raw.get_channel_types()) if t not in ("mag", "grad")]
        if non_meg:
            raw.pick(picks="meg", exclude="bads")
        raw.filter(1.0, 45.0, verbose=False)
        try:
            projs, _ = mne.preprocessing.compute_proj_ecg(
                raw, n_mag=2, n_grad=0, average=True, verbose=False)
            raw.add_proj(projs)
            raw.apply_proj()
        except Exception:
            pass
        # Resampling already done before injection; skip here
        return raw

    def _pipeline_ssp_ecg_eog(raw):
        raw = raw.copy()
        # pick only if non-MEG channels remain
        non_meg = [ch for ch, t in zip(raw.ch_names, raw.get_channel_types()) if t not in ("mag", "grad")]
        if non_meg:
            raw.pick(picks="meg", exclude="bads")
        raw.filter(1.0, 45.0, verbose=False)
        try:
            projs_ecg, _ = mne.preprocessing.compute_proj_ecg(
                raw, n_mag=2, n_grad=0, average=True, verbose=False)
            raw.add_proj(projs_ecg)
        except Exception:
            pass
        try:
            projs_eog, _ = mne.preprocessing.compute_proj_eog(
                raw, n_mag=1, n_grad=0, average=True, verbose=False)
            raw.add_proj(projs_eog)
        except Exception:
            pass
        raw.apply_proj()
        # Resampling already done before injection; skip here
        return raw

    return [
        PipelineConfig("minimal_1-45Hz", _pipeline_minimal,
                       "Bandpass 1-45 Hz, resample"),
        PipelineConfig("wide_0.1-100Hz", _pipeline_wide,
                       "Wide bandpass 0.1-100 Hz, resample"),
        PipelineConfig("narrow_1-30Hz", _pipeline_narrow,
                       "Narrow bandpass 1-30 Hz, resample"),
        PipelineConfig("notch50_1-45Hz", _pipeline_notch50,
                       "Notch 50+100 Hz, bandpass 1-45 Hz"),
        PipelineConfig("ssp_ecg_1-45Hz", _pipeline_ssp_ecg,
                       "SSP ECG (2 proj), bandpass 1-45 Hz"),
        PipelineConfig("ssp_ecg_eog_1-45Hz", _pipeline_ssp_ecg_eog,
                       "SSP ECG+EOG, bandpass 1-45 Hz"),
    ]
