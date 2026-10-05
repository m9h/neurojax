"""Gap-aware feature windows for downstream physiological modelling."""

import numpy as np
from scipy.signal import welch

from neurojax.io.observations import Observation


def window_observations(
    streams, *, window_s=30.0, start_s=None, stop_s=None, min_coverage=0.8
):
    """Mean in nonoverlapping [start, stop) windows, without filling gaps.

    Coverage uses the median native sample interval; a missing interval cannot
    become valid simply because all remaining samples are finite. Window times
    are centres. Values below minimum coverage remain NaN and invalid.
    """
    if not streams or window_s <= 0 or not 0 < min_coverage <= 1:
        raise ValueError("streams, positive window_s and coverage in (0,1] required")
    if any(len(stream.time_s) < 2 for stream in streams.values()):
        raise ValueError("at least two samples per stream required for coverage")
    start = min(s.time_s[0] for s in streams.values()) if start_s is None else start_s
    stop = (
        max(s.time_s[-1] + np.median(np.diff(s.time_s)) for s in streams.values())
        if stop_s is None
        else stop_s
    )
    if not np.isfinite([start, stop, window_s]).all() or stop <= start:
        raise ValueError("finite increasing window bounds required")
    n = int(np.floor((stop - start) / window_s + 1e-9))
    centres = start + (np.arange(n) + 0.5) * window_s
    result = {}
    for name, stream in streams.items():
        dt = np.median(np.diff(stream.time_s))
        indices = np.floor((stream.time_s - start) / window_s).astype(int)
        use = stream.valid & (indices >= 0) & (indices < n)
        counts = np.bincount(indices[use], minlength=n)
        sums = np.bincount(indices[use], weights=stream.values[use], minlength=n)
        coverage = np.minimum(counts * dt / window_s, 1.0)
        valid = (coverage >= min_coverage) & (counts > 0)
        values = np.full(n, np.nan)
        np.divide(sums, counts, out=values, where=valid)
        result[name] = Observation(
            centres,
            values,
            valid,
            stream.unit,
            dict(
                stream.metadata,
                aggregation="mean",
                window_s=window_s,
                min_coverage=min_coverage,
                coverage=coverage.tolist(),
            ),
        )
    return result


def standardize_observations(streams, reference):
    """Centre/scale using a separate baseline recording; retain provenance.

    Invalid or constant baselines fail explicitly. No test/drug data are used
    to choose the normalization. Output is dimensionless, not autonomic tone.
    """
    result = {}
    for name, stream in streams.items():
        baseline = reference[name]
        if baseline.unit != stream.unit:
            raise ValueError(f"baseline unit mismatch for {name}")
        data = baseline.values[baseline.valid]
        if len(data) < 2 or np.std(data) <= 0:
            raise ValueError(f"insufficient varying baseline for {name}")
        mean, sd = float(np.mean(data)), float(np.std(data, ddof=1))
        result[name] = Observation(
            stream.time_s,
            (stream.values - mean) / sd,
            stream.valid,
            "1",
            dict(
                stream.metadata,
                baseline_mean=mean,
                baseline_sd=sd,
                original_unit=stream.unit,
                baseline_source=baseline.metadata,
            ),
        )
    return result


def eeg_bandpower(raw, *, band=(8.0, 13.0), window_s=30.0):
    """Sensor-mean band power from caller-cleaned MNE Raw, with annotation QC.

    No automatic artifact cleaning or source reconstruction. Rejects a whole
    window containing any annotation-masked/NaN samples; never joins
    disconnected samples for spectral estimation. Units are V².
    """
    fs = raw.info["sfreq"]
    low, high = band
    if not 0 < low < high < fs / 2 or window_s <= 0:
        raise ValueError("invalid band/window")
    data = raw.get_data(picks="eeg", reject_by_annotation="NaN")
    width = int(round(window_s * fs))
    if width < 2:
        raise ValueError("window must contain at least two samples")
    n = data.shape[1] // width
    values = np.full(n, np.nan)
    valid = np.zeros(n, bool)
    for i in range(n):
        segment = data[:, i * width : (i + 1) * width]
        # Even a short gap invalidates the PSD: interpolation is not implicit.
        if not np.isfinite(segment).all():
            continue
        frequencies, psd = welch(
            segment, fs=fs, nperseg=min(width, int(2 * fs)), axis=-1
        )
        select = (frequencies >= low) & (frequencies <= high)
        values[i] = np.mean(np.trapezoid(psd[:, select], frequencies[select], axis=-1))
        valid[i] = True
    return Observation(
        (np.arange(n) + 0.5) * width / fs,
        values,
        valid,
        "V^2",
        {
            "method": "Welch, mean across good EEG sensors",
            "band_hz": list(band),
            "window_s": width / fs,
            "time_origin": "recording_start",
            "requires_caller_artifact_cleaning": True,
            "gap_policy": "reject entire window",
        },
    )
