"""Release-shaped fixtures: dose order, exclusions, gaps and portable exchange."""

import json

import numpy as np
import pytest

from neurojax.analysis.autonomic import window_observations
from neurojax.io.dmt_autonomic import load_dmt_derivatives
from neurojax.io.observations import Observation, load_observations, save_observations


def test_derivatives_preserve_counterbalanced_dose_and_exclusions(tmp_path):
    (tmp_path / "participants.tsv").write_text(
        "participant_id\tsession1_dose_mg\tsession2_dose_mg\tincluded_ecg\tincluded_eda\tincluded_resp\n"
        "sub-04\t40\t20\t1\t0\t1\n"
    )
    for modality, column in [
        ("ecg", "ECG_Rate"),
        ("eda", "EDA_Phasic"),
        ("resp", "RSP_RVT"),
    ]:
        folder = tmp_path / "derivatives/preprocessing/phys" / modality / "dmt_high"
        folder.mkdir(parents=True)
        path = folder / "S04_dmt_session1_high.csv"
        path.write_text(f"time,{column}\n0,1\n0.004,nan\n0.008,3\n")
        path.with_name(path.stem + "_info.json").write_text(
            json.dumps({"method": "author", "missing": float("nan")})
        )
    result = load_dmt_derivatives(tmp_path, "sub-04", 1, "dmt")
    assert result["heart_rate"].unit == "bpm"
    assert result["heart_rate"].metadata["dose_mg"] == 40
    assert result["heart_rate"].metadata["sidecar"]["method"] == "author"
    assert result["heart_rate"].metadata["sidecar"]["missing"] is None
    save_observations(tmp_path / "export.npz", result)
    np.testing.assert_array_equal(result["heart_rate"].valid, [True, False, True])
    assert not result["eda_phasic"].valid.any()
    assert result["rvt"].unit == "a.u./s"


def test_windowing_does_not_bridge_missing_data_or_clock_gaps():
    stream = Observation(
        np.array([0.0, 1.0, 2.0, 10.0]),
        np.array([1.0, 3.0, np.nan, 8.0]),
        np.array([True, True, False, True]),
        "bpm",
        {},
    )
    result = window_observations(
        {"hr": stream}, window_s=2.0, stop_s=12.0, min_coverage=0.75
    )["hr"]
    np.testing.assert_allclose(result.time_s, [1, 3, 5, 7, 9, 11])
    assert result.values[0] == 2
    np.testing.assert_array_equal(
        result.valid, [True, False, False, False, False, False]
    )


def test_portable_roundtrip_and_invalid_time(tmp_path):
    stream = Observation(
        np.array([0.0, 2.0]),
        np.array([1.0, np.nan]),
        np.array([True, False]),
        "uS",
        {"source": "fixture"},
    )
    path = tmp_path / "observations.npz"
    save_observations(path, {"eda": stream})
    actual = load_observations(path)["eda"]
    np.testing.assert_array_equal(actual.valid, stream.valid)
    assert actual.metadata == stream.metadata
    with pytest.raises(ValueError, match="increasing"):
        Observation(np.array([1.0, 0.0]), np.ones(2), np.ones(2, dtype=bool), "uS", {})


def test_upstream_filename_cvx_and_explicit_quality(tmp_path):
    (tmp_path / "participants.tsv").write_text(
        "participant_id\tsession1_dose_mg\tsession2_dose_mg\t"
        "included_ecg\tincluded_eda\tincluded_resp\n"
        "sub-04\t20\t40\t1\t1\t1\n"
    )
    folder = tmp_path / "derivatives/preprocessing/phys/eda/dmt_high"
    folder.mkdir(parents=True)
    (folder / "S04_rs_Reposo_2_high.csv").write_text("time,EDA_Phasic\n0,1\n0.004,2\n")
    (folder / "S04_rs_Reposo_2_high_cvx_decomposition.csv").write_text(
        "time,SMNA\n0,3\n0.004,4\n"
    )
    ecg = tmp_path / "derivatives/preprocessing/phys/ecg/rs_high"
    ecg.mkdir(parents=True)
    (ecg / "S04_rs_Reposo_2_high.csv").write_text(
        "time,ECG_Rate,ECG_Quality\n0,70,0.9\n0.004,71,0.2\n"
    )
    streams = load_dmt_derivatives(tmp_path, "S04", 2, "rs", min_ecg_quality=0.5)
    np.testing.assert_array_equal(streams["smna"].values, [3, 4])
    np.testing.assert_array_equal(streams["heart_rate"].valid, [True, False])
    (folder / "S04_rs_session2_high.csv").write_text("time,EDA_Phasic\n0,1\n")
    with pytest.raises(ValueError, match="ambiguous"):
        load_dmt_derivatives(tmp_path, "S04", 2, "rs")


def test_baseline_standardization_and_eeg_annotation_mask():
    import mne

    from neurojax.analysis.autonomic import eeg_bandpower, standardize_observations

    baseline = Observation(
        np.arange(3.0), np.array([10.0, 20.0, 30.0]), np.ones(3, bool), "bpm", {}
    )
    drug = Observation(
        np.arange(3.0), np.array([20.0, 30.0, 40.0]), np.ones(3, bool), "bpm", {}
    )
    result = standardize_observations({"hr": drug}, {"hr": baseline})["hr"]
    np.testing.assert_allclose(result.values, [0, 1, 2])
    assert result.unit == "1"
    fs = 100.0
    time = np.arange(400) / fs
    raw = mne.io.RawArray(
        np.sin(2 * np.pi * 10 * time)[None] * 1e-6,
        mne.create_info(["Cz"], fs, "eeg"),
        verbose=False,
    )
    raw.set_annotations(mne.Annotations([2.5], [0.2], ["BAD_movement"]))
    power = eeg_bandpower(raw, window_s=2.0)
    np.testing.assert_array_equal(power.valid, [True, False])
    np.testing.assert_allclose(power.values[0], 0.5e-12, rtol=0.01)


def test_brainvision_reader_preserves_channels_and_annotations(tmp_path):
    from neurojax.io.dmt_autonomic import load_dmt_raw

    (tmp_path / "participants.tsv").write_text(
        "participant_id\tsession1_dose_mg\tsession2_dose_mg\t"
        "included_ecg\tincluded_eda\tincluded_resp\n"
        "sub-04\t40\t20\t1\t1\t1\n"
    )
    folder = tmp_path / "original/physiology/DMT_1/S04"
    folder.mkdir(parents=True)
    channels = ["Cz", "ECG", "RESP", "GSR", "Eog", "R_EYE", "Ekg1"]
    samples = np.arange(70, dtype="<f4").reshape(10, 7)
    (folder / "record.eeg").write_bytes(samples.tobytes())
    (folder / "record.vmrk").write_text(
        "Brain Vision Data Exchange Marker File, Version 1.0\n"
        "[Common Infos]\nDataFile=record.eeg\n[Marker Infos]\n"
        "Mk1=Stimulus,S 1,3,1,0\n"
    )
    (folder / "record.vhdr").write_text(
        "Brain Vision Data Exchange Header File Version 1.0\n"
        "[Common Infos]\nDataFile=record.eeg\nMarkerFile=record.vmrk\n"
        "DataFormat=BINARY\nDataOrientation=MULTIPLEXED\n"
        "NumberOfChannels=7\nSamplingInterval=4000\n"
        "[Binary Infos]\nBinaryFormat=IEEE_FLOAT_32\n[Channel Infos]\n"
        + "".join(f"Ch{i + 1}={name},,1,µV\n" for i, name in enumerate(channels))
    )
    raw = load_dmt_raw(tmp_path, "S04", 1, "dmt", preload=True)
    assert raw.get_channel_types() == [
        "eeg",
        "ecg",
        "misc",
        "misc",
        "misc",
        "eog",
        "ecg",
    ]
    assert raw.info["bads"] == ["Eog"]
    np.testing.assert_allclose(raw.get_data()[0], samples[:, 0] * 1e-6)
    assert "Stimulus/S 1" in raw.annotations.description
