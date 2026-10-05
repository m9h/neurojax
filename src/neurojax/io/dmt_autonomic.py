"""Adapters for D'Amelio et al., Zenodo 10.5281/zenodo.19893951.

Reads an extracted release locally. No automatic multi-GB download. Dose
order and modality exclusions come from participants.tsv, never session ID.
"""

import csv
import json
import re
from pathlib import Path

import numpy as np

from neurojax.io.observations import Observation

DOI = "10.5281/zenodo.19893951"
EEG_CHANNELS = (
    "Fp1",
    "Fp2",
    "F3",
    "F4",
    "C3",
    "C4",
    "P3",
    "P4",
    "O1",
    "O2",
    "F7",
    "F8",
    "T7",
    "T8",
    "P7",
    "P8",
    "Fz",
    "Cz",
    "Pz",
    "FC2",
    "CP1",
    "CP2",
    "FC5",
    "FC6",
    "CP5",
    "CP6",
    "TP9",
    "TP10",
)
FEATURES = {
    "ecg": {"ECG_Rate": ("heart_rate", "bpm")},
    "eda": {
        "EDA_Phasic": ("eda_phasic", "uS"),
        "EDA_Tonic": ("eda_tonic", "uS"),
        "SMNA": ("smna", "a.u."),
    },
    "resp": {
        "RSP_Rate": ("respiration_rate", "breaths/min"),
        "RSP_RVT": ("rvt", "a.u./s"),
    },
}


def _participant(root, subject, session, state):
    match = re.fullmatch(r"(?:sub-|S)?(\d{1,2})", subject)
    if match is None or session not in (1, 2) or state not in ("dmt", "rs"):
        raise ValueError("use subject S04/sub-04, session 1/2, state dmt/rs")
    number = int(match.group(1))
    with (root / "participants.tsv").open() as file:
        rows = list(csv.DictReader(file, delimiter="\t"))
    matches = [row for row in rows if row["participant_id"] == f"sub-{number:02d}"]
    if len(matches) != 1:
        raise ValueError(f"participant sub-{number:02d} not uniquely present")
    dose = int(matches[0][f"session{session}_dose_mg"])
    if dose not in (20, 40):
        raise ValueError(f"unsupported dose {dose}")
    return f"S{number:02d}", matches[0], dose


def _included(value):
    if value.lower() in ("1", "true", "yes"):
        return True
    if value.lower() in ("0", "false", "no"):
        return False
    raise ValueError(f"unknown inclusion flag {value!r}")


def _derivative_files(root, subject, session, state, dose, modality):
    parent = root / "derivatives/preprocessing/phys" / modality
    folders = [parent / f"{state}_{dose}"]
    # Actual ZIP stores some resting-state CSVs under dmt_{dose}, although
    # the release README documents rs_{dose}. Require the exact rs filename.
    if state == "rs":
        folders.append(parent / f"dmt_{dose}")
    experiment = f"DMT_{session}" if state == "dmt" else f"Reposo_{session}"
    # Release README and upstream writer use different filename conventions.
    bases = [
        folder / name
        for folder in folders
        for name in (
            f"{subject}_{state}_session{session}_{dose}.csv",
            f"{subject}_{state}_{experiment}_{dose}.csv",
        )
    ]
    found = [path for path in bases if path.exists()]
    if len(found) > 1:
        raise ValueError(f"ambiguous derivative files: {found}")
    if not found:
        return []
    base = found[0]
    cvx = base.with_name(base.stem + "_cvx_decomposition.csv")
    return [base] + ([cvx] if modality == "eda" and cvx.exists() else [])


def load_dmt_derivatives(
    root,
    subject,
    session,
    state,
    *,
    offset_s=0.0,
    respect_inclusion=True,
    min_ecg_quality=None,
):
    """Read author-derived HR, EDA/SMNA and respiration with independent grids.

    Nonfinite samples and participant-level exclusions are masked. The optional
    ECG_Quality threshold must be chosen explicitly; quality scales are not
    assumed to represent calibrated probabilities. Missing modalities are
    omitted. No interpolation or recording concatenation is performed.
    ``offset_s`` maps the recording clock to a caller-defined common clock;
    zero is recording start, not an independently verified inhalation marker.
    RVT is a belt-amplitude proxy, not calibrated ventilation. SMNA units are
    arbitrary deconvolution units, not a direct nerve recording.
    """
    root = Path(root)
    subject, participant, dose_mg = _participant(root, subject, session, state)
    dose = "high" if dose_mg == 40 else "low"
    if not np.isfinite(offset_s):
        raise ValueError("offset_s must be finite")
    streams = {}
    for modality, columns in FEATURES.items():
        included = _included(participant[f"included_{modality}"])
        for path in _derivative_files(root, subject, session, state, dose, modality):
            data = np.genfromtxt(path, delimiter=",", names=True, dtype=float, ndmin=1)
            if "time" not in data.dtype.names:
                raise ValueError(f"missing time column: {path}")
            sidecar_path = path.with_name(path.stem + "_info.json")
            sidecar = (
                json.loads(sidecar_path.read_text(), parse_constant=lambda _: None)
                if sidecar_path.exists()
                else {}
            )
            for column, (name, unit) in columns.items():
                if column not in data.dtype.names:
                    continue
                if name in streams:
                    raise ValueError(f"duplicate feature {name}")
                valid = np.isfinite(data[column])
                if respect_inclusion and not included:
                    valid[:] = False
                if modality == "ecg" and min_ecg_quality is not None:
                    if "ECG_Quality" not in data.dtype.names:
                        raise ValueError("ECG_Quality required for quality threshold")
                    valid &= np.isfinite(data["ECG_Quality"]) & (
                        data["ECG_Quality"] >= min_ecg_quality
                    )
                streams[name] = Observation(
                    data["time"] + offset_s,
                    data[column],
                    valid,
                    unit,
                    {
                        "dataset_doi": DOI,
                        "source": str(path.resolve()),
                        "column": column,
                        "subject": subject,
                        "session": session,
                        "state": state,
                        "dose_mg": dose_mg,
                        "included": included,
                        "respect_inclusion": respect_inclusion,
                        "time_origin": "recording_start",
                        "offset_s": offset_s,
                        "min_ecg_quality": min_ecg_quality,
                        "sidecar": sidecar,
                        "sidecar_nonfinite_policy": "JSON NaN/Infinity mapped to null",
                    },
                )
    if not streams:
        raise FileNotFoundError("no supported derivative features found")
    return streams


def load_dmt_raw(root, subject, session, state, *, preload=False):
    """Load BrainVision EEG/physiology with correct modality channel types.

    Returns an MNE Raw with annotations and SI units. Does not clean artifacts
    or infer administration onset. Dedicated ECG is distinct from Ekg1/Ekg2;
    the disconnected auxiliary Eog channel is marked bad.
    """
    import mne

    root = Path(root)
    subject, participant, dose = _participant(root, subject, session, state)
    experiment = f"DMT_{session}" if state == "dmt" else f"Reposo_{session}"
    folder = root / "original/physiology" / experiment / subject
    paths = list(folder.glob("*.vhdr"))
    if len(paths) != 1:
        raise ValueError(
            f"expected one BrainVision header in {folder}, got {len(paths)}"
        )
    raw = mne.io.read_raw_brainvision(paths[0], preload=preload, verbose=False)
    types = {name: "misc" for name in raw.ch_names}
    types.update({name: "eeg" for name in EEG_CHANNELS if name in types})
    types.update({name: "ecg" for name in ("ECG", "Ekg1", "Ekg2") if name in types})
    types.update({name: "eog" for name in ("R_EYE", "L_EYE") if name in types})
    raw.set_channel_types(types, verbose=False)
    if "Eog" in raw.ch_names:
        raw.info["bads"] = list(set(raw.info["bads"] + ["Eog"]))
    return raw
