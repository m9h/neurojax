# DMT EEG and autonomic observations

Adapter for [D'Amelio et al.'s release](https://doi.org/10.5281/zenodo.19893951)
and [author analysis code](https://github.com/tomdamelio/dmt-emotions).
Download and extract `original.zip` and `derivatives.zip` locally, keeping
`participants.tsv` at the same root. No data are downloaded by the adapter.
These are inhaled **N,N-DMT**, not 5-MeO-DMT, recordings, without fMRI.
Attribute the CC-BY-4.0 dataset and associated papers when using the data.

## Export author-derived physiological features

From the neurojax checkout:

```bash
PYTHONPATH=src python examples/dmt_observations.py /path/to/release observations.npz \
  --subject S04 --session 1 --window-s 30
```

The default features are heart rate, SMNA and RVT. The loader reads the
participant's actual dose assignment and modality-specific inclusion flags;
session 1 does not imply low dose. Excluded streams retain values but all
samples are invalid by default. The export example refuses excluded streams.
Missing files/columns are omitted, so request only features actually present.
Both the documented release filename and upstream writer filename are
supported, with ambiguity treated as an error. The actual archive also stores
some resting-state files in `dmt_high`/`dmt_low`; the adapter checks the exact
resting-state filename there as well as in the documented `rs_*` directory.
SMNA can reside in the separate
`_cvx_decomposition.csv` file. Peak indices and processing metadata are
preserved from `_info.json`; no new peak detection or cvxEDA fit is performed.
Nonstandard JSON NaN/Infinity entries in release sidecars become `null`, with
the conversion policy recorded in metadata.

Windows are nonoverlapping, with timestamps at their centres. Coverage uses
the native sampling interval; NaN padding and missing windows stay invalid.
Each feature is centred and scaled against valid windows of the separate,
paired resting-state recording. The baseline is not concatenated with DMT:
there is an intervening reporting/administration interval. Times are relative
to recording start. The adapter does not establish an inhalation marker;
pass an explicit `offset_s` to the loader if a verified common clock is known.

HR is bpm; respiratory rate is breaths/min; EDA components are uS; SMNA is an
arbitrary-unit deconvolution estimate; RVT is belt-amplitude units/second,
not calibrated ventilation. No ECG quality cutoff is silently applied;
`min_ecg_quality` is optional and explicit. Finite samples are not necessarily
artifact-free: inspect the authors' QC and your own signal reports.

## EEG extension

`load_dmt_raw(root, subject, session, state)` returns an MNE BrainVision Raw.
The dedicated ECG, Ekg1/Ekg2, eye references, GSR and respiration are kept
distinct from EEG. The disconnected auxiliary `Eog` is marked bad.
Raw channels, annotations and physical units are retained. This loader does
not automatically perform EEG artifact cleaning.

After validated EEG cleaning, call
`neurojax.analysis.autonomic.eeg_bandpower(raw, band=(8, 13), window_s=30)`
to produce sensor-mean alpha power in V². Annotation-masked windows are
rejected entirely. Apply identical processing to the paired baseline before
normalization. Add this observation to the export if testing an explicit
EEG–autonomic shared-drive hypothesis; alpha suppression and peripheral
responses need not have the same signed loading.

## Shared file contract

`neurojax.io.observations` writes a pickle-free NPZ with scalar JSON `manifest`:
`schema = neurophys-observations/1`, and a list of stream records containing
`name`, `key`, `unit`, `metadata`. For each key (e.g. `s0`), arrays are
`s0_time_s`, `s0_values`, `s0_valid`. All are matching 1D arrays; timestamps
are finite and increasing. Masks are intersected with finite values.
Units, provenance, clock offsets, baseline scaling and sidecars travel with
each stream. vpjax reads this format without importing neurojax.

See vpjax's `docs/tutorials/dmt_shared_drive.md` for reduced joint inference.
