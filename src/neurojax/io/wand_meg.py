"""WAND (Welsh Advanced Neuroimaging Database) MEG loader.

Loads CTF .ds MEG recordings from the WAND dataset (CUBRIC, Cardiff).
WAND is a multi-modal BIDS dataset hosted on GIN with git-annex.

The MEG data lives in ``ses-01/meg/`` as CTF .ds directories containing:
  - ``.meg4`` — raw time series (git-annex, ~1GB per task)
  - ``.res4`` — sensor layout (git-annex, needed by MNE)
  - ``.acq``  — acquisition parameters (available without annex)
  - ``.hc``   — head coil positions (available without annex)

Typical usage::

    loader = WANDMEGLoader("/path/to/wand")
    raw = loader.load_resting("sub-08033")
    parcellated = loader.source_reconstruct(raw, n_parcels=38)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional

import mne
import numpy as np

logger = logging.getLogger(__name__)

# WAND CTF system: 275 MEG channels
WAND_N_MEG_CHANNELS = 275
WAND_GIN_URL = "https://gin.g-node.org/CUBRIC/WAND"

# Available tasks in ses-01/meg/
WAND_TASKS = ["resting", "auditorymotor", "mmn", "simon", "visual"]


class WANDMEGLoader:
    """Loader for WAND CTF MEG data.

    Parameters
    ----------
    bids_root : str or path
        Root of the WAND BIDS directory (contains ``sub-XXXXX/``).
    """

    def __init__(self, bids_root: str | os.PathLike) -> None:
        self.bids_root = Path(bids_root)
        if not self.bids_root.is_dir():
            raise FileNotFoundError(f"WAND root not found: {self.bids_root}")

    # -- discovery -----------------------------------------------------------

    def list_subjects(self) -> List[str]:
        """Return sorted list of subject IDs with MEG data."""
        subjects = []
        for p in sorted(self.bids_root.iterdir()):
            if p.is_dir() and p.name.startswith("sub-"):
                meg_dir = p / "ses-01" / "meg"
                if meg_dir.is_dir():
                    subjects.append(p.name)
        return subjects

    def list_tasks(self, subject: str) -> List[str]:
        """Return available MEG tasks for a subject."""
        meg_dir = self.bids_root / subject / "ses-01" / "meg"
        tasks = []
        for ds in sorted(meg_dir.iterdir()):
            if ds.suffix == ".ds" and ds.is_dir():
                # Extract task name from e.g. sub-08033_ses-01_task-resting.ds
                parts = ds.stem.split("task-")
                if len(parts) == 2:
                    tasks.append(parts[1])
        return tasks

    def _ds_path(self, subject: str, task: str) -> Path:
        """Return path to the .ds directory for a subject/task."""
        meg_dir = self.bids_root / subject / "ses-01" / "meg"
        ds_name = f"{subject}_ses-01_task-{task}.ds"
        return meg_dir / ds_name

    def is_available(self, subject: str, task: str = "resting") -> bool:
        """Check if actual MEG data (not just annex pointer) is available."""
        ds = self._ds_path(subject, task)
        meg4 = ds / f"{ds.stem}.meg4"
        res4 = ds / f"{ds.stem}.res4"
        # Annex pointers are < 200 bytes; real meg4 files are > 1MB
        return (
            meg4.exists()
            and meg4.stat().st_size > 1000
            and res4.exists()
            and res4.stat().st_size > 1000
        )

    # -- loading -------------------------------------------------------------

    def load_raw(
        self,
        subject: str,
        task: str = "resting",
        preload: bool = True,
    ) -> mne.io.Raw:
        """Load a CTF .ds recording as MNE Raw.

        Parameters
        ----------
        subject : str
            Subject ID (e.g. ``"sub-08033"``).
        task : str
            Task name (default ``"resting"``).
        preload : bool
            Whether to load data into memory.

        Returns
        -------
        mne.io.Raw

        Raises
        ------
        FileNotFoundError
            If the .ds directory or annex data is missing.
        """
        ds_path = self._ds_path(subject, task)
        if not ds_path.is_dir():
            raise FileNotFoundError(f"CTF .ds not found: {ds_path}")

        if not self.is_available(subject, task):
            meg4 = ds_path / f"{ds_path.stem}.meg4"
            raise FileNotFoundError(
                f"MEG data is a git-annex pointer ({meg4.stat().st_size} bytes). "
                f"Run: cd {self.bids_root} && datalad get {meg4.relative_to(self.bids_root)}"
            )

        logger.info("Loading %s task-%s from %s", subject, task, ds_path)
        raw = mne.io.read_raw_ctf(str(ds_path), preload=preload, verbose=False)
        return raw

    def load_resting(self, subject: str, **kwargs) -> mne.io.Raw:
        """Convenience: load resting-state MEG."""
        return self.load_raw(subject, task="resting", **kwargs)

    def load_headshape(self, subject: str) -> np.ndarray:
        """Load digitized headshape points.

        Returns
        -------
        points : (N, 3) array — headshape coordinates in cm
        """
        meg_dir = self.bids_root / subject / "ses-01" / "meg"
        pos_file = meg_dir / f"{subject}_ses-01_headshape.pos"
        if not pos_file.exists():
            raise FileNotFoundError(f"Headshape not found: {pos_file}")

        points = []
        with open(pos_file) as f:
            n_points = int(f.readline().strip())
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4 and parts[0] in ("EXTRA", "Nasion", "LPA", "RPA"):
                    points.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif len(parts) >= 4 and parts[0].startswith("HPI"):
                    continue  # Skip HPI coils

        return np.array(points)

    # -- preprocessing -------------------------------------------------------

    def preprocess(
        self,
        raw: mne.io.Raw,
        l_freq: float = 1.0,
        h_freq: float = 45.0,
        resample: float = 250.0,
        reject_mag: float = 4000e-15,
    ) -> mne.io.Raw:
        """Standard preprocessing: filter, resample, pick MEG.

        Parameters
        ----------
        raw : mne.io.Raw
        l_freq, h_freq : float
            Bandpass filter edges (Hz).
        resample : float
            Target sampling rate (Hz).
        reject_mag : float
            Reject threshold for magnetometers (T).

        Returns
        -------
        mne.io.Raw — preprocessed, MEG channels only
        """
        raw = raw.copy()
        raw.pick_types(meg=True, ref_meg=False, eog=False, stim=False)
        raw.filter(l_freq, h_freq, verbose=False)
        raw.resample(resample, verbose=False)
        logger.info(
            "Preprocessed: %d channels, %.1f Hz, %.1f s",
            len(raw.ch_names), raw.info["sfreq"], raw.times[-1],
        )
        return raw

    # -- source reconstruction -----------------------------------------------

    def source_reconstruct(
        self,
        raw: mne.io.Raw,
        parcellation: str = "aparc",
        n_parcels: Optional[int] = None,
        method: str = "sLORETA",
        snr: float = 3.0,
        max_duration: float = 120.0,
        spacing: str = "oct5",
    ) -> np.ndarray:
        """Source reconstruct and parcellate MEG data.

        Uses a template MRI (fsaverage) with a single-sphere forward model.
        Parcellates in chunks to keep memory under control.

        Parameters
        ----------
        raw : mne.io.Raw
            Preprocessed MEG data.
        parcellation : str
            FreeSurfer parcellation atlas (``"aparc"`` for Desikan-Killiany
            68 parcels, ``"aparc.a2009s"`` for Destrieux 148).
        n_parcels : int, optional
            If set, PCA-reduce parcellated data to this many components.
        method : str
            Inverse method (``"sLORETA"``, ``"dSPM"``, ``"MNE"``).
        snr : float
            Assumed SNR for regularisation (lambda2 = 1/snr^2).
        max_duration : float
            Maximum duration (seconds) to source-reconstruct. Longer
            recordings are cropped to save memory. Set to ``None``
            to use the full recording.
        spacing : str
            Source space resolution (``"oct5"`` ~1k sources/hemi,
            ``"oct6"`` ~4k — use oct5 for memory efficiency).

        Returns
        -------
        parcellated : (n_samples, n_parcels) array
            Sign-flipped parcellated source time courses.
        """
        # Crop to max_duration to avoid OOM
        if max_duration is not None and raw.times[-1] > max_duration:
            logger.info(
                "Cropping from %.0fs to %.0fs for source reconstruction",
                raw.times[-1], max_duration,
            )
            raw = raw.copy().crop(tmax=max_duration)

        subjects_dir = mne.datasets.fetch_fsaverage(verbose=False)
        if isinstance(subjects_dir, str):
            subjects_dir = Path(subjects_dir).parent
        else:
            subjects_dir = Path(subjects_dir).parent

        fs_subject = "fsaverage"

        # Source space (oct5 = ~1k sources/hemi, oct6 = ~4k)
        src = mne.setup_source_space(
            fs_subject, spacing=spacing,
            subjects_dir=str(subjects_dir), verbose=False,
        )

        # Forward model — use a fixed-origin sphere to avoid needing
        # digitization points (CTF .ds often lacks EXTRA dig points).
        meg_picks = mne.pick_types(raw.info, meg=True)
        meg_locs = np.array([raw.info["chs"][p]["loc"][:3] for p in meg_picks])
        r0 = meg_locs.mean(axis=0)
        r0[2] -= 0.04  # shift toward head centre

        sphere = mne.make_sphere_model(
            r0=r0, head_radius=0.09, verbose=False,
        )
        fwd = mne.make_forward_solution(
            raw.info, trans="fsaverage",
            src=src, bem=sphere,
            eeg=False, verbose=False,
        )

        # Noise covariance (first 30s as proxy)
        noise_cov = mne.compute_raw_covariance(
            raw, tmin=0, tmax=min(30.0, raw.times[-1]),
            method="shrunk", verbose=False,
        )

        # Inverse operator
        inv = mne.minimum_norm.make_inverse_operator(
            raw.info, fwd, noise_cov, verbose=False,
        )

        # Labels for parcellation
        labels = mne.read_labels_from_annot(
            fs_subject, parc=parcellation,
            subjects_dir=str(subjects_dir), verbose=False,
        )
        labels = [l for l in labels if "unknown" not in l.name.lower()]

        # Apply inverse in chunks to keep memory bounded
        lambda2 = 1.0 / snr ** 2
        sfreq = raw.info["sfreq"]
        chunk_sec = 30.0  # process 30s at a time
        chunk_samples = int(chunk_sec * sfreq)
        total_samples = len(raw.times)
        all_label_ts = []

        for start in range(0, total_samples, chunk_samples):
            tmin = start / sfreq
            tmax = min((start + chunk_samples) / sfreq, raw.times[-1])
            raw_chunk = raw.copy().crop(tmin=tmin, tmax=tmax)

            stc = mne.minimum_norm.apply_inverse_raw(
                raw_chunk, inv, lambda2=lambda2, method=method, verbose=False,
            )
            label_ts = mne.extract_label_time_course(
                stc, labels, src, mode="mean_flip", verbose=False,
            )
            all_label_ts.append(label_ts)
            del stc  # free memory immediately
            logger.info("  chunk %.0f-%.0fs done", tmin, tmax)

        parcellated = np.concatenate(all_label_ts, axis=1).T  # (n_samples, n_parcels)
        logger.info(
            "Parcellated: %s (%d parcels from %s)",
            parcellated.shape, len(labels), parcellation,
        )

        if n_parcels is not None and n_parcels < parcellated.shape[1]:
            from neurojax.data.loading import prepare_pca
            import jax.numpy as jnp
            parcellated = np.array(
                prepare_pca(jnp.array(parcellated), n_pca_components=n_parcels)
            )

        return parcellated

    # -- end-to-end pipeline -------------------------------------------------

    def run_pipeline(
        self,
        subject: str,
        task: str = "resting",
        parcellation: str = "aparc",
        save_dir: Optional[str | os.PathLike] = None,
    ) -> np.ndarray:
        """Full pipeline: load → preprocess → source reconstruct → parcellate.

        Parameters
        ----------
        subject : str
        task : str
        parcellation : str
        save_dir : path, optional
            If set, saves parcellated data as ``{subject}_task-{task}_parc.npy``.

        Returns
        -------
        parcellated : (n_samples, n_parcels) array
        """
        raw = self.load_raw(subject, task=task)
        raw = self.preprocess(raw)
        parcellated = self.source_reconstruct(raw, parcellation=parcellation)

        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            fname = f"{subject}_task-{task}_parc.npy"
            np.save(save_dir / fname, parcellated)
            logger.info("Saved %s", save_dir / fname)

        return parcellated

    def __repr__(self) -> str:
        subs = self.list_subjects()
        return f"WANDMEGLoader(root={self.bids_root}, n_subjects={len(subs)})"
