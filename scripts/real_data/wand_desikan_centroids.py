#!/usr/bin/env python
"""Desikan-68 region centroids on fsaverage, in the exact `parcels68` column order.

The MEG parcels (`parcels68.npy`) were extracted with MNE's Desikan `aparc` labels
(``wand_source_prep.py``); the geometric connectome-harmonic graph must use the same
labels in the same order.  Saves the 3-D centroids (and names) so the harmonic
analysis (`.venv-models`, no MNE) can build the region graph.

    .venv-oracle/bin/python scripts/real_data/wand_desikan_centroids.py
"""

import os

import numpy as np
import mne
from mne.datasets import fetch_fsaverage

OUT = os.environ.get("WAND_OUT", "/data/datasets/wand_src")


def main():
    fs_dir = fetch_fsaverage(verbose="ERROR")
    subjects_dir = os.path.dirname(fs_dir)
    labels = [l for l in mne.read_labels_from_annot(
        "fsaverage", "aparc", subjects_dir=subjects_dir, verbose="ERROR")
        if "unknown" not in l.name]
    cent = np.array([l.pos.mean(0) for l in labels])          # (68, 3) metres, RAS
    names = [l.name for l in labels]
    np.save(os.path.join(OUT, "desikan68_centroids.npy"), cent)
    with open(os.path.join(OUT, "desikan68_names.txt"), "w") as f:
        f.write("\n".join(names))
    print(f"{len(labels)} Desikan labels; centroids {cent.shape} -> {OUT}")
    print("first few:", names[:4])


if __name__ == "__main__":
    main()
