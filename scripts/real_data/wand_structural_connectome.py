#!/usr/bin/env python
"""Build the sub-08033 Desikan-68 structural connectome (probtrackx2_gpu) for the
proper Atasoy connectome harmonics.

Uses TRACULA's BBR-registered parcellation `aparc+aseg.bbr.nii.gz` (already in
diffusion space) + `dmri.bedpostX` — so no manual DWI↔anat registration.  Extracts
the 68 Desikan cortical ROIs **in the exact `parcels68` / MEG column order** (so the
resulting 68×68 connectivity needs no reordering) and runs probtrackx2_gpu in
`--network` mode → `fdt_network_matrix` → `desikan68_SC.npy`.

    .venv-models/bin/python scripts/real_data/wand_structural_connectome.py
"""

import os
import subprocess

import numpy as np

FSLDIR = os.environ.get("FSLDIR", "/home/mhough/fsl")
FSLBIN = os.path.join(FSLDIR, "share/fsl/bin")
OUT = os.environ.get("WAND_OUT", "/data/datasets/wand_src")
TRAC = "/data/raw/wand/derivatives/freesurfer/tracula/sub-08033_ses-02"
APARC = os.path.join(TRAC, "dlabel/diff/aparc+aseg.bbr.nii.gz")
BPX = os.path.join(TRAC, "dmri.bedpostX")
WORK = os.path.join(OUT, "sc_08033")

# Standard FreeSurfer Desikan aparc order (index = label - 1000/2000); corpuscallosum
# (idx 4) is excluded by the MEG parcellation.
DESIKAN = [
    "bankssts", "caudalanteriorcingulate", "caudalmiddlefrontal", "corpuscallosum",
    "cuneus", "entorhinal", "fusiform", "inferiorparietal", "inferiortemporal",
    "isthmuscingulate", "lateraloccipital", "lateralorbitofrontal", "lingual",
    "medialorbitofrontal", "middletemporal", "parahippocampal", "paracentral",
    "parsopercularis", "parsorbitalis", "parstriangularis", "pericalcarine",
    "postcentral", "posteriorcingulate", "precentral", "precuneus",
    "rostralanteriorcingulate", "rostralmiddlefrontal", "superiorfrontal",
    "superiorparietal", "superiortemporal", "supramarginal", "frontalpole",
    "temporalpole", "transversetemporal", "insula",
]
APARC_IDX = {name: i + 1 for i, name in enumerate(DESIKAN)}


def fsl(*args):
    subprocess.run([os.path.join(FSLBIN, args[0]), *args[1:]], check=True,
                   env=dict(os.environ, FSLDIR=FSLDIR,
                            FSLOUTPUTTYPE="NIFTI_GZ"))


def main():
    os.makedirs(WORK, exist_ok=True)
    names = [n for n in open(os.path.join(OUT, "desikan68_names.txt")).read().split("\n") if n]
    assert len(names) == 68, f"expected 68 MEG parcels, got {len(names)}"

    # extract the 68 cortical ROI masks IN MEG ORDER
    seeds, labels = [], []
    for nm in names:
        base, hemi = nm.rsplit("-", 1)
        label = (1000 if hemi == "lh" else 2000) + APARC_IDX[base]
        mask = os.path.join(WORK, f"roi_{label}.nii.gz")
        fsl("fslmaths", APARC, "-thr", str(label), "-uthr", str(label), "-bin", mask)
        seeds.append(mask)
        labels.append(label)
    # report ROI voxel counts (catch empty/misregistered ROIs before the long run)
    counts = [int(float(subprocess.run([os.path.join(FSLBIN, "fslstats"), s, "-V"],
              capture_output=True, text=True,
              env=dict(os.environ, FSLDIR=FSLDIR)).stdout.split()[0])) for s in seeds]
    print(f"68 ROIs extracted (MEG order); voxels: min={min(counts)} "
          f"max={max(counts)} median={int(np.median(counts))}; empty={counts.count(0)}")
    if counts.count(0):
        print("WARNING: empty ROIs present — check registration before trusting the SC")

    seedfile = os.path.join(WORK, "seeds.txt")
    with open(seedfile, "w") as f:
        f.write("\n".join(seeds) + "\n")
    np.save(os.path.join(OUT, "desikan68_SC_labels.npy"), np.array(labels))

    mask = os.path.join(BPX, "nodif_brain_mask.nii.gz")
    pdir = os.path.join(WORK, "probtrackx")
    print("running probtrackx2_gpu --network (this is the long step)...", flush=True)
    subprocess.run([
        os.path.join(FSLBIN, "probtrackx2_gpu"),
        "-x", seedfile, "--network",
        "-s", os.path.join(BPX, "merged"), "-m", mask,
        "--dir=" + pdir, "--forcedir",
        "-l", "--onewaycondition", "--pd",
        "-c", "0.2", "-S", "2000", "--steplength=0.5",
        "-P", "5000", "--fibthresh=0.01", "--distthresh=0.0", "--sampvox=0.0",
    ], check=True, env=dict(os.environ, FSLDIR=FSLDIR, FSLOUTPUTTYPE="NIFTI_GZ"))

    SC = np.loadtxt(os.path.join(pdir, "fdt_network_matrix"))
    SC = 0.5 * (SC + SC.T)                                  # symmetrise streamline counts
    np.save(os.path.join(OUT, "desikan68_SC.npy"), SC)
    print(f"structural connectome {SC.shape} -> {OUT}/desikan68_SC.npy  "
          f"(density {np.mean(SC > 0):.2f}, max {SC.max():.0f})")


if __name__ == "__main__":
    main()
