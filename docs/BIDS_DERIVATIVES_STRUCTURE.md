# BIDS Derivatives Structure for WAND Processing

## Design Principles

1. **Native space first** — all processing stays in native (subject) space as long as possible
2. **BIDS derivatives compliant** — follows [BEP003](https://bids-specification.readthedocs.io/en/stable/derivatives/introduction.html)
3. **FSL DL registration when needed** — FSL's new DL registration tool + OM-1 template for standard space
4. **Provenance** — each pipeline produces a `dataset_description.json` with tool versions
5. **Interoperability** — outputs consumable by neurojax `BIDSConnectomeLoader` and `WANDMEGLoader`

## Directory Layout

```
wand/
├── sub-08033/                          # Raw BIDS (unchanged)
│   ├── ses-01/meg/                     # CTF .ds MEG data
│   ├── ses-02/dwi/                     # AxCaliber + CHARMED
│   ├── ses-02/anat/                    # T1w, VFA, QMT
│   ├── ses-03/anat/                    # T1w, T2w
│   ├── ses-03/func/                    # fMRI
│   ├── ses-04/mrs/                     # MRS (sLASER, 4 VOIs)
│   ├── ses-05/mrs/                     # MRS session 2
│   ├── ses-06/anat/                    # MP2RAGE, multi-echo GRE
│   └── ses-08/tms/                     # TMS-EEG SICI
│
└── derivatives/
    │
    ├── fsl-anat/                       # Stage A: Structural preprocessing
    │   ├── dataset_description.json
    │   └── sub-08033/
    │       ├── ses-02/anat/
    │       │   └── sub-08033_ses-02_T1w.anat/   # fsl_anat output
    │       │       ├── T1_biascorr.nii.gz        # bias-corrected T1
    │       │       ├── T1_biascorr_brain.nii.gz  # brain extracted
    │       │       ├── T1_biascorr_brain_mask.nii.gz
    │       │       ├── T1_fast_seg.nii.gz        # FAST tissue seg
    │       │       ├── T1_fast_pve_0.nii.gz      # CSF PVE
    │       │       ├── T1_fast_pve_1.nii.gz      # GM PVE
    │       │       ├── T1_fast_pve_2.nii.gz      # WM PVE
    │       │       ├── first_results/             # FIRST subcortical
    │       │       ├── T1_to_MNI_nonlin_coeff.nii.gz   # FNIRT warp
    │       │       └── T1_to_MNI_nonlin_field.nii.gz
    │       └── ses-03/anat/
    │           └── sub-08033_ses-03_T1w.anat/
    │
    ├── freesurfer/                     # Stage B: Surface reconstruction
    │   ├── dataset_description.json
    │   └── sub-08033/                  # Standard FreeSurfer layout
    │       ├── mri/
    │       │   ├── brain.mgz
    │       │   ├── aparc+aseg.mgz      # Desikan parcellation
    │       │   ├── aparc.a2009s+aseg.mgz  # Destrieux
    │       │   ├── ThalamicNuclei.v13.T1.mgz  # Thalamic nuclei
    │       │   └── hippoSfVolumes-T1.v22.txt
    │       ├── surf/
    │       │   ├── lh.white, rh.white
    │       │   ├── lh.pial, rh.pial
    │       │   ├── lh.sphere.reg, rh.sphere.reg
    │       │   └── lh.T1wT2w_ratio.mgz  # Myelin map
    │       ├── label/
    │       └── scripts/
    │
    ├── fsl-dwi/                        # Stage C: DWI preprocessing
    │   ├── dataset_description.json
    │   └── sub-08033/
    │       └── ses-02/dwi/
    │           ├── topup/
    │           │   ├── topup_results_fieldcoef.nii.gz
    │           │   └── topup_results_movpar.txt
    │           ├── eddy/
    │           │   ├── AxCaliber1_eddy.nii.gz     # Corrected volumes
    │           │   ├── AxCaliber1_eddy.eddy_rotated_bvecs
    │           │   ├── AxCaliber2_eddy.nii.gz
    │           │   ├── AxCaliber3_eddy.nii.gz
    │           │   ├── AxCaliber4_eddy.nii.gz
    │           │   ├── CHARMED_eddy.nii.gz
    │           │   └── eddy_qc/                   # eddy_quad QC
    │           ├── dtifit/
    │           │   ├── dti_FA.nii.gz               # Native space
    │           │   ├── dti_MD.nii.gz
    │           │   ├── dti_V1.nii.gz               # Principal eigenvector
    │           │   └── dti_L1.nii.gz
    │           ├── nodif_brain.nii.gz
    │           ├── nodif_brain_mask.nii.gz
    │           └── acqparams.txt
    │
    ├── fsl-bedpostx/                   # Stage D: Fiber orientation estimation
    │   ├── dataset_description.json
    │   └── sub-08033/
    │       └── ses-02/dwi/
    │           └── CHARMED.bedpostX/
    │               ├── merged_f1samples.nii.gz     # Fiber fraction 1
    │               ├── merged_th1samples.nii.gz     # Theta 1
    │               ├── merged_ph1samples.nii.gz     # Phi 1
    │               ├── merged_f2samples.nii.gz     # Fiber fraction 2
    │               ├── merged_f3samples.nii.gz     # Fiber fraction 3
    │               ├── mean_f1samples.nii.gz
    │               ├── dyads1.nii.gz               # Mean fiber direction
    │               └── nodif_brain_mask.nii.gz
    │
    ├── fsl-xtract/                     # Stage E: Automated tract segmentation
    │   ├── dataset_description.json
    │   └── sub-08033/
    │       └── ses-02/
    │           ├── tracts/
    │           │   ├── af_l/                       # Arcuate fasciculus L
    │           │   │   ├── densityNorm.nii.gz      # Native space
    │           │   │   └── waytotal
    │           │   ├── af_r/
    │           │   ├── cst_l/                      # Corticospinal tract
    │           │   ├── cst_r/
    │           │   ├── fx/                         # Fornix
    │           │   ├── ifo_l/                      # IFOF
    │           │   ├── ilf_l/                      # ILF
    │           │   ├── slf1_l/ slf2_l/ slf3_l/     # SLF subdivisions
    │           │   ├── unc_l/ unc_r/               # Uncinate
    │           │   └── fma/ fmi/                   # Forceps major/minor
    │           └── stats/
    │               ├── xtract_stats_FA.csv
    │               └── xtract_stats_MD.csv
    │
    ├── fsl-tracula/                    # Stage F: FreeSurfer+FSL tracts
    │   ├── dataset_description.json
    │   └── sub-08033/
    │       └── dmri/
    │           ├── dpath/
    │           │   ├── lh.cst_AS/                  # Corticospinal
    │           │   ├── rh.cst_AS/
    │           │   ├── fmajor_PP/                  # Forceps major
    │           │   └── fminor_PP/
    │           └── pathstats.overall.txt
    │
    ├── fsl-reg/                        # Stage G: Registration transforms
    │   ├── dataset_description.json
    │   └── sub-08033/
    │       ├── native-to-struct/                   # DWI/func → T1w
    │       │   ├── ses-02_dwi_to_T1w.mat           # FLIRT affine
    │       │   ├── ses-03_func_to_T1w.mat
    │       │   └── ses-04_mrs_to_T1w.mat
    │       ├── struct-to-standard/                 # T1w → MNI/OM-1
    │       │   ├── T1w_to_MNI_affine.mat           # FLIRT
    │       │   ├── T1w_to_MNI_warp.nii.gz          # FNIRT
    │       │   ├── MNI_to_T1w_warp.nii.gz          # Inverse
    │       │   ├── T1w_to_OM1_warp.nii.gz          # FSL DL reg (when available)
    │       │   └── OM1_to_T1w_warp.nii.gz
    │       └── cross-session/                      # ses-XX → ses-02 (anchor)
    │           ├── ses-03_to_ses-02.mat
    │           ├── ses-04_to_ses-02.mat
    │           ├── ses-05_to_ses-02.mat
    │           └── ses-06_to_ses-02.mat
    │
    ├── fsl-connectome/                 # Stage H: Structural connectivity
    │   ├── dataset_description.json
    │   └── sub-08033/
    │       └── ses-02/
    │           ├── sub-08033_atlas-desikan_desc-sift2_connectivity.csv
    │           ├── sub-08033_atlas-desikan_desc-meanlength_connectivity.csv
    │           ├── sub-08033_atlas-destrieux_desc-sift2_connectivity.csv
    │           ├── sub-08033_atlas-destrieux_desc-meanlength_connectivity.csv
    │           └── parcellation/
    │               ├── desikan_parc_diff.nii.gz     # Native DWI space
    │               └── destrieux_parc_diff.nii.gz
    │
    ├── fsl-mrs/                        # Stage I: MRS quantification
    │   ├── dataset_description.json
    │   └── sub-08033/
    │       ├── ses-04/mrs/
    │       │   ├── anteriorcingulate/
    │       │   │   ├── voxel_mask_T1w.nii.gz       # SVS voxel in T1 space
    │       │   │   ├── tissue_fractions.json        # GM/WM/CSF fracs
    │       │   │   ├── fit_report.html              # fsl_mrs report
    │       │   │   ├── concentrations.csv           # GABA, Glu, NAA, etc.
    │       │   │   └── spectra/
    │       │   │       ├── data.nii.gz              # Preprocessed FID
    │       │   │       ├── fit.nii.gz
    │       │   │       └── residual.nii.gz
    │       │   ├── occipital/
    │       │   ├── rightauditory/
    │       │   └── smleft/
    │       └── ses-05/mrs/
    │           └── (same structure)
    │
    ├── microstructure/                 # Stage J: Biophysical models
    │   ├── dataset_description.json
    │   └── sub-08033/
    │       └── ses-02/
    │           ├── axcaliber_combined.nii.gz        # Merged eddy-corrected
    │           ├── axcaliber_combined.bval
    │           ├── axcaliber_combined.bvec
    │           ├── dipy/                            # DIPY estimates
    │           │   ├── dti_FA.nii.gz
    │           │   ├── dki_MK.nii.gz
    │           │   └── mapmri_RTOP.nii.gz
    │           ├── sbi4dwi/                         # dmipy-jax estimates
    │           │   ├── axcaliber_diameter.nii.gz     # Native space
    │           │   ├── axcaliber_vf.nii.gz           # Volume fraction
    │           │   ├── sandi_fsoma.nii.gz
    │           │   └── velocity_map.nii.gz           # Hursh-Rushton
    │           └── comparison/
    │               └── cross_method_stats.csv
    │
    ├── qmri/                           # Stage K: Quantitative MRI
    │   ├── dataset_description.json
    │   └── sub-08033/
    │       ├── ses-02/
    │       │   ├── T1map.nii.gz                    # VFA T1 mapping
    │       │   ├── QMT_bpf.nii.gz                  # Bound pool fraction (myelin)
    │       │   └── QMT_kf.nii.gz                   # Forward exchange rate
    │       └── ses-06/
    │           ├── T2star_map.nii.gz                # Multi-echo GRE fit
    │           ├── R2star_map.nii.gz
    │           └── T1_MP2RAGE.nii.gz
    │
    ├── advanced-freesurfer/            # Stage L: Beyond recon-all
    │   ├── dataset_description.json
    │   └── sub-08033/
    │       ├── myelin/
    │       │   ├── T1w_T2w_ratio.nii.gz            # Native T1 space
    │       │   ├── T1w_T2w_ratio_norm.nii.gz
    │       │   ├── lh.T1wT2w_ratio.mgz             # Surface
    │       │   └── rh.T1wT2w_ratio.mgz
    │       ├── samseg/                              # Multimodal segmentation
    │       ├── synthseg/                            # Cross-session parcellation
    │       │   ├── ses-02_synthseg.nii.gz
    │       │   ├── ses-03_synthseg.nii.gz
    │       │   ├── ses-04_synthseg.nii.gz
    │       │   └── ses-06_synthseg.nii.gz
    │       ├── thalamus/                            # Thalamic nuclei
    │       ├── hippocampus/                         # Subfields
    │       └── hypothalamus/                        # Subunits
    │
    ├── neurojax-meg/                   # Stage M: MEG dynamics
    │   ├── dataset_description.json
    │   └── sub-08033/
    │       └── ses-01/
    │           ├── source/
    │           │   ├── parcellated_desikan.npy       # (T, 68)
    │           │   └── parcellated_destrieux.npy     # (T, 148)
    │           ├── hmm/
    │           │   ├── state_probabilities.npy       # (T, K) gamma
    │           │   ├── state_means.npy               # (K, C)
    │           │   ├── state_covariances.npy         # (K, C, C)
    │           │   └── summary_stats.json            # FO, lifetime, SR
    │           ├── dynemo/
    │           │   ├── alpha.npy                     # (T, K)
    │           │   ├── mode_means.npy
    │           │   └── mode_covariances.npy
    │           └── spectra/
    │               ├── state_psd.npy                 # (K, F, C)
    │               ├── state_coherence.npy            # (K, F, C, C)
    │               └── frequencies.npy
    │
    └── neurojax-tms/                   # Stage N: TMS fitting
        ├── dataset_description.json
        └── sub-08033/
            └── ses-08/
                ├── empirical_tep.npy                 # (n_sensors, n_samples)
                ├── fitted_params.json                # G, per-region A, etc.
                ├── simulated_tep.npy                 # Fitted model output
                ├── contribution_matrix.npy            # (n_regions, n_timepoints)
                ├── local_network_transition.json      # t_transition in ms
                └── virtual_lesion/
                    └── region_XX_effect.npy
```

## Registration Strategy

### Native Space Anchor: ses-02 T1w
All sessions register TO ses-02 T1w (the DWI session), keeping the structural
connectome as the spatial reference.

### Cross-Session Registration
```
ses-01 (MEG) → ses-02 T1w  (via MNE coregistration, head model)
ses-03 T1w   → ses-02 T1w  (FLIRT 6-DOF rigid)
ses-04 T1w   → ses-02 T1w  (FLIRT 6-DOF rigid)
ses-05 T1w   → ses-02 T1w  (FLIRT 6-DOF rigid)
ses-06 T1w   → ses-02 T1w  (FLIRT 6-DOF rigid)
```

### Standard Space Registration (when needed)
```
Native T1w → MNI152 1mm   : FLIRT (affine) + FNIRT (nonlinear)
Native T1w → OM-1 template: FSL DL registration (when available)
```

The OM-1 template and FSL's new DL registration tool will replace FNIRT
for standard space transforms once available. Until then, FNIRT serves
as the fallback.

### MRS Voxel Registration
```
SVS .dat → svs_segment → voxel_mask in T1w space → tissue fractions
                       → fsl_mrs --tissue_frac for water-scaled quantification
```

## Processing Order

```
A. fsl_anat (T1w from ses-02 and ses-03)          ← first, provides brain mask + FAST
B. FreeSurfer recon-all (ses-02 T1w + ses-03 T2w)  ← parallel with A
C. DWI preprocessing (topup + eddy)                 ← parallel with A/B
D. bedpostx (after C)                                ← GPU-accelerated if available
E. xtract (after D + registration from G)
F. TRACULA (after B + D)
G. Registration transforms (after A + B)             ← all native→struct→standard
H. Connectome construction (after D + G)
I. MRS quantification (after A for tissue seg)
J. Microstructure estimation (after C)               ← DIPY/sbi4dwi/DMI.jl
K. Quantitative MRI (VFA, QMT, T2*)                 ← parallel with everything
L. Advanced FreeSurfer (after B)
M. MEG dynamics (after B + H for source model)
N. TMS fitting (after H + M)
```

## dataset_description.json Template

Each derivatives directory contains:

```json
{
    "Name": "fsl-dwi",
    "BIDSVersion": "1.9.0",
    "PipelineDescription": {
        "Name": "FSL DWI Preprocessing",
        "Version": "6.0.7.22",
        "CodeURL": "https://github.com/m9h/neurojax/tree/master/scripts/wand_processing"
    },
    "GeneratedBy": [{
        "Name": "FSL",
        "Version": "6.0.7.22",
        "Container": {
            "Type": "native",
            "Tag": "macOS"
        }
    }],
    "SourceDatasets": [{
        "URL": "https://gin.g-node.org/CUBRIC/WAND",
        "Version": "1.0.0"
    }]
}
```
