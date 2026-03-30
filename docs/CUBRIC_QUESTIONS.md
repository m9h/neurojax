# Questions for the CUBRIC Team (WAND Dataset)

## QMT Protocol Details
1. **MT pulse duration (Trf)**: What is the RF pulse duration for the qMT_TFL4 sequence? We need this for QUIT `qi qmt` fitting. The JSON sidecars don't include it. Estimated ~15ms but need confirmation.
2. **MT pulse shape**: Is it a Gaussian, Fermi, or hard pulse? What are the integral ratios (p1/p2) for the CW-equivalent power calculation?
3. **QMT B1 map**: Was a B1+ map acquired alongside the QMT data in ses-02? B1 correction is critical for accurate BPF estimation.
4. **QMT saturation angles**: The BIDS FlipAngle field shows 332/628/333 degrees — these are the MT saturation flip angles. Are these the nominal pulse angles or the effective angles after B1 correction?

## TRUST Protocol
5. **Effective echo times (eTEs)**: What are the T2 preparation eTEs for the `ep2d_TRUST_AsymShTE` sequence? Standard TRUST uses [0, 40, 80, 160] ms but the asymmetric short-TE variant may differ. This is critical for the T2→SvO₂ calibration.
6. **TRUST volume ordering**: Are the 24 volumes ordered as control-label pairs (ctrl1, label1, ctrl2, label2...) or blocks (ctrl1-12, label1-12)? And are they grouped by eTE?
7. **TRUST eTE encoding**: Is the eTE information encoded anywhere in the DICOM headers or protocol PDF that didn't make it into the BIDS conversion?

## Inversion Recovery (Blood T1)
8. **IR protocol**: The `mg_IR_T1_VE11c` sequence has 960 timepoints — is this a Look-Locker continuous readout or multi-TI with distinct inversions? The FrameTimesStart spacing is irregular.
9. **IR flip angle**: What is the readout flip angle for the Look-Locker train? Needed for the Deichmann-Haase T1* → T1 correction.
10. **Expected blood T1**: Has the CUBRIC team published blood T1 values from this protocol on the Connectom/Prisma?

## mcDESPOT / VFA
11. **SPGR flip angles**: The 4D SPGR has 8 volumes. Are the flip angles [2, 4, 6, 8, 10, 12, 14, 18]° (standard CUBRIC mcDESPOT)? The BIDS sidecar only shows FlipAngle=18 (presumably the last volume).
12. **SSFP flip angles and phase cycling**: The 4D SSFP has 16 volumes. Is this 8 FAs × 2 phase cycles (0°/180°), or 16 distinct FAs? What are the FA values?
13. **SPGR-IR**: What is the inversion time and readout scheme for the SPGR-IR? Is this for B1 mapping (DESPOT1-HIFI)?

## pCASL
14. **Labelling plane**: Where is the pCASL labelling plane positioned relative to anatomy? Is it at the level of the vertebral/internal carotid arteries?
15. **Background suppression**: Were background suppression pulses used? The JSON shows `SAT2` in ScanOptions.
16. **Multi-PLD**: Is this a single-PLD (2.0s) or multi-PLD acquisition? The 110 volumes suggest 55 control-label pairs at one PLD.

## Cross-Session Registration
17. **Within-subject alignment**: Has the CUBRIC team validated cross-session registration for this protocol? ses-02 (Connectom 300mT/m) and ses-03 (Prisma) are on different scanners — is a simple rigid-body registration sufficient or is there measurable geometric distortion between them?

## Data Sharing
18. **Full cohort**: Are all 170 subjects available on OpenNeuro, or only a subset? The current download shows ~170 subject directories.
19. **TMS-EEG format**: ses-08 TMS data is in .mat format — what software/toolbox generated these? Is there a BIDS sidecar or README describing the SICI protocol parameters?

## Skull / Head Modeling
20. **UTE/ZTE data**: Is there any ultrashort TE or zero TE acquisition in WAND for direct skull bone imaging? This would enable BabelBrain's direct approach rather than relying on pseudo-CT from T1w.
21. **Skull conductivity validation**: Has the CUBRIC team compared pseudo-CT-derived skull conductivity against the QMT/VFA measurements in bone? The qMRI data in ses-02 (QMT macromolecular content, VFA T1 in skull) could provide ground-truth bone density.

## MRS Basis Sets
24. **sLASER basis spectra**: Does CUBRIC have pre-computed basis spectra for the sLASER sequence used in ses-04 (TE=78ms, 7T)? Either as FSL-MRS .BASIS format, LCModel .basis, or VESPA output?
25. **MEGA-PRESS basis spectra**: Same question for the MEGA-PRESS GABA editing sequence in ses-05 (TE=68ms, 3T). Need both ON and OFF basis sets.
26. **Pulse sequence JSON**: Does CUBRIC have the pulse sequence timing files (RF pulse shapes, gradient timings) in a format compatible with fsl_mrs_sim for density matrix simulation?

## Software / Methods
22. **Sean Deoni mcDESPOT software**: Is the CUBRIC team using Deoni's original mcDESPOT MATLAB code, QUIT, or another tool for the VFA/mcDESPOT fitting? Any recommended parameter settings for this specific protocol?
23. **Processing pipeline**: Has the CUBRIC team published a recommended processing pipeline for the WAND qMRI data? Any parameter files or configs for QUIT/qMRLab?
