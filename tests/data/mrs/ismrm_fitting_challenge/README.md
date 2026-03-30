# ISMRM MRS Fitting Challenge Dataset

## Source
- **Name:** ISMRM MRS Study Group Fitting Challenge
- **Citation:** Marjanska M, et al. "Results and interpretation of the fitChallenge."
  Presented at ISMRM 2021 Workshop on MRS.
- **Primary link:** https://experts.umn.edu/en/datasets/mrs-fitting-challenge-data-setup-by-ismrm-mrs-study-group
- **MRSHub listing:** https://mrshub.netlify.app/datasets_svs/
- **Data DOI:** 10.13020/3bk2-bv32

## Description
28 synthetic (simulated) MRS spectra at 3T with known ground-truth metabolite
concentrations. Designed to benchmark fitting algorithms (LCModel, OSPREY,
FSL-MRS, etc.). Includes spectra at varying SNR and linewidth conditions.

## Format
- **LCModel .RAW** format (primary)
- Also available as MATLAB .mat files
- Basis set included

## Ground Truth
Full ground-truth concentration tables for all 28 spectra are included in the
dataset. Metabolites include: NAA, NAAG, tCr, Cho, mI, Glu, Gln, GABA,
GSH, Tau, Asp, and others.

## License
CC-BY 4.0 (University of Minnesota Data Repository)

## Download

```bash
# Option 1: Download from UMN Data Repository
# Visit: https://conservancy.umn.edu/handle/11299/217895
# and download the ZIP file manually

# Option 2: Direct download (if available)
wget -O fitting_challenge.zip "https://conservancy.umn.edu/bitstream/handle/11299/217895/FittingChallenge.zip"
```

## Notes
- No registration required
- This is THE standard benchmark for MRS fitting algorithms
- Ground truth makes this ideal for validating quantification accuracy
