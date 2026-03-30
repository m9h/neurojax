# Big GABA Dataset

## Source
- **Name:** Big GABA — A Multi-site MEGA-PRESS Study
- **Citation:** Mikkelsen M, et al. "Big GABA: Edited MR spectroscopy at 24
  research sites." NeuroImage 159 (2017): 32-45.
  DOI: 10.1016/j.neuroimage.2017.07.021
- **NITRC:** https://www.nitrc.org/projects/biggaba/
- **MRSHub:** https://mrshub.netlify.app/datasets_svs/

## Description
Multi-site MEGA-PRESS GABA-edited MRS dataset acquired at 24 sites worldwide
(285 subjects total). Acquired on GE, Philips, and Siemens scanners at 3T.
This is the definitive multi-vendor MEGA-PRESS reproducibility dataset.

Each subject has:
- MEGA-PRESS GABA-edited spectra (TE=68ms, 320 averages typical)
- Water reference
- Structural T1w MRI for tissue correction

## Format
- **Siemens TWIX** (.dat)
- **Philips SDAT/SPAR** (.sdat/.spar)
- **GE P-file** (.7)
- Some data also available in NIfTI-MRS format via conversion

## Ground Truth
No absolute ground-truth concentrations (real in-vivo data), but extensive
multi-site reproducibility metrics are published. GABA+/tCr ratios and
tissue-corrected GABA concentrations are reported for all sites.

## License
Data use agreement required via NITRC.

## Download

```bash
# REGISTRATION REQUIRED
# 1. Create account at https://www.nitrc.org/account/register.php
# 2. Go to https://www.nitrc.org/projects/biggaba/
# 3. Accept data use agreement
# 4. Download from the "Downloads" tab

# After registration, data can be downloaded via NITRC command line:
# nitrc_ir download -p biggaba
```

## Notes
- **Registration required** -- cannot be downloaded without NITRC account + DUA
- This is the gold-standard MEGA-PRESS multi-site dataset
- Data includes all 3 major MR vendors (GE, Philips, Siemens)
- Very large dataset (~50 GB total for all sites)
- Consider downloading a single site for testing (e.g., Site 01 Siemens)
