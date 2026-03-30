# SMART MRS Dataset

## Source
- **Name:** SMART MRS — Simulated Metabolite Artifacts for Realistic Testing
- **Citation:** Bugler H, et al. "SMART MRS: A framework for generating
  simulated MRS datasets with realistic artifacts." Magnetic Resonance in
  Medicine (2024). DOI: 10.1002/mrm.30042
- **GitHub:** https://github.com/HarryBugler/SMART-MRS
- **Data:** Check GitHub releases or associated Zenodo deposit

## Description
Framework for generating simulated MEGA-PRESS spectra with realistic artifacts
including:
- Frequency drift
- Phase errors
- Lipid contamination
- Subject motion artifacts
- Residual water
- Subtraction artifacts

Provides ground-truth clean spectra alongside corrupted versions, making it
ideal for testing artifact correction and quality control algorithms.

## Format
- **MATLAB .mat** files (primary)
- May also include NIfTI-MRS or Python-compatible formats
- Simulated using FID-A or similar simulation frameworks

## Ground Truth
Full ground truth available:
- Clean (artifact-free) spectra
- Known artifact parameters (drift magnitude, phase error, etc.)
- Known metabolite concentrations for simulated spectra

## License
Check GitHub repository for license (likely MIT or BSD).

## Download

```bash
# Clone the repository
git clone https://github.com/HarryBugler/SMART-MRS.git

# Or download just the data
# Check releases: https://github.com/HarryBugler/SMART-MRS/releases
# Check Zenodo: search for "SMART MRS Bugler" on zenodo.org
```

## Notes
- The GitHub repo may contain both the simulation code AND pre-generated datasets
- If only code is in the repo, you may need to run the simulation to generate data
- Particularly useful for testing MEGA-PRESS artifact detection/correction
