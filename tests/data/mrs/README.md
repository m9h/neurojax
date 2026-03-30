# MRS Test Datasets

Test datasets for validating the MEGA-PRESS spectral editing pipeline.

## Datasets

| Dataset | Type | Ground Truth | Format | Registration |
|---------|------|-------------|--------|-------------|
| `ismrm_fitting_challenge/` | Synthetic SVS (28 spectra) | Yes (full concentrations) | LCModel .RAW, .mat | None |
| `big_gaba/` | In-vivo MEGA-PRESS (285 subj, 24 sites) | No (reproducibility metrics) | TWIX, SDAT, P-file | **NITRC DUA required** |
| `smart_mrs/` | Simulated MEGA-PRESS with artifacts | Yes (clean spectra + artifact params) | .mat, possibly NIfTI-MRS | None |
| `mrshub_edited_examples/` | In-vivo MEGA-PRESS examples | No | Various + NIfTI-MRS | None |

## Quick Start

Run the download script to fetch all freely available datasets:

```bash
cd /home/mhough/dev/neurojax/tests/data/mrs/
bash download_all.sh
```

## Notes
- Big GABA requires NITRC registration and a data use agreement
- The ISMRM Fitting Challenge is the best dataset for quantification validation
- SMART MRS is best for artifact correction algorithm testing
- MRSHub examples are best for I/O and format conversion testing
