# MRSHub Edited MRS Example Datasets

## Source
- **Name:** MRSHub Example Datasets (Edited/MEGA-PRESS)
- **MRSHub datasets page:** https://mrshub.netlify.app/datasets_svs/
- **MRSHub GitHub org:** https://github.com/mrshub

## Datasets Available on MRSHub

### 1. Osprey Example Data (includes MEGA-PRESS)
- **URL:** https://github.com/schorschinho/osprey/tree/develop/exampledata
- **Description:** Example datasets shipped with Osprey, including MEGA-PRESS
  GABA data from Siemens, Philips, and GE scanners.
- **Format:** TWIX (.dat), SDAT/SPAR, P-file (.7)

### 2. FID-A Example Data
- **URL:** https://github.com/CIC-methods/FID-A/tree/master/exampleData
- **Description:** Example MRS data including edited sequences
- **Format:** Various vendor formats

### 3. MRSHub Example Datasets Repository
- **URL:** https://github.com/mrshub/mrshub-examples
- **Description:** Curated example datasets for MRS processing tutorials
- **Format:** Mixed (NIfTI-MRS, vendor-specific)

### 4. spec2nii Test Data
- **URL:** https://github.com/wtclarke/spec2nii/tree/master/tests
- **Description:** Test data for the NIfTI-MRS conversion tool, includes
  MEGA-PRESS examples from multiple vendors
- **Format:** Various vendor formats + NIfTI-MRS (.nii.gz)

### 5. Gannet Example Data
- **URL:** https://github.com/markmikkelsen/Gannet/tree/main/ExampleData
- **Description:** Example MEGA-PRESS data from the Gannet GABA fitting toolbox
- **Format:** Various vendor formats

## Download

```bash
# Osprey MEGA-PRESS examples
git clone --depth 1 --filter=blob:none --sparse https://github.com/schorschinho/osprey.git osprey_temp
cd osprey_temp && git sparse-checkout set exampledata/sdat/MEGA && cp -r exampledata/sdat/MEGA ../osprey_mega && cd .. && rm -rf osprey_temp

# spec2nii MEGA-PRESS test data (NIfTI-MRS format -- most useful)
git clone --depth 1 --filter=blob:none --sparse https://github.com/wtclarke/spec2nii.git spec2nii_temp
cd spec2nii_temp && git sparse-checkout set tests && cp -r tests ../spec2nii_tests && cd .. && rm -rf spec2nii_temp

# Gannet example data
wget https://github.com/markmikkelsen/Gannet/archive/refs/heads/main.zip -O gannet.zip
# Then extract just ExampleData/

# Alternatively, download individual files from the Osprey example datasets:
# https://github.com/schorschinho/osprey/raw/develop/exampledata/sdat/MEGA/sub-01/mrs/sub-01_MEGA.sdat
# https://github.com/schorschinho/osprey/raw/develop/exampledata/sdat/MEGA/sub-01/mrs/sub-01_MEGA.spar
```

## License
- Osprey: BSD-3-Clause
- FID-A: BSD-3-Clause
- spec2nii: BSD-3-Clause
- Gannet: BSD-2-Clause

## Notes
- These are real in-vivo datasets (no ground truth concentrations)
- Most useful for testing I/O, preprocessing, and format conversion
- spec2nii test data is particularly useful as it covers NIfTI-MRS format
- Osprey and Gannet examples are the most relevant for MEGA-PRESS GABA
