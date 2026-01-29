# OpenNeuro Data Access Standards

> [!IMPORTANT]
> **Do NOT use `openneuro-py` for accessing binary EEG/MEG files.**
> It reliably retrieves metadata (JSON/TSV) but typically fails to download binary files (`.eeg`, `.set`, `.vhdr`) or access S3 buckets correctly.

## The Standard: Use `datalad`

For all `neurojax` data ingestion tasks involving OpenNeuro, **you must use Datalad**.

### Installation
Ensure `datalad` is installed:
```bash
pip install datalad
# OR
conda install -c conda-forge datalad
```

### Workflow
1.  **Install the Dataset** (Clones the repository structure only):
    ```bash
    datalad install https://github.com/OpenNeuroDatasets/dsXXXXXX.git
    cd dsXXXXXX
    ```

2.  **Retrieve Specific Data** (Downloads the actual large binary files):
    ```bash
    # Get all potential data for subject 001
    datalad get sub-001/ses-01/eeg/*
    
    # Or get the whole subject recursively
    datalad get -r sub-001/
    ```

### Why?
- **Git-Annex Pointers:** OpenNeuro datasets are git-annex repositories. The files you see in a clone are symlinks (pointers). `datalad get` resolves these pointers by downloading the content from S3/Amazon.
- **Reliability:** Datalad handles the S3 backend resolution robustly, whereas the S3 direct download tools often fail on "requester pays" or permission issues for public datasets.

### Example Skills
See `neurojax_asr_validation` for a reference implementation using Datalad.
