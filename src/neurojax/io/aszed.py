"""
ASZED (African Schizophrenia EEG Dataset) Loader.

References:
    - Zenodo: Search "ASZED" or "African Schizophrenia EEG Dataset"
"""
import os
import requests
import json
import logging
import zipfile
from pathlib import Path
from typing import Optional, Union, List, Dict

import mne
import numpy as np

logger = logging.getLogger(__name__)

ZENODO_API_URL = "https://zenodo.org/api/records"
DATASET_QUERY = "ASZED African Schizophrenia EEG Dataset"
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "neurojax" / "aszed"

def _resolve_zenodo_record() -> Dict:
    """Finds the ASZED dataset ID dynamically via Zenodo API."""
    params = {
        "q": DATASET_QUERY,
        "sort": "bestmatch",
        "size": 1
    }
    try:
        response = requests.get(ZENODO_API_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        if not data.get("hits", {}).get("hits"):
            raise ValueError(f"No Zenodo records found for query: {DATASET_QUERY}")
        
        record = data["hits"]["hits"][0]
        logger.info(f"Resolved ASZED dataset to Zenodo Record {record['id']} ({record['metadata']['title']})")
        return record
    except Exception as e:
        raise RuntimeError(f"Failed to resolve ASZED dataset from Zenodo: {e}")

def download_aszed(
    cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR,
    force_update: bool = False
) -> Path:
    """
    Downloads the ASZED dataset from Zenodo.
    
    Args:
        cache_dir: Directory to store the dataset.
        force_update: If True, re-downloads even if files exist.
        
    Returns:
        Path to the dataset directory.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Resolve Record
    record = _resolve_zenodo_record()
    record_id = str(record["id"])
    dataset_dir = cache_dir / record_id
    
    if dataset_dir.exists() and any(dataset_dir.iterdir()) and not force_update:
        logger.info(f"Dataset found at {dataset_dir}, skipping download.")
        return dataset_dir

    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Download Files
    files = record.get("files", [])
    if not files:
        raise ValueError("Zenodo record has no files attached.")
        
    for file_info in files:
        fname = file_info["key"] if "key" in file_info else file_info["filename"] # API v1 vs newer
        url = file_info["links"]["self"]
        out_path = dataset_dir / fname
        
        if out_path.exists() and not force_update:
            continue
            
        logger.info(f"Downloading {fname} from {url}...")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(out_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    
    # 3. Extract if zip
    for archive in dataset_dir.glob("*.zip"):
        logger.info(f"Extracting {archive.name}...")
        with zipfile.ZipFile(archive, 'r') as zf:
            zf.extractall(dataset_dir)
            
    return dataset_dir

def load_aszed_subject(
    subject_id: str,
    group: str = "SCZ",
    paradigm: str = "40hz_assr",
    cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR,
    download: bool = True
) -> mne.io.Raw:
    """
    Load data for a specific subject and paradigm.
    
    Args:
        subject_id: e.g., "001" or "1"
        group: "SCZ" (Schizophrenia) or "HC" (Healthy Control)
        paradigm: Paradigm substring to match (e.g., "40Hz", "Resting", "Oddball")
        cache_dir: Data location.
        download: Auto-download if missing.
        
    Returns:
        mne.io.Raw object.
    """
    dataset_dir = Path(cache_dir)
    # If using dynamic ID assumption, usually ID is a subdir. 
    # But user might pass a direct path. We check subdirs.
    if not any(dataset_dir.iterdir()) and download:
        path = download_aszed(cache_dir)
    else:
        # Find the record dir if it exists inside
        subdirs = [d for d in dataset_dir.iterdir() if d.is_dir()]
        if len(subdirs) == 1 and subdirs[0].name.isdigit():
             path = subdirs[0]
        else:
             path = dataset_dir

    # File naming in ASZED is typically: "Group_ID_Paradigm.edf" or similar structure
    # We walk to find a match.
    # Pattern: *{group}*{subject_id}*{paradigm}*
    
    # Normalize ID to prevent mismatches (e.g. "1" vs "001")
    # Actually ASZED usually uses "SCZ1", "SCZ01", etc. We'll fuzzy match filename.
    
    candidates = list(path.rglob(f"*{group}*{subject_id}*.edf"))
    if not candidates:
        # Try finding just by ID if group is implicit
        candidates = list(path.rglob(f"*{subject_id}*.edf"))
        
    # Filter by paradigm
    matched = [f for f in candidates if paradigm.lower() in f.name.lower().replace(" ", "")]
    
    if not matched:
        raise FileNotFoundError(
            f"No EDF file found for Subject={subject_id}, Group={group}, Paradigm={paradigm} in {path}. "
            f"Candidates found: {[f.name for f in candidates]}"
        )
        
    if len(matched) > 1:
        # Warn if multiple match, return first or strict error?
        logger.warning(f"Multiple matches found: {[f.name for f in matched]}. Loading first.")
        
    fpath = matched[0]
    logger.info(f"Loading {fpath}...")
    
    # Load with MNE
    raw = mne.io.read_raw_edf(fpath, preload=True)
    return raw
