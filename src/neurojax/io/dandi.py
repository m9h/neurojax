"""
DANDI Archive Integration for NeuroJAX.

This module provides tools to stream Neural Data Without Borders (NWB) files
directly from the DANDI Archive using fsspec and pynwb, avoiding the need
for full downloads of massive datasets.
"""

import os
from typing import List, Optional, Generator, Any
from dandi.dandiapi import DandiAPIClient, RemoteDandiset
import pynwb
from pynwb import NWBHDF5IO
import fsspec
import h5py
from fsspec.implementations.cached import CachingFileSystem

class DandiLoader:
    def __init__(self, dandiset_id: str, version: str = "draft"):
        """
        Initialize DANDI loader for a specific dataset.
        
        Args:
            dandiset_id (str): The ID of the Dandiset (e.g. '000005')
            version (str): Version to use (default: 'draft')
        """
        self.dandiset_id = dandiset_id
        self.version = version
        self.client = DandiAPIClient()

    def list_files(self, path_prefix: str = "") -> List[str]:
        """List files in the Dandiset, optionally filtering by path."""
        dandiset = self.client.get_dandiset(self.dandiset_id, self.version)
        assets = dandiset.get_assets()
        
        # Simple client-side filtering (API might support path filter in future)
        files = []
        for asset in assets:
            path = asset.path
            if path.startswith(path_prefix):
                files.append(path)
        return files

    def stream_nwb(self, file_path: str) -> pynwb.file.NWBFile:
        """
        Stream an NWB file from DANDI without full download.
        
        Args:
            file_path (str): Path within the Dandiset (e.g. 'sub-01/sub-01_ecephys.nwb')
            
        Returns:
            pynwb.file.NWBFile: The opened NWB file object.
            Note: This file must be kept open while accessing lazy data.
        """
        dandiset = self.client.get_dandiset(self.dandiset_id, self.version)
        asset = dandiset.get_asset_by_path(file_path)
        
        # Get the S3 URL (Blob Asset)
        s3_url = asset.get_content_url(follow_redirects=1, strip_query=True)
        
        # Setup caching filesystem for performance
        fs = fsspec.filesystem("http")
        
        # Open via h5py using the ROS3 driver or fsspec file object
        # Using fsspec -> h5py is robust for HTTP streaming
        f = fs.open(s3_url, "rb")
        file = h5py.File(f, "r")
        io = NWBHDF5IO(file=file, mode='r', load_namespaces=True)
        nwbfile = io.read()
        return nwbfile, io

def find_high_freq_recordings(nwbfile: pynwb.file.NWBFile, min_fs: float = 20000.0) -> List[Any]:
    """
    Find ElectricalSeries with sampling rate > min_fs (e.g. 20kHz).
    """
    recordings = []
    # Search acquisition
    for name, obj in nwbfile.acquisition.items():
        if isinstance(obj, pynwb.ecephys.ElectricalSeries):
            if obj.rate and obj.rate >= min_fs:
                recordings.append(obj)
    
    # Search processing (LFP is usually low, but filtered data might be here)
    for processing_module in nwbfile.processing.values():
        if processing_module.data_interfaces:
             for name, obj in processing_module.data_interfaces.items():
                 if isinstance(obj, pynwb.ecephys.LFP): # LFP is specific, check content
                     for es_name, es in obj.electrical_series.items():
                         if es.rate and es.rate >= min_fs:
                             recordings.append(es)
                 elif isinstance(obj, pynwb.ecephys.ElectricalSeries):
                     if obj.rate and obj.rate >= min_fs:
                         recordings.append(obj)
                         
    return recordings
