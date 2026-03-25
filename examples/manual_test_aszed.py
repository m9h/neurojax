import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Mock MNE
from unittest.mock import MagicMock
sys.modules["mne"] = MagicMock()

from neurojax.io.aszed import _resolve_zenodo_record

print("Attempting to resolve ASZED dataset...")
try:
    record = _resolve_zenodo_record()
    print(f"SUCCESS: Resolved Record ID: {record['id']}")
    print(f"Title: {record['metadata']['title']}")
    print(f"Files: {len(record.get('files', []))}")
except Exception as e:
    print(f"FAILURE: {e}")
