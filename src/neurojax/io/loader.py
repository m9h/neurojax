from pathlib import Path
import mne

def load_data(fname: str | Path, preload: bool = True) -> mne.io.BaseRaw:
    fname = str(Path(fname).resolve())
    return mne.io.read_raw(fname, preload=preload)
