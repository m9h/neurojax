"""Inspect HCP trialinfo."""
import scipy.io
import numpy as np

fname = 'examples/hcp_data/104012_MEG_10-Motort_tmegpreproc_trialinfo.mat'

try:
    mat = scipy.io.loadmat(fname, squeeze_me=True, struct_as_record=False)
    print("Keys:", mat.keys())
    if 'trlInfo' in mat:
        info = mat['trlInfo']
        print(f"Type: {type(info)}")
        if hasattr(info, 'lockTrl'):
            lockTrl = info.lockTrl
            print(f"lockTrl shape: {lockTrl.shape}")
            # Likely a cell array of runs
            for i, run_element in enumerate(lockTrl):
                print(f"-- Run {i} --")
                if isinstance(run_element, np.ndarray) and run_element.ndim > 1:
                    triggers = run_element[:, -1]
                    print("Unique triggers:", np.unique(triggers))
                else:
                    print("Values:", run_element)
        pass
        print("First 20 rows:\n", info[:20])
except Exception as e:
    print(f"Failed: {e}")
