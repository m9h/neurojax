
import tarfile
import gzip
import io
import re
from pathlib import Path
from typing import Dict, List, Tuple
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

def load_uci_eeg(dataset_path: str, max_subjects: int = None) -> Tuple[jax.Array, Dict]:
    """
    Loads the UCI EEG Dataset (Begleiter et al.) from the nested tarball structure.
    
    Args:
        dataset_path: Path to the main tar.gz file (e.g., 'smni_eeg_data.tar.gz')
        max_subjects: Optional limit on number of subjects to load (for testing).
        
    Returns:
        X: JAX Array of shape (n_trials, n_channels, n_timepoints)
        metadata: Dictionary containing 'trials', 'channels', 'timepoints', 'subject_ids', 'conditions'
    """
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    # The dataset structure is:
    # smni_eeg_data.tar.gz -> folder smni_eeg_data/
    #   -> a_1_co2a0000364.tar.gz (Subject/Condition archive)
    #       -> co2a0000364.rd.000.gz (Trial file)
    
    # We need to accumulate data into a list of trials first
    # Each trial file contains 64 channels.
    
    # regex for plotting extraction
    # # 120 trials, 64 chans, 416 samples 368 post_stim samples
    # # 3.906000 msecs uV
    # # S1 obj , trial 0
    # # FP1 chan 0
    # 0 FP1 0 -8.921
    
    all_trials_data = []
    trial_metadata = []
    
    processed_subjects = 0
    
    with tarfile.open(path, "r:gz") as main_tar:
        # Iterate over subject archives
        for member in tqdm(main_tar.getmembers(), desc="Reading Subject Archives"):
            if not member.isfile() or not member.name.endswith(".tar.gz"):
                continue
                
            if max_subjects and processed_subjects >= max_subjects:
                break
                
            # Extract subject archive into memory
            f_obj = main_tar.extractfile(member)
            if f_obj is None:
                continue
                
            subject_tar_bytes = io.BytesIO(f_obj.read())
            
            with tarfile.open(fileobj=subject_tar_bytes, mode="r:gz") as sub_tar:
                for trial_member in sub_tar.getmembers():
                    if not trial_member.isfile() or not ".rd." in trial_member.name:
                        continue
                        
                    # Extract trial file
                    trial_f = sub_tar.extractfile(trial_member)
                    if trial_f is None:
                        continue
                        
                    # Handle gzipped trial files (usually they are .gz inside the tar)
                    # The filename usually ends in .gz, check member name
                    content_stream = trial_f
                    if trial_member.name.endswith(".gz"):
                        content_stream = gzip.GzipFile(fileobj=trial_f, mode='rb')
                    
                    try:
                        content_text = content_stream.read().decode('ascii')
                    except Exception as e:
                        print(f"Failed to decode {trial_member.name}: {e}")
                        continue
                        
                    # Parse the ASCII content
                    # We expect 64 channels * 256 samples (Wait, header says 416 samples? Let's verify)
                    # The file contains lines: TrialNum ChannelName SampleNum Value
                    
                    # Optimization: Use regex or simpler splitting
                    lines = content_text.strip().split('\n')
                    
                    # Parse header lines for metadata if needed, usually starting with #
                    # Skip comments
                    data_lines = [l for l in lines if not l.startswith('#')]
                    
                    # We need to organize this into (Channels, Time)
                    # Assuming standard channel order or we need to map them.
                    # Let's collect all points and pivot.
                    
                    # List of (ChanIdx, TimeIdx, Value)
                    # Better to dictionary first to ensure ordering
                    
                    # Standard 64 channel names might be needed if consistent ordering varies
                    # For now, let's assume we can sort by channel name or use the index from the file if available?
                    # The example line: "0 FP1 0 -8.921" -> Trial=0, Chan=FP1, Sample=0, Val=-8.921
                    
                    # Let's verify channel integer index presence.
                    # "0 FP1 0 -8.921"
                    # It seems "FP1" is the name. "chan 0" is in the header line above blocks.
                    
                    # Parsing strategy:
                    # 1. unique channels
                    # 2. unique time points
                    
                    # This is slow in pure python for 120 trials * 64 chans * 256 samples * N subjects.
                    # However, raw data is small enough.
                    
                    # Structured parse
                    t_data = [] # List of values
                    
                    # Use numpy for faster parsing if possible? 
                    # The data is "0 FP1 0 -8.921", space separated.
                    # We can drop the first column (trial num repeated) and channel name if we trust order.
                    # But channel order might permute.
                    
                    # Faster approach:
                    # Iterate lines, build dict {chan: [values sorted by time]}
                    
                    current_trial_data = {}
                    
                    for line in data_lines:
                        parts = line.split()
                        if len(parts) < 4: continue
                        
                        # trial_num = parts[0]
                        chan_name = parts[1]
                        # sample_num = int(parts[2])
                        val = float(parts[3])
                        
                        if chan_name not in current_trial_data:
                            current_trial_data[chan_name] = []
                        current_trial_data[chan_name].append(val)
                        
                    # Convert to array (N_chans, N_time)
                    # We enforce a sorted key order for channels to ensure consistency across trials
                    sorted_chans = sorted(current_trial_data.keys())
                    
                    # Create matrix
                    # Check dimensions
                    n_time = len(current_trial_data[sorted_chans[0]])
                    n_chans = len(sorted_chans)
                    
                    trial_mat = np.zeros((n_chans, n_time), dtype=np.float32)
                    for i, ch in enumerate(sorted_chans):
                        vals = current_trial_data[ch]
                        # Truncate or pad if necessary? usually fixed size
                        trial_mat[i, :len(vals)] = vals
                        
                    all_trials_data.append(trial_mat)
                    
                    # Metadata
                    trial_metadata.append({
                        'subject': member.name.split('.')[0], # rough subject id
                        'condition': 'matched' if 'a_m' in member.name or 'c_m' in member.name else 'non-matched', # heuristic based on filename
                        'trial_id': trial_member.name,
                        'channels': sorted_chans
                    })
                    
            processed_subjects += 1

    # Stack into one big array
    # Shape: (Trials, Channels, Time)
    # We might have different lengths? The paper mentions 256 samples (1 sec). 
    # The header file said 416 samples.
    # We should trim to common length.
    
    if not all_trials_data:
        return jnp.array([]), {}

    # Check shapes
    lens = [t.shape[1] for t in all_trials_data]
    min_len = min(lens)
    
    print(f"Trimming to minimum duration: {min_len} samples")
    X = np.stack([t[:, :min_len] for t in all_trials_data])
    
    info = {
        'channel_names': trial_metadata[0]['channels'],
        'sfreq': 256.0, # From paper/dataset desc
        'trial_info': trial_metadata
    }
    
    return jnp.array(X), info
