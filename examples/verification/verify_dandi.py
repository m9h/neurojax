
import os
import sys

# Ensure src is in path
sys.path.append(os.path.join(os.getcwd(), 'src'))

try:
    from neurojax.io.dandi import DandiLoader, find_high_freq_recordings
    print("[Success] Imported neurojax.io.dandi")
except ImportError as e:
    print(f"[Error] Failed to import: {e}")
    sys.exit(1)

def verify_dandiset(dandiset_id="000041"):
    print(f"Connecting to Dandiset {dandiset_id}...")
    try:
        loader = DandiLoader(dandiset_id)
        files = loader.list_files()
        print(f"Found {len(files)} files.")
        if not files:
            print("No files found. Exiting.")
            return

        # Pick the first NWB file
        nwb_path = files[0]
        print(f"Streaming {nwb_path}...")
        
        nwbfile, io = loader.stream_nwb(nwb_path)
        print(f"Opened NWB File: {nwbfile}")
        
        # Check for high frequency data
        recordings = find_high_freq_recordings(nwbfile, min_fs=20000.0)
        print(f"Found {len(recordings)} high-frequency (>20kHz) recordings.")
        
        for rec in recordings:
            print(f" - {rec.name}: {rec.rate} Hz, Shape: {rec.data.shape}")
            # Try reading a tiny chunk to prove streaming works
            chunk = rec.data[0:10]
            print(f"   Sample chunk: {chunk.shape}")

        io.close()
        print("[SUCCESS] DANDI Verification Complete.")

    except Exception as e:
        print(f"[FAILURE] DANDI Verification Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_dandiset()
