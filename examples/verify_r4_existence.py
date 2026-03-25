
import fsspec

def check_r4():
    print("Checking S3 for Release 4...")
    fs = fsspec.filesystem("s3", anon=True)
    root = "fcp-indi/data/Projects/HBN/BIDS_EEG/"
    
    try:
        dirs = fs.ls(root)
        print("Found directories:")
        r4_found = False
        for d in dirs:
            print(f" - {d}")
            if "R4" in d or "Release4" in d:
                r4_found = True
                print(f"--> MATCH: {d}")
                
                # Check inside R4 for animation data
                print("Checking contents of R4 candidate...")
                sub_dirs = fs.ls(d)
                # Check a subject
                for sub in sub_dirs:
                    if "sub-" in sub:
                         eeg_files = fs.ls(f"{sub}/eeg/")
                         print(f"Checking {sub} files: {len(eeg_files)}")
                         for f in eeg_files[:10]: # Check first 10
                             if "DespicableMe" in f:
                                 print(f"[SUCCESS] Found DespicableMe in {f}")
                                 return
                         break # Just check one subject
        
        if not r4_found:
            print("No explicit 'R4' directory found. Might be 'cmi_bids' (latest)?")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_r4()
