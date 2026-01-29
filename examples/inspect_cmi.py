
import fsspec
import os

def list_hbn_tasks(subject_id="sub-NDARAA075AMK"):
    print(f"Inspecting CMI HBN S3 for {subject_id}...")
    fs = fsspec.filesystem("s3", anon=True)
    
    base_path = f"fcp-indi/data/Projects/HBN/BIDS_EEG/cmi_bids_R1/"
    print(f"Listing release: {base_path}")
    
    try:
        files = fs.ls(base_path)
        print(f"Found {len(files)} subjects in R1")
        
        # Find a valid subject
        for subj_path in files:
            subj = os.path.basename(subj_path)
            if not subj.startswith("sub-"):
                continue
                
            print(f"\nChecking subject: {subj}")
            eeg_path = f"{subj_path}/eeg/"
            try:
                eeg_files = fs.ls(eeg_path)
                print(f"Found {len(eeg_files)} EEG files:")
                has_rest = False
                has_task = False
                
                for ef in eeg_files:
                    fname = os.path.basename(ef)
                    print(f"   - {fname}")
                    if "RestingState" in fname: has_rest = True
                    if "Surround" in fname or "Contrast" in fname or "Video" in fname: has_task = True
                
                if has_rest and has_task:
                    print(f"--> [MATCH] Subject {subj} has both Resting and Task!")
                    break
            except:
                pass
    except Exception as e:
        print(f"[ERROR] Failed to list bucket: {e}")
    except Exception as e:
        print(f"[ERROR] Failed to list bucket: {e}")
    except Exception as e:
        print(f"[ERROR] Failed to list bucket: {e}")

if __name__ == "__main__":
    list_hbn_tasks()
