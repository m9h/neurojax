
import os
import subprocess
import shutil
from pathlib import Path

# Configuration
DATASET_URL = "https://github.com/childmindresearch/HBN-EEG-BIDS.git"
INSTALL_DIR = Path("downloads/HBN_R4_Probe")
TARGET_RELEASE = "R4" # Searching for this in tags

def run_cmd(cmd, cwd=None):
    print(f"RUNNING: {cmd}")
    try:
        subprocess.run(cmd, shell=True, check=True, cwd=cwd)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: {e}")
        return False
    return True

def probe_hbn():
    if os.path.exists(INSTALL_DIR):
        print(f"Directory {INSTALL_DIR} exists. Removing for fresh probe...")
        shutil.rmtree(INSTALL_DIR)
    
    os.makedirs(INSTALL_DIR.parent, exist_ok=True)
    
    print(f"--- 1. Installing Metadata from {DATASET_URL} ---")
    # Using datalad install (shallow if possible? DataLad doesn't strongly support shallow clones of git-annex easily without --dataset)
    # But just cloning the git part is fast.
    if not run_cmd(f"datalad install {DATASET_URL} {INSTALL_DIR}"):
        print("Failed to install dataset.")
        return

    print("\n--- 2. Checking Releases (Tags) ---")
    # List tags
    result = subprocess.run("git tag", shell=True, cwd=INSTALL_DIR, capture_output=True, text=True)
    tags = result.stdout.splitlines()
    print(f"Available tags: {tags}")
    
    # Try to find R4
    r4_tag = None
    for t in tags:
        if "R4" in t or "release-4" in t.lower() or "v4" in t:
            r4_tag = t
            break
            
    if r4_tag:
        print(f"--> Found Release 4 Candidate: {r4_tag}. Checking out...")
        run_cmd(f"git checkout {r4_tag}", cwd=INSTALL_DIR)
    else:
        print("--> No explicit 'R4' tag found. staying on default branch (likely latest).")

    print("\n--- 3. Probing for Animation/Movie Tasks ---")
    # Look for files with 'task-DespicableMe', 'task-ThePresent', etc.
    # We can use find or just walk.
    # We'll check a few subjects.
    
    found_tasks = set()
    subject_count = 0
    
    for root, dirs, files in os.walk(INSTALL_DIR):
        if ".git" in root: continue
        
        for f in files:
            if "task-" in f and ("eeg.json" in f or "eeg.set" in f):
                # Extract task name
                parts = f.split("_")
                for p in parts:
                    if p.startswith("task-"):
                        task = p.split("-")[1]
                        found_tasks.add(task)
                        
        if "sub-" in root:
            subject_count += 1
            
    print(f"Scanned {subject_count} directory visits (approx).")
    print(f"Found Tasks: {os.path.basename(str(found_tasks))}") # Just printing the set
    
    for t in found_tasks:
        print(f" - {t}")
        
    required_animations = ["DespicableMe", "ThePresent", "DiaryOfAWimpyKid"]
    matches = [t for t in found_tasks if t in required_animations]
    
    if matches:
        print(f"\n[SUCCESS] Found Animation Task Data: {matches}")
    else:
        print("\n[WARNING] No Animation Task Data found in this release/dataset.")

if __name__ == "__main__":
    probe_hbn()
