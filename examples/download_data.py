
import os
import openneuro

def download_data():
    dataset = 'ds003768'
    target_dir = 'downloads/ds003768'
    os.makedirs(target_dir, exist_ok=True)
    
    print(f"Downloading {dataset} (sub-01) to {target_dir}...")
    try:
        openneuro.download(dataset=dataset, target_dir=target_dir, include=['sub-01'])
        print("Download complete.")
    except Exception as e:
        print(f"Error downloading data: {e}")

if __name__ == "__main__":
    download_data()
