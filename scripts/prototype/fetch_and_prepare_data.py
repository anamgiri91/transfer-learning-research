import os
import requests
import zipfile
from pathlib import Path
from tqdm import tqdm

def download_zenodo_record(record_id: str, output_dir: Path):
    """Downloads files from a Zenodo record."""
    url = f"https://zenodo.org/api/records/{record_id}"
    print(f"Fetching metadata for Zenodo record {record_id}...")
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for file_info in data.get('files', []):
        file_url = file_info['links']['self']
        file_name = file_info['key']
        file_size = file_info['size']
        file_path = output_dir / file_name
        
        if file_path.exists() and file_path.stat().st_size == file_size:
            print(f"File {file_name} already exists. Skipping download.")
            continue
            
        print(f"Downloading {file_name} ({file_size / (1024*1024):.2f} MB)...")
        with requests.get(file_url, stream=True) as r:
            r.raise_for_status()
            with open(file_path, 'wb') as f, tqdm(
                desc=file_name,
                total=file_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in r.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    bar.update(size)

def main():
    PROJECT_ROOT = Path(__file__).parent.parent
    RAW_DIR = PROJECT_ROOT / "data" / "raw"
    
    # 1. Download OpenBind EV-A71 2A Data (Zenodo 20026661)
    print("Step 1: Downloading dataset...")
    download_zenodo_record("20026661", RAW_DIR)
    
    # 2. Extract Data
    print("\nStep 2: Extracting dataset...")
    zip_path = RAW_DIR / "OpenBind_EV-A71_2A.zip"
    if zip_path.exists():
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(RAW_DIR)
        print("Extraction complete.")
    else:
        print("Error: Zip file not found!")
    
    # 3. Clean and Prepare
    print("\nStep 3: Preprocessing data...")
    import pandas as pd
    PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    metadata_csv = RAW_DIR / "OpenBind_EV-A71_2A" / "EV-A71_2A_metadata.csv"
    if not metadata_csv.exists():
        print(f"Error: {metadata_csv} not found!")
        return
        
    df = pd.read_csv(metadata_csv)
    
    # Rename columns to standard names
    df = df.rename(columns={
        "smiles": "SMILES",
        "experimental_pKD": "pKD",
        "complex_name": "Complex"
    })
    
    # Filter out missing affinities
    df_clean = df.dropna(subset=['pKD']).copy()
    
    # Check 601 entries
    print(f"Original dataset size: {len(df)}")
    print(f"Cleaned dataset size: {len(df_clean)}")
    
    master_path = PROCESSED_DIR / "master.csv"
    df_clean.to_csv(master_path, index=False)
    print(f"Saved processed data to {master_path}")
    
    print("\nDone! Data fetching and preprocessing complete.")

if __name__ == "__main__":
    main()
