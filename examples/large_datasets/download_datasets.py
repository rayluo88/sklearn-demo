#!/usr/bin/env python
"""
Script to download and prepare sample large datasets for the examples.
"""
import os
import argparse
import time
import gzip
import shutil
from pathlib import Path
import urllib.request
import pandas as pd

# Create data directory inside the examples folder
DATA_DIR = Path(__file__).parent / "data"
os.makedirs(DATA_DIR, exist_ok=True)

def download_file(url, target_path, display_name=None):
    """Download a file with progress tracking."""
    if display_name is None:
        display_name = os.path.basename(target_path)
    
    if os.path.exists(target_path):
        print(f"{display_name} already exists. Skipping download.")
        return
    
    print(f"Downloading {display_name}...")
    start_time = time.time()
    
    # Use urllib to download with a simple progress bar
    def report_progress(block_num, block_size, total_size):
        read_so_far = block_num * block_size
        if total_size > 0:
            percent = read_so_far * 100 / total_size
            elapsed = time.time() - start_time
            print(f"\rProgress: {percent:.1f}% ({read_so_far/1024/1024:.1f} MB) - "
                  f"Elapsed: {elapsed:.1f}s", end="", flush=True)
    
    try:
        urllib.request.urlretrieve(url, target_path, reporthook=report_progress)
        elapsed = time.time() - start_time
        print(f"\nDownload completed in {elapsed:.1f} seconds.")
    except Exception as e:
        print(f"\nError downloading file: {e}")
        if os.path.exists(target_path):
            os.remove(target_path)
        raise

def extract_gz(gz_path, output_path):
    """Extract a gzip file."""
    if os.path.exists(output_path):
        print(f"{os.path.basename(output_path)} already exists. Skipping extraction.")
        return
    
    print(f"Extracting {os.path.basename(gz_path)}...")
    start_time = time.time()
    
    with gzip.open(gz_path, 'rb') as f_in:
        with open(output_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    elapsed = time.time() - start_time
    print(f"Extraction completed in {elapsed:.1f} seconds.")

def create_sample(csv_path, sample_path, nrows=100000):
    """Create a smaller sample from a large CSV file."""
    if os.path.exists(sample_path):
        print(f"{os.path.basename(sample_path)} already exists. Skipping sample creation.")
        return
    
    print(f"Creating {nrows} row sample from {os.path.basename(csv_path)}...")
    start_time = time.time()
    
    # Read the first nrows
    df = pd.read_csv(csv_path, nrows=nrows)
    df.to_csv(sample_path, index=False)
    
    elapsed = time.time() - start_time
    print(f"Sample creation completed in {elapsed:.1f} seconds.")
    print(f"Sample size: {os.path.getsize(sample_path)/1024/1024:.1f} MB")

def download_nyc_taxi_data():
    """Download and prepare NYC Yellow Taxi Trip Data."""
    # Source URL (Jan 2019 data)
    url = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2019-01.parquet"
    parquet_path = os.path.join(DATA_DIR, "yellow_tripdata_2019-01.parquet")
    
    # Download parquet file
    download_file(url, parquet_path, "NYC Taxi Data (Parquet)")
    
    # Create a sample in CSV format for the conversion example
    csv_sample_path = os.path.join(DATA_DIR, "taxi_sample.csv")
    if not os.path.exists(csv_sample_path):
        print("Converting sample to CSV format...")
        df = pd.read_parquet(parquet_path)
        df.head(100000).to_csv(csv_sample_path, index=False)
        print(f"CSV sample created: {os.path.getsize(csv_sample_path)/1024/1024:.1f} MB")

def download_airline_data():
    """Download and prepare Airline On-Time Performance data."""
    # Source URL
    url = "https://transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_2019_1.zip"
    zip_path = os.path.join(DATA_DIR, "airline_data_2019_01.zip")
    csv_path = os.path.join(DATA_DIR, "airline_data_2019_01.csv")
    
    # Download zip file
    download_file(url, zip_path, "Airline Performance Data (ZIP)")
    
    # Extract the CSV (if it doesn't already exist)
    if not os.path.exists(csv_path):
        print("Extracting zip file...")
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get the CSV file name inside the zip
            csv_name = [f for f in zip_ref.namelist() if f.endswith('.csv')][0]
            zip_ref.extract(csv_name, DATA_DIR)
            # Rename to a consistent name
            os.rename(os.path.join(DATA_DIR, csv_name), csv_path)
        print(f"CSV extracted: {os.path.getsize(csv_path)/1024/1024:.1f} MB")

def main():
    parser = argparse.ArgumentParser(description="Download datasets for large data processing examples")
    parser.add_argument("--dataset", choices=["nyc_taxi", "airline", "all"], default="all",
                      help="Which dataset to download (default: all)")
    args = parser.parse_args()
    
    if args.dataset in ["nyc_taxi", "all"]:
        download_nyc_taxi_data()
    
    if args.dataset in ["airline", "all"]:
        download_airline_data()
    
    print("\nAll downloads completed!")
    print(f"Data directory: {DATA_DIR}")

if __name__ == "__main__":
    main() 