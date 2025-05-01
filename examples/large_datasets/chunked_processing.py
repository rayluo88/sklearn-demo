#!/usr/bin/env python
"""
Example showing how to process large CSV files in chunks efficiently.

This script demonstrates processing the NYC Taxi data without 
loading the entire dataset into memory at once.
"""
import os
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import mlflow

# Directory where data is stored
DATA_DIR = Path(__file__).parent / "data"

def chunked_csv_processing(file_path, chunk_size=10000):
    """
    Process a large CSV file in chunks with pandas.
    
    Args:
        file_path: Path to the CSV file
        chunk_size: Number of rows to process in each chunk
    
    Returns:
        Dictionary with aggregated results
    """
    # Initialize statistical accumulators
    total_rows = 0
    sum_fare = 0
    sum_squared_fare = 0
    fare_counts = {}  # Count of fares by hour
    
    # Time the processing
    start_time = time.time()
    
    print(f"Processing {file_path} in chunks of {chunk_size} rows...")
    
    # Create a chunked reader for the CSV file
    # This is memory-efficient as it only loads `chunk_size` rows at a time
    chunk_iter = pd.read_csv(file_path, chunksize=chunk_size)
    
    # Process each chunk
    for i, chunk in enumerate(chunk_iter):
        # Track total rows processed
        rows_in_chunk = len(chunk)
        total_rows += rows_in_chunk
        
        # Extract the pickup hour (assuming tpep_pickup_datetime exists)
        if 'tpep_pickup_datetime' in chunk.columns:
            # Convert to datetime if needed
            chunk['pickup_hour'] = pd.to_datetime(chunk['tpep_pickup_datetime']).dt.hour
            
            # Aggregate fares by hour
            hour_counts = chunk.groupby('pickup_hour')['fare_amount'].sum().to_dict()
            
            # Update the overall counts
            for hour, fare in hour_counts.items():
                if hour in fare_counts:
                    fare_counts[hour] += fare
                else:
                    fare_counts[hour] = fare
        
        # Calculate running statistics for fare amount
        if 'fare_amount' in chunk.columns:
            sum_fare += chunk['fare_amount'].sum()
            sum_squared_fare += (chunk['fare_amount'] ** 2).sum()
        
        # Progress update every 5 chunks
        if (i + 1) % 5 == 0:
            elapsed = time.time() - start_time
            rows_per_sec = total_rows / elapsed if elapsed > 0 else 0
            print(f"Processed {total_rows} rows in {elapsed:.2f} seconds ({rows_per_sec:.1f} rows/sec)")
    
    # Calculate final statistics
    if total_rows > 0:
        avg_fare = sum_fare / total_rows
        # Standard deviation formula: sqrt(E[X²] - E[X]²)
        std_fare = np.sqrt((sum_squared_fare / total_rows) - (avg_fare ** 2))
    else:
        avg_fare = std_fare = 0
    
    elapsed = time.time() - start_time
    print(f"Finished processing {total_rows} rows in {elapsed:.2f} seconds")
    
    # Return the aggregated results
    return {
        'total_rows': total_rows,
        'avg_fare': avg_fare,
        'std_fare': std_fare,
        'fare_by_hour': fare_counts,
        'processing_time': elapsed
    }

def chunked_transformation(file_path, chunk_size=10000):
    """
    Demonstrate applying scikit-learn transformations to chunks of data.
    
    Args:
        file_path: Path to the CSV file
        chunk_size: Number of rows to process in each chunk
    
    Returns:
        Path to the transformed data file
    """
    # Output file for transformed data
    output_path = Path(file_path).parent / f"{Path(file_path).stem}_transformed.csv"
    
    # Create a StandardScaler (must fit on all data first)
    print("First pass: fitting the scaler...")
    
    # Initialize accumulators for the fit
    sum_values = 0
    sum_squares = 0
    n_samples = 0
    
    # Target columns for scaling
    num_cols = ['fare_amount', 'trip_distance']
    
    # First pass: collect statistics
    chunk_iter = pd.read_csv(file_path, chunksize=chunk_size)
    for chunk in chunk_iter:
        if all(col in chunk.columns for col in num_cols):
            chunk_features = chunk[num_cols].values
            n_chunk_samples = chunk_features.shape[0]
            
            # Update accumulators
            sum_values += np.sum(chunk_features, axis=0)
            sum_squares += np.sum(np.square(chunk_features), axis=0)
            n_samples += n_chunk_samples
    
    # Calculate mean and std manually
    means = sum_values / n_samples
    stds = np.sqrt(sum_squares / n_samples - np.square(means))
    
    print(f"Fitted stats - Means: {means}, Stds: {stds}")
    
    # Second pass: apply the transformation
    print(f"Second pass: transforming the data and writing to {output_path}...")
    
    # Initialize output file
    first_chunk = True
    
    # Process each chunk
    chunk_iter = pd.read_csv(file_path, chunksize=chunk_size)
    for chunk in chunk_iter:
        # Apply the transformation manually
        if all(col in chunk.columns for col in num_cols):
            # Standardize the numeric columns
            for i, col in enumerate(num_cols):
                chunk[f"{col}_scaled"] = (chunk[col] - means[i]) / stds[i]
            
            # Write to output file
            if first_chunk:
                chunk.to_csv(output_path, index=False, mode='w')
                first_chunk = False
            else:
                chunk.to_csv(output_path, index=False, mode='a', header=False)
    
    print(f"Transformation complete. Results saved to {output_path}")
    return output_path

def plot_fare_by_hour(fare_by_hour):
    """
    Plot fare amounts by hour.
    
    Args:
        fare_by_hour: Dictionary with hour -> fare amount mapping
    """
    # Sort by hour
    hours = sorted(fare_by_hour.keys())
    fares = [fare_by_hour[h] for h in hours]
    
    plt.figure(figsize=(10, 6))
    plt.bar(hours, fares)
    plt.xlabel('Hour of Day')
    plt.ylabel('Total Fare Amount')
    plt.title('NYC Taxi Fares by Hour of Day')
    plt.xticks(range(0, 24))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save the plot
    plot_path = os.path.join(DATA_DIR, 'fares_by_hour.png')
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Plot saved to {plot_path}")
    return plot_path

def main():
    parser = argparse.ArgumentParser(description="Demo of processing large CSV files in chunks")
    parser.add_argument("--file", type=str, default=None, 
                      help="Path to CSV file (default: taxi_sample.csv in data directory)")
    parser.add_argument("--chunk-size", type=int, default=10000,
                      help="Chunk size for processing (default: 10000)")
    args = parser.parse_args()
    
    # Determine which file to process
    if args.file:
        file_path = args.file
    else:
        file_path = os.path.join(DATA_DIR, "taxi_sample.csv")
        if not os.path.exists(file_path):
            print(f"Sample file {file_path} not found.")
            print("Please download the sample data first by running:")
            print("python examples/large_datasets/download_datasets.py")
            return
    
    # Start MLflow run
    with mlflow.start_run(run_name="chunked-csv-processing"):
        # Log parameters
        mlflow.log_param("file", file_path)
        mlflow.log_param("chunk_size", args.chunk_size)
        
        # Process the file in chunks
        results = chunked_csv_processing(file_path, args.chunk_size)
        
        # Log metrics
        mlflow.log_metric("total_rows", results['total_rows'])
        mlflow.log_metric("avg_fare", results['avg_fare'])
        mlflow.log_metric("std_fare", results['std_fare'])
        mlflow.log_metric("processing_time", results['processing_time'])
        
        # Create and log the plot
        if results['fare_by_hour']:
            plot_path = plot_fare_by_hour(results['fare_by_hour'])
            mlflow.log_artifact(plot_path)
        
        # Apply scikit-learn transformation to chunks
        transformed_path = chunked_transformation(file_path, args.chunk_size)
        mlflow.log_artifact(transformed_path)
        
        print("\nResults:")
        print(f"Total rows processed: {results['total_rows']}")
        print(f"Average fare: ${results['avg_fare']:.2f}")
        print(f"Standard deviation of fares: ${results['std_fare']:.2f}")
        print(f"Processing time: {results['processing_time']:.2f} seconds")
        
        print("\nExperiment tracking:")
        print(f"MLflow run ID: {mlflow.active_run().info.run_id}")
        print(f"Track this run in the MLflow UI with: mlflow ui")

if __name__ == "__main__":
    main() 