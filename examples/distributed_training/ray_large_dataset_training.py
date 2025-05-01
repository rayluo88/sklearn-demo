#!/usr/bin/env python
"""
Distributed training example using Ray with a larger dataset.

This script demonstrates how to:
1. Load a large dataset in chunks
2. Distribute the training across multiple Ray workers
3. Create an ensemble model from the distributed training
4. Track experiments with MLflow
"""
import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
import ray
import mlflow

# Add parent directory to path to import download_datasets
sys.path.append(str(Path(__file__).parent.parent))
from large_datasets.download_datasets import download_nyc_taxi_data

# Directory where data is stored
DATA_DIR = Path(__file__).parent.parent / "large_datasets" / "data"

def load_taxi_data_sample(sample_size=None):
    """
    Load the NYC taxi dataset sample.
    
    Args:
        sample_size: Number of rows to sample (None for all)
        
    Returns:
        X_train, X_test, y_train, y_test: Split datasets
    """
    # Ensure data directory exists
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Download the dataset if not already present
    parquet_path = DATA_DIR / "yellow_tripdata_2019-01.parquet"
    if not os.path.exists(parquet_path):
        print("Downloading NYC Taxi dataset...")
        download_nyc_taxi_data()
    
    print(f"Loading data from {parquet_path}...")
    
    # Create a sample task - predict whether tip was > 20% of fare
    if sample_size:
        df = pd.read_parquet(parquet_path, engine='pyarrow').sample(sample_size)
    else:
        df = pd.read_parquet(parquet_path, engine='pyarrow')
    
    print(f"Dataset loaded: {len(df)} rows")
    
    # Clean data
    df = df.dropna(subset=['fare_amount', 'tip_amount', 'passenger_count'])
    
    # Filter to valid fare and tip amounts
    df = df[(df['fare_amount'] > 0) & (df['tip_amount'] >= 0)]
    
    # Create target: 1 if tip > 20% of fare, 0 otherwise
    df['tip_percent'] = (df['tip_amount'] / df['fare_amount']) * 100
    df['high_tip'] = (df['tip_percent'] > 20).astype(int)
    
    # Select features
    features = [
        'fare_amount', 'passenger_count', 'trip_distance',
        'PULocationID', 'DOLocationID', 'payment_type'
    ]
    
    # Convert timestamps to hour features
    df['pickup_hour'] = pd.to_datetime(df['tpep_pickup_datetime']).dt.hour
    df['dropoff_hour'] = pd.to_datetime(df['tpep_dropoff_datetime']).dt.hour
    features.extend(['pickup_hour', 'dropoff_hour'])
    
    # One-hot encode payment_type
    if 'payment_type' in features:
        dummies = pd.get_dummies(df['payment_type'], prefix='payment')
        df = pd.concat([df, dummies], axis=1)
        features.remove('payment_type')
        features.extend(dummies.columns.tolist())
    
    # Split data
    X = df[features]
    y = df['high_tip']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    
    # Scale numeric features
    numeric_features = ['fare_amount', 'passenger_count', 'trip_distance']
    scaler = StandardScaler()
    X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
    X_test[numeric_features] = scaler.transform(X_test[numeric_features])
    
    return X_train, X_test, y_train, y_test, features

@ray.remote
def train_model_partition(X_partition, y_partition, params):
    """
    Train a model on a partition of the data.
    This function runs as a Ray remote task.
    
    Args:
        X_partition: Features for this partition
        y_partition: Labels for this partition
        params: Model hyperparameters
        
    Returns:
        Trained model
    """
    # Create and train the model on this partition
    model = RandomForestClassifier(**params)
    model.fit(X_partition, y_partition)
    return model

def train_distributed(X_train, y_train, n_partitions=4, params=None):
    """
    Train models in a distributed fashion using Ray.
    
    Args:
        X_train: Training features
        y_train: Training labels
        n_partitions: Number of partitions/workers
        params: Model hyperparameters
        
    Returns:
        ensemble_model: Ensemble of models trained on different partitions
        training_time: Time taken for training
    """
    if params is None:
        params = {'n_estimators': 50, 'random_state': 42}
    
    start_time = time.time()
    
    # Split the data into partitions
    partition_size = len(X_train) // n_partitions
    X_partitions = [X_train.iloc[i:i+partition_size] for i in range(0, len(X_train), partition_size)]
    y_partitions = [y_train.iloc[i:i+partition_size] for i in range(0, len(y_train), partition_size)]
    
    # Ensure we don't have more partitions than intended
    X_partitions = X_partitions[:n_partitions]
    y_partitions = y_partitions[:n_partitions]
    
    # Train models in parallel using Ray
    print(f"Training {n_partitions} distributed models...")
    model_futures = [
        train_model_partition.remote(X_partitions[i], y_partitions[i], params) 
        for i in range(len(X_partitions))
    ]
    
    # Wait for all models to finish training
    distributed_models = ray.get(model_futures)
    
    # Create a voting ensemble from the distributed models
    ensemble_model = VotingClassifier(
        estimators=[(f'model_{i}', model) for i, model in enumerate(distributed_models)],
        voting='soft'  # Use probability predictions for voting
    )
    
    # Fit the ensemble on a small portion of data to enable prediction
    # (VotingClassifier needs fit before predict)
    ensemble_model.fit(X_train.iloc[:100], y_train.iloc[:100])
    
    training_time = time.time() - start_time
    
    return ensemble_model, distributed_models, training_time

def train_regular(X_train, y_train, params=None):
    """
    Train a single model on the full dataset.
    
    Args:
        X_train: Training features
        y_train: Training labels
        params: Model hyperparameters
        
    Returns:
        model: Trained model
        training_time: Time taken for training
    """
    if params is None:
        params = {'n_estimators': 200, 'random_state': 42}
    
    print("Training single model on full dataset...")
    start_time = time.time()
    
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    
    return model, training_time

def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate a model and print metrics.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        model_name: Name for reporting
        
    Returns:
        accuracy: Model accuracy
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n{model_name} Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))
    
    return accuracy

def main():
    parser = argparse.ArgumentParser(description="Distributed training demo with Ray on NYC Taxi dataset")
    parser.add_argument("--partitions", type=int, default=4,
                      help="Number of partitions/workers for distributed training")
    parser.add_argument("--sample", type=int, default=100000,
                      help="Number of samples to use from the dataset")
    parser.add_argument("--address", type=str, default=None,
                      help="Address of Ray cluster (if not running locally)")
    args = parser.parse_args()
    
    # Initialize Ray - either connect to a cluster or run locally
    if args.address:
        ray.init(address=args.address)
        print(f"Connected to Ray cluster at {args.address}")
    else:
        ray.init()
        print("Running Ray in local mode")
    
    print(f"Available Ray resources: {ray.available_resources()}")
    
    # Start MLflow run
    with mlflow.start_run(run_name="distributed-taxi-prediction"):
        # Log parameters
        mlflow.log_param("partitions", args.partitions)
        mlflow.log_param("sample_size", args.sample)
        mlflow.log_param("ray_address", args.address if args.address else "local")
        
        # Load and preprocess data
        X_train, X_test, y_train, y_test, features = load_taxi_data_sample(args.sample)
        print(f"Training data shape: {X_train.shape}")
        
        # Train distributed model
        ensemble_model, distributed_models, dist_time = train_distributed(
            X_train, y_train, n_partitions=args.partitions
        )
        
        # Train regular model with equivalent compute
        regular_model, reg_time = train_regular(X_train, y_train)
        
        # Evaluate both models
        dist_accuracy = evaluate_model(ensemble_model, X_test, y_test, "Distributed Ensemble Model")
        reg_accuracy = evaluate_model(regular_model, X_test, y_test, "Regular Model")
        
        # Log metrics
        mlflow.log_metric("distributed_training_time", dist_time)
        mlflow.log_metric("regular_training_time", reg_time)
        mlflow.log_metric("distributed_accuracy", dist_accuracy)
        mlflow.log_metric("regular_accuracy", reg_accuracy)
        mlflow.log_metric("speedup_factor", reg_time / dist_time)
        
        # Log models
        mlflow.sklearn.log_model(regular_model, "regular_model")
        
        # We can't directly log the ensemble with its estimators due to Ray objects
        # So let's save the individual models
        for i, model in enumerate(distributed_models):
            mlflow.sklearn.log_model(model, f"distributed_model_{i}")
        
        # Extract and log feature importance from models
        feature_imp = pd.DataFrame(
            regular_model.feature_importances_,
            index=features,
            columns=['importance']
        ).sort_values('importance', ascending=False)
        
        feature_imp_path = "feature_importance.csv"
        feature_imp.to_csv(feature_imp_path)
        mlflow.log_artifact(feature_imp_path)
        os.remove(feature_imp_path)  # Clean up local file
        
        # Print summary
        print("\nPerformance Summary:")
        print(f"{'Model':<25} {'Time (s)':<12} {'Accuracy':<10}")
        print("-" * 50)
        print(f"{'Distributed Model':<25} {dist_time:<12.2f} {dist_accuracy:<10.4f}")
        print(f"{'Regular Model':<25} {reg_time:<12.2f} {reg_accuracy:<10.4f}")
        print(f"Speedup factor: {reg_time/dist_time:.2f}x")
        
        print("\nMLflow tracking:")
        print(f"Run ID: {mlflow.active_run().info.run_id}")
        print("View results with: mlflow ui")
    
    # Shut down Ray
    ray.shutdown()
    print("Ray has been shut down")

if __name__ == "__main__":
    main() 