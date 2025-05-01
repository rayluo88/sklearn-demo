#!/usr/bin/env python
"""
Example of distributed training using Ray.

This script demonstrates how to train models in a distributed fashion using Ray.
Key features demonstrated:
- Parallelized model training across multiple workers
- Ensemble combination of distributed models
- Comparison with non-distributed training
- Integration with MLflow for experiment tracking
"""
import os
import time
import argparse
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import ray
import mlflow

def load_and_preprocess_data(random_state=42):
    """
    Load and preprocess a dataset for demonstration.
    
    Returns:
        X_train, X_test, y_train, y_test: Split datasets
    """
    # Load breast cancer dataset (larger than Iris but still manageable)
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    # Scale the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test, data.feature_names

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
    X_partitions = [X_train[i:i+partition_size] for i in range(0, len(X_train), partition_size)]
    y_partitions = [y_train[i:i+partition_size] for i in range(0, len(y_train), partition_size)]
    
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
    ensemble_model.fit(X_train[:100], y_train[:100])
    
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
    parser = argparse.ArgumentParser(description="Distributed training demo with Ray")
    parser.add_argument("--partitions", type=int, default=4,
                      help="Number of partitions/workers for distributed training")
    parser.add_argument("--local", action="store_true",
                      help="Run Ray in local mode (default)")
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
    with mlflow.start_run(run_name="distributed-training-demo"):
        # Log parameters
        mlflow.log_param("partitions", args.partitions)
        mlflow.log_param("ray_address", args.address if args.address else "local")
        
        # Load and preprocess data
        X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_data()
        print(f"Training data shape: {X_train.shape}")
        
        # Train distributed model
        ensemble_model, distributed_models, dist_time = train_distributed(
            X_train, y_train, n_partitions=args.partitions
        )
        
        # Train regular model with equivalent compute
        # For fair comparison, regular model has n_estimators = partitions * partition_n_estimators
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
            index=feature_names,
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