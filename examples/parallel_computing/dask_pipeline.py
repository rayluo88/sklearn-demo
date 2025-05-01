"""
Example of enhancing the ML pipeline with Dask for parallel computing
"""
import numpy as np
import mlflow
import mlflow.sklearn
import os
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report
from dask.distributed import Client, LocalCluster
from dask_ml.model_selection import train_test_split
from dask_ml.wrappers import ParallelPostFit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import dask.array as da

# Constants
N_NEIGHBORS = 3
TEST_SIZE = 0.2
RANDOM_STATE = 42

def run_dask_pipeline():
    """Runs ML pipeline with Dask parallel computing."""
    
    # Start Dask client - adjust the number of workers based on your system
    # For real distributed computing, you'd configure an external cluster
    # and connect to it instead of creating a local one
    cluster = LocalCluster(n_workers=2, threads_per_worker=1)
    client = Client(cluster)
    print(f"Dask dashboard available at: {client.dashboard_link}")
    
    # Start MLflow run
    with mlflow.start_run(run_name="dask-parallel-pipeline"):
        mlflow.set_tag("computing_framework", "dask")
        
        # Log parameters
        mlflow.log_param("n_neighbors", N_NEIGHBORS)
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("dask_workers", 2)
        
        # 1. Load Data
        print("\nLoading Iris dataset...")
        iris = load_iris()
        
        # Convert to Dask arrays for distributed computing
        # In a real-world scenario with large datasets, you'd load data directly as dask arrays
        X = da.from_array(iris.data, chunks="auto")
        y = da.from_array(iris.target, chunks="auto")
        
        print(f"Dataset loaded and converted to Dask arrays")
        print(f"X shape: {X.shape}, chunks: {X.chunks}")
        
        # 2. Split data
        print("\nSplitting data using Dask...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        
        # For scaling, convert back to numpy arrays
        # (in a real scenario with large data, you'd use dask-ml's scalers)
        X_train_local = X_train.compute()
        X_test_local = X_test.compute()
        y_train_local = y_train.compute()
        y_test_local = y_test.compute()
        
        # 3. Scale features
        print("\nScaling features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_local)
        X_test_scaled = scaler.transform(X_test_local)
        
        # 4. Initialize model with parallel prediction capability
        print("\nTraining model with parallel prediction capability...")
        base_model = KNeighborsClassifier(n_neighbors=N_NEIGHBORS)
        model = ParallelPostFit(base_model)
        
        # Train the model
        model.fit(X_train_scaled, y_train_local)
        print("Model training complete")
        
        # 5. Make predictions
        print("\nMaking predictions in parallel...")
        y_pred = model.predict(X_test_scaled)
        
        # 6. Evaluate model
        accuracy = accuracy_score(y_test_local, y_pred)
        report = classification_report(y_test_local, y_pred, target_names=iris.target_names)
        
        print(f"\nAccuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(report)
        
        # Log metrics and model
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, "dask-knn-model")
        
        # Log classification report as text
        report_path = "classification_report.txt"
        with open(report_path, "w") as f:
            f.write(report)
        mlflow.log_artifact(report_path)
        os.remove(report_path)
        
        print("\nModel and metrics logged to MLflow")
        
    # Shut down Dask client
    client.close()
    cluster.close()
    print("\nDask cluster shut down")

if __name__ == "__main__":
    run_dask_pipeline() 