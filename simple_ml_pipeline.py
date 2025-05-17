import numpy as np
# from sklearn.datasets import load_iris # No longer needed
import pandas as pd # For loading CSV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.sklearn
import os # For handling artifact paths
import hashlib # For data hashing

# Define parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_NEIGHBORS = 5  # default value for KNN
SCALER_NAME = "StandardScaler"
DATA_PATH = "data/iris.csv" # Path to DVC-tracked data

def run_pipeline():
    """Runs a simple machine learning pipeline with MLflow tracking, using DVC-tracked data."""

    # Start an MLflow run
    with mlflow.start_run():
        mlflow.set_tag("ml_task", "iris_classification_dvc")
        mlflow.set_tag("data_source", DATA_PATH)
        print("--- MLflow Run Started (with DVC-tracked data) ---")

        # --- Log Parameters ---
        print("\nLogging parameters to MLflow...")
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("n_neighbors", N_NEIGHBORS)
        mlflow.log_param("scaler", SCALER_NAME)
        mlflow.log_param("data_path", DATA_PATH)
        print("Parameters logged.")

        # 1. Load Data from DVC-tracked CSV
        print(f"\nLoading dataset from {DATA_PATH}...")
        try:
            df = pd.read_csv(DATA_PATH)
        except FileNotFoundError:
            print(f"ERROR: Data file not found at {DATA_PATH}.")
            print("Please ensure you have run 'dvc pull' if the data is not present locally.")
            mlflow.set_tag("run_status", "failed_data_missing")
            return

        feature_columns = [col for col in df.columns if col != 'target']
        X = df[feature_columns].values
        y = df['target'].values
        
        # Define Iris target names (as they are not directly in the CSV like from sklearn.datasets)
        # Order corresponds to target values 0, 1, 2
        iris_target_names = np.array(['setosa', 'versicolor', 'virginica']) 
        
        print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"Feature names: {feature_columns}")
        print(f"Target classes (from mapping): {iris_target_names}")


        # --- Data Versioning (using hash of loaded data) ---
        print("\nVersioning dataset (based on loaded CSV)...")
        # Create a hash of the data
        data_hash = hashlib.sha256(X.tobytes() + y.tobytes()).hexdigest()
        mlflow.log_param("dataset_content_hash", data_hash) # Renamed to distinguish from file hash DVC uses
        mlflow.set_tag("dataset_content_sha256", data_hash)
        print(f"Dataset content hash (SHA256): {data_hash}")

        # Save dataset (as loaded) as an artifact
        dataset_filename = "iris_dataset_loaded.npz" # Changed name to reflect it's the loaded version
        np.savez_compressed(
            dataset_filename,
            X=X,
            y=y,
            feature_names=np.array(feature_columns), # Ensure it's an array for npz
            target_names=iris_target_names
        )
        mlflow.log_artifact(dataset_filename, artifact_path="dataset_loaded_from_csv")
        try:
            os.remove(dataset_filename)
        except OSError as e:
            print(f"Error removing temporary dataset file: {e}")
        print(f"Loaded dataset saved as MLflow artifact: dataset_loaded_from_csv/{dataset_filename}")
        # --- End Data Versioning ---

        # 2. Preprocessing: Split Data
        print("\nSplitting data into training and testing sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        print(f"Training set size: {X_train.shape[0]}")
        print(f"Testing set size: {X_test.shape[0]}")

        # 3. Preprocessing: Scale Features
        print("\nScaling features using StandardScaler...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        print("Features scaled.")

        # 4. Model Training
        print("\nTraining K-Nearest Neighbors model...")
        model = KNeighborsClassifier(n_neighbors=N_NEIGHBORS)
        model.fit(X_train_scaled, y_train)
        print("Model training complete.")

        # 5. Model Prediction
        print("\nMaking predictions on the test set...")
        y_pred = model.predict(X_test_scaled)

        # 6. Model Evaluation
        print("\nEvaluating model performance...")
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=iris_target_names) # Use defined target names

        print(f"\nAccuracy on Test Set: {accuracy:.4f}")
        print("\nClassification Report:")
        print(report)

        # --- Log Metrics ---
        print("\nLogging metrics to MLflow...")
        mlflow.log_metric("accuracy", accuracy)
        print("Metrics logged.")

        # --- Log Artifacts (Model and Report) ---
        print("\nLogging model and report artifact to MLflow...")
        mlflow.sklearn.log_model(model, "knn-model")

        report_filename = "classification_report.txt"
        with open(report_filename, 'w') as f:
            f.write(report)
        mlflow.log_artifact(report_filename)
        try:
            os.remove(report_filename)
        except OSError as e:
            print(f"Error removing temporary report file: {e}")

        print("Model and report logged as artifacts.")
        mlflow.set_tag("run_status", "success")
        print("\n--- MLflow Run Finished ---")

if __name__ == "__main__":
    run_pipeline() 