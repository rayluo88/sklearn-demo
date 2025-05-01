import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.sklearn
import os # For handling artifact paths

# Define parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_NEIGHBORS = 5  # default value for KNN, 3
SCALER_NAME = "StandardScaler"

def run_pipeline():
    """Runs a simple machine learning pipeline with MLflow tracking."""

    # Start an MLflow run
    with mlflow.start_run():
        mlflow.set_tag("ml_task", "iris_classification")
        print("--- MLflow Run Started ---")

        # --- Log Parameters ---
        print("\nLogging parameters to MLflow...")
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("n_neighbors", N_NEIGHBORS)
        mlflow.log_param("scaler", SCALER_NAME)
        print("Parameters logged.")

        # 1. Load Data
        print("\nLoading Iris dataset...")
        iris = load_iris()
        X, y = iris.data, iris.target
        print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"Target classes: {iris.target_names}")

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
        report = classification_report(y_test, y_pred, target_names=iris.target_names)

        print(f"\nAccuracy on Test Set: {accuracy:.4f}")
        print("\nClassification Report:")
        print(report)

        # --- Log Metrics ---
        print("\nLogging metrics to MLflow...")
        mlflow.log_metric("accuracy", accuracy)
        # You could parse the report string to log precision/recall per class if needed
        print("Metrics logged.")

        # --- Log Artifacts (Model and Report) ---
        print("\nLogging model and report artifact to MLflow...")
        # Log the scikit-learn model
        mlflow.sklearn.log_model(model, "knn-model")

        # Log the classification report as a text file artifact
        report_filename = "classification_report.txt"
        with open(report_filename, 'w') as f:
            f.write(report)
        mlflow.log_artifact(report_filename)
        # Clean up the temp file (MLflow copies it to its artifact store)
        try:
            os.remove(report_filename)
        except OSError as e:
            print(f"Error removing temporary report file: {e}")

        print("Model and report logged as artifacts.")
        print("\n--- MLflow Run Finished ---")

if __name__ == "__main__":
    run_pipeline() 