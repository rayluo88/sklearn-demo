import os
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
from datetime import datetime

# Import Feast
from feast import FeatureStore
from pathlib import Path

# Constants
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "Iris Classification with Feast"
MODEL_NAME = "iris-classifier"
DATA_PATH = "data/iris.csv" # Still used to get entity dataframe source
FEAST_REPO_PATH = "feature_repository/"
FEAST_SOURCE_PARQUET_PATH = "data/iris_feast_source.parquet" # Source for entity IDs and timestamps


def get_training_data_from_feast(store: FeatureStore):
    """
    Retrieves training data (features and target) from Feast.
    Assumes prepare_feast_data.py has been run and iris_feast_source.parquet exists.
    """
    print("Retrieving training data from Feast...")

    # Load the entity data (iris_id, event_timestamp, and target_class for creating labels)
    # from the parquet file that was used as the source for Feast.
    # This file contains all necessary information for constructing the entity_df.
    try:
        entity_source_df = pd.read_parquet(FEAST_SOURCE_PARQUET_PATH)
        print(f"Successfully loaded entity source data from {FEAST_SOURCE_PARQUET_PATH}")
    except Exception as e:
        print(f"Error loading {FEAST_SOURCE_PARQUET_PATH}: {e}")
        print("Please ensure 'prepare_feast_data.py' has been run successfully.")
        raise

    # The entity_df needs 'iris_id' and 'event_timestamp'
    # We also need 'target_class' to create our y labels after fetching features.
    entity_df_for_feast = entity_source_df[['iris_id', 'event_timestamp', 'target_class']]

    # Fetch historical features
    # We defined 'target_class' also as a feature in the FeatureView for simplicity
    # to retrieve it along with other features.
    training_df = store.get_historical_features(
        entity_df=entity_df_for_feast,
        features=[
            "iris_features:sepal_length",
            "iris_features:sepal_width",
            "iris_features:petal_length",
            "iris_features:petal_width",
            "iris_features:target_class" # Retrieve target_class as a feature
        ],
    ).to_df()
    
    print("Training data retrieved from Feast:")
    print(training_df.head())

    # Separate features (X) and target (y)
    # The target_class was retrieved as part of the features
    X = training_df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    y = training_df['target_class']
    
    return X, y, training_df # Return full df for MLflow logging if needed


def run_pipeline():
    """Main function to run the ML pipeline."""
    print(f"MLflow Tracking URI: {MLFLOW_TRACKING_URI}")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    # Check if experiment exists, if not create it
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        print(f"Experiment '{EXPERIMENT_NAME}' not found. Creating new experiment.")
        mlflow.create_experiment(EXPERIMENT_NAME)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=f"Feast_Pipeline_Run_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
        print(f"MLflow Run ID: {run.info.run_id}")
        mlflow.log_param("data_source", "Feast Feature Store")
        mlflow.log_param("feast_repo_path", FEAST_REPO_PATH)

        # Initialize Feature Store
        # The path should be relative to the current working directory where simple_ml_pipeline.py is run
        # or an absolute path.
        store = FeatureStore(repo_path=str(Path(FEAST_REPO_PATH).resolve()))

        # 1. Get training data from Feast
        try:
            X, y, training_data_df = get_training_data_from_feast(store)
            # Log the parquet file path used as source for entities for traceability
            mlflow.log_param("feast_entity_source_parquet", FEAST_SOURCE_PARQUET_PATH)
        except Exception as e:
            print(f"Failed to get training data from Feast: {e}")
            mlflow.log_param("data_retrieval_status", "failed")
            mlflow.log_param("data_retrieval_error", str(e))
            return

        # Log dataset details (optional, could log a sample or hash)
        # For demo, let's log the shape
        mlflow.log_param("training_data_shape", training_data_df.shape)
        
        # 2. Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)

        # 3. Model Training (Random Forest with GridSearchCV)
        print("Training RandomForestClassifier with GridSearchCV...")
        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        print(f"Best parameters: {grid_search.best_params_}")
        mlflow.log_params(grid_search.best_params_)
        
        # 4. Model Evaluation
        print("Evaluating model...")
        y_pred = best_model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision_weighted", precision)
        mlflow.log_metric("recall_weighted", recall)
        mlflow.log_metric("f1_weighted", f1)
        
        # Cross-validation scores
        cv_scores = cross_val_score(best_model, X, y, cv=5)
        mlflow.log_metric("cv_accuracy_mean", cv_scores.mean())
        mlflow.log_metric("cv_accuracy_std", cv_scores.std())
        print(f"  Cross-validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        # 5. Log model with MLflow
        print("Logging model to MLflow...")
        # Define a signature for the model
        from mlflow.models.signature import infer_signature
        signature = infer_signature(X_train, best_model.predict(X_train))
        
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path=MODEL_NAME, # This will be a subdirectory under the run's artifact root
            signature=signature,
            registered_model_name=MODEL_NAME # Register the model
        )
        print(f"Model '{MODEL_NAME}' logged and registered.")
        
        # Log a sample of the training data (optional, for reproducibility)
        # training_data_df.head().to_csv("training_sample.csv", index=False)
        # mlflow.log_artifact("training_sample.csv")

    print("ML Pipeline with Feast completed successfully.")
    return best_model

if __name__ == "__main__":
    # First, ensure that the data for Feast is prepared.
    # You would typically run `python prepare_feast_data.py` once before this.
    # And then `feast apply` and `feast materialize` from the `feature_repository` directory.
    
    # For this script to run, it's assumed `feast apply` has been run in `feature_repository`
    # and `iris_feast_source.parquet` exists in `data/`.
    
    print("Starting ML pipeline...")
    run_pipeline()
    print("Pipeline execution finished.") 