import os
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import wandb # Replaces mlflow
from datetime import datetime
import joblib # Added for saving the model

# Import Feast
from feast import FeatureStore
from pathlib import Path

# Constants
WANDB_PROJECT_NAME = "iris-feast-wandb-demo" # New: W&B project name
MODEL_NAME = "iris-classifier" # Reused for W&B artifact name
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
    
    return X, y, training_df # Return full df for logging if needed


def run_wandb_pipeline(): # Renamed function
    """Main function to run the ML pipeline with W&B tracking."""
    
    # Initialize W&B
    # Assumes `wandb login` has been run in the terminal.
    run = wandb.init(
        project=WANDB_PROJECT_NAME,
        name=f"WandB_Feast_Run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config={ # Initial configuration parameters
            "data_source": "Feast Feature Store",
            "feast_repo_path": FEAST_REPO_PATH,
            "model_type": "RandomForestClassifier"
        }
    )
    print(f"W&B Run ID: {run.id}, URL: {run.get_url()}")

    # Initialize Feature Store
    store = FeatureStore(repo_path=str(Path(FEAST_REPO_PATH).resolve()))

    # 1. Get training data from Feast
    try:
        X, y, training_data_df = get_training_data_from_feast(store)
        # Log the parquet file path used as source for entities for traceability
        wandb.config.update({"feast_entity_source_parquet": FEAST_SOURCE_PARQUET_PATH})
    except Exception as e:
        print(f"Failed to get training data from Feast: {e}")
        # Log error status to W&B summary (optional)
        wandb.summary["data_retrieval_status"] = "failed"
        wandb.summary["data_retrieval_error"] = str(e)
        run.finish() # Ensure W&B run is finished if an error occurs early
        return

    # Log dataset details
    wandb.config.update({"training_data_shape": training_data_df.shape})
    
    # 2. Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    wandb.config.update({"test_size": 0.2, "random_state": 42})

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
    wandb.config.update(grid_search.best_params_) # Log best parameters
    
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
    
    # Log metrics to W&B
    wandb.log({
        "accuracy": accuracy,
        "precision_weighted": precision,
        "recall_weighted": recall,
        "f1_weighted": f1
    })
    
    # Cross-validation scores
    cv_scores = cross_val_score(best_model, X, y, cv=5)
    wandb.log({
        "cv_accuracy_mean": cv_scores.mean(),
        "cv_accuracy_std": cv_scores.std()
    })
    print(f"  Cross-validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # 5. Log model with W&B
    print("Logging model to W&B...")
    # Save the model to a file first
    model_filename = f"{MODEL_NAME}.joblib"
    joblib.dump(best_model, model_filename)

    # Create a W&B artifact
    model_artifact = wandb.Artifact(
        MODEL_NAME, 
        type="model",
        description="RandomForestClassifier for Iris dataset, trained with Feast features.",
        metadata=dict(grid_search.best_params_) # Add best params to artifact metadata
    )
    model_artifact.add_file(model_filename) # Add the saved model file to the artifact

    # Log the artifact
    run.log_artifact(model_artifact) # Use run.log_artifact() as 'run' is in scope
    
    print(f"Model '{MODEL_NAME}' logged to W&B as an artifact: {model_filename}")
    
    # Log a sample of the training data (optional, for reproducibility, as a W&B Table)
    # try:
    #     sample_df_for_wandb = training_data_df.head()
    #     wandb_table = wandb.Table(dataframe=sample_df_for_wandb)
    #     wandb.log({"training_sample_table": wandb_table})
    #     print("Logged a sample of the training data as a W&B Table.")
    # except Exception as e:
    #     print(f"Could not log training sample as W&B Table: {e}")


    run.finish() # Explicitly finish the W&B run
    print("W&B ML Pipeline with Feast completed successfully.")
    return best_model

if __name__ == "__main__":
    # First, ensure that the data for Feast is prepared.
    # You would typically run `python prepare_feast_data.py` once before this.
    # And then `feast apply` and `feast materialize` from the `feature_repository` directory.
    
    # For this script to run, it's assumed `feast apply` has been run in `feature_repository`
    # and `iris_feast_source.parquet` exists in `data/`.
    
    print("Starting ML pipeline with W&B tracking...")
    run_wandb_pipeline()
    print("Pipeline execution finished.") 