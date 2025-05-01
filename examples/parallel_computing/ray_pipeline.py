"""
Example of enhancing the ML pipeline with Ray for parallel computing
"""
import numpy as np
import mlflow
import mlflow.sklearn
import os
import ray
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from ray.data import from_numpy
from ray.tune.tuner import Tuner
from ray.tune.search.optuna import OptunaSearch
from ray.air.config import RunConfig, ScalingConfig
from ray.tune.tune_config import TuneConfig
from ray.train.sklearn import SklearnTrainer

# Constants
TEST_SIZE = 0.2
RANDOM_STATE = 42


def train_model(config):
    """Training function for Ray to distribute."""
    # This function will be executed on each worker
    import numpy as np
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score
    
    # Load data
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create and train model with config parameters
    model = KNeighborsClassifier(n_neighbors=config["n_neighbors"])
    model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    # In newer Ray versions, we need to return metrics in a standardized format
    return {"accuracy": accuracy, "model": model, "scaler": scaler}


def run_ray_hyperparameter_tuning():
    """Runs hyperparameter tuning with Ray Tune."""
    print("Starting Ray for distributed computing...")
    ray.init(num_cpus=2)  # Adjust based on your system
    
    with mlflow.start_run(run_name="ray-hyperparameter-tuning"):
        mlflow.set_tag("computing_framework", "ray")
        mlflow.set_tag("task", "hyperparameter_tuning")
        
        # Log fixed parameters
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("random_state", RANDOM_STATE)
        
        print("\nRunning hyperparameter tuning with Ray Tune...")
        # Define the hyperparameter search space
        param_space = {
            "n_neighbors": ray.tune.randint(1, 10)
        }
        
        # Configure the tuning process
        tune_config = TuneConfig(
            num_samples=5,  # Number of hyperparameter combinations to try
            search_alg=OptunaSearch(),  # Using Optuna for efficient search
            metric="accuracy",
            mode="max",
        )
        
        # Create a Tuner
        trainer = Tuner(
            trainable=train_model,
            param_space=param_space,
            tune_config=tune_config,
            run_config=RunConfig(
                name="knn_tuning",
                local_dir="./ray_results",
            )
        )
        
        # Run the tuning
        results = trainer.fit()
        
        # Get the best trial
        best_result = results.get_best_result(metric="accuracy", mode="max")
        best_config = best_result.config
        best_accuracy = best_result.metrics["accuracy"]
        
        print(f"\nBest hyperparameters found: {best_config}")
        print(f"Best accuracy: {best_accuracy:.4f}")
        
        # Run the best model configuration again to get the model
        best_run_result = train_model(best_config)
        best_model = best_run_result["model"]
        
        # Log the best hyperparameters and metrics
        mlflow.log_params(best_config)
        mlflow.log_metric("best_accuracy", best_accuracy)
        
        # Log the best model
        mlflow.sklearn.log_model(best_model, "ray-tuned-knn-model")
        
        print("\nBest model logged to MLflow")
    
    # Shut down Ray
    ray.shutdown()
    print("\nRay shut down")


def run_ray_batch_inference():
    """Example of using Ray for parallel batch inference."""
    print("Starting Ray for distributed batch inference...")
    ray.init(num_cpus=2)  # Adjust based on your system
    
    with mlflow.start_run(run_name="ray-batch-inference"):
        mlflow.set_tag("computing_framework", "ray")
        mlflow.set_tag("task", "batch_inference")
        
        # 1. Load Data
        print("\nLoading Iris dataset...")
        iris = load_iris()
        X, y = iris.data, iris.target
        
        # 2. Train a basic model (this would normally be a pre-trained model)
        print("\nTraining a basic model...")
        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(X, y)
        
        # 3. Create a Ray Dataset for parallel inference
        print("\nCreating Ray Dataset for inference...")
        # In a real scenario, this could be loading from files/database
        inference_dataset = from_numpy(X)
        
        # 4. Define a remote function for batch prediction
        @ray.remote
        def predict_batch(model, batch):
            return model.predict(batch)
        
        # 5. Split the dataset into batches and process in parallel
        print("\nPerforming batch inference in parallel...")
        batches = inference_dataset.split(n=4)  # Split into 4 batches
        
        # Create remote tasks for each batch
        prediction_refs = []
        for batch in batches:
            batch_np = np.array(list(batch.iter_rows()))
            prediction_refs.append(predict_batch.remote(model, batch_np))
        
        # Get results
        predictions = ray.get(prediction_refs)
        
        # Combine the results
        all_predictions = np.concatenate(predictions)
        
        # 6. Evaluate results
        accuracy = accuracy_score(y, all_predictions)
        report = classification_report(y, all_predictions, target_names=iris.target_names)
        
        print(f"\nBatch inference accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(report)
        
        # Log metrics and model
        mlflow.log_metric("batch_inference_accuracy", accuracy)
        mlflow.sklearn.log_model(model, "ray-batch-model")
        
        # Log classification report as text
        report_path = "batch_classification_report.txt"
        with open(report_path, "w") as f:
            f.write(report)
        mlflow.log_artifact(report_path)
        os.remove(report_path)
        
        print("\nBatch inference results logged to MLflow")
    
    # Shut down Ray
    ray.shutdown()
    print("\nRay shut down")


if __name__ == "__main__":
    try:
        # Run hyperparameter tuning example
        run_ray_hyperparameter_tuning()
        
        # Run batch inference example
        run_ray_batch_inference()
    except Exception as e:
        print(f"Error running Ray examples: {e}")
        # Print more detailed traceback
        import traceback
        traceback.print_exc() 