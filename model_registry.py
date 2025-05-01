import mlflow
from mlflow.tracking import MlflowClient

def register_model():
    """Register the latest model in the MLflow Model Registry."""
    # Get the experiment ID (using the default experiment in this case)
    client = MlflowClient()
    experiment = client.get_experiment_by_name("Default")
    
    if experiment is None:
        print("No experiment found. Run the pipeline first.")
        return
    
    # Get the latest run with highest accuracy
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.accuracy DESC"]
    )
    
    if not runs:
        print("No runs found. Run the pipeline first.")
        return
    
    best_run = runs[0]
    run_id = best_run.info.run_id
    accuracy = best_run.data.metrics.get("accuracy", 0)
    
    print(f"Found best run (ID: {run_id}) with accuracy: {accuracy:.4f}")
    
    # Register the model from this run
    model_uri = f"runs:/{run_id}/knn-model"
    model_name = "iris-classifier"
    
    # Register model in the MLflow Model Registry
    model_details = mlflow.register_model(model_uri, model_name)
    
    print(f"Model registered with name: {model_details.name}")
    print(f"Model version: {model_details.version}")
    
    # Optional: Set model version stage (Staging, Production, Archived)
    client.transition_model_version_stage(
        name=model_details.name,
        version=model_details.version,
        stage="Staging",
    )
    
    print(f"Model {model_details.name} version {model_details.version} transitioned to Staging")
    
    return model_details

if __name__ == "__main__":
    register_model() 