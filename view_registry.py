import mlflow
from mlflow.tracking import MlflowClient

def view_registered_models():
    """Display all registered models and their versions."""
    client = MlflowClient()
    
    # List all registered models
    print("=== Registered Models ===")
    for rm in client.search_registered_models():
        print(f"Model: {rm.name}")
    
    # Get details for our specific model
    model_name = "iris-classifier"
    try:
        model_details = client.get_registered_model(model_name)
        print(f"\n=== Model: {model_name} ===")
        print(f"Description: {model_details.description}")
        print(f"Latest versions: {len(model_details.latest_versions)}")
        
        # Get all versions of this model
        print("\n=== Model Versions ===")
        for version in client.search_model_versions(f"name='{model_name}'"):
            print(f"  Version: {version.version}")
            print(f"  Stage: {version.current_stage}")
            print(f"  Status: {version.status}")
            print(f"  Run ID: {version.run_id}")
            
            # Get the run that created this model version
            run = client.get_run(version.run_id)
            accuracy = run.data.metrics.get("accuracy", "N/A")
            print(f"  Accuracy: {accuracy}")
            print("  ---------")
    except Exception as e:
        print(f"Error retrieving model '{model_name}': {e}")
        print("Have you registered the model by running 'python model_registry.py'?")

if __name__ == "__main__":
    view_registered_models() 