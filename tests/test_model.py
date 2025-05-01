import pytest
import numpy as np
from sklearn.datasets import load_iris
import mlflow.pyfunc
import os
import sys

# Add the parent directory to the path so we can import from the root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_model_prediction():
    """Test that a model loaded from MLflow can make predictions."""
    try:
        # Try to load registered model first (requires MLflow Model Registry)
        model = mlflow.pyfunc.load_model("models:/iris-classifier/latest")
    except Exception as e:
        # If model registry fails, look for local runs
        print(f"Could not load from registry: {e}")
        # Find the most recent run
        runs_dir = "mlruns/0"  # Default experiment
        if not os.path.exists(runs_dir):
            pytest.skip("No MLflow runs found. Please run the pipeline first.")
            
        # Find newest run by sorting directories by modification time
        run_ids = []
        for item in os.listdir(runs_dir):
            if os.path.isdir(os.path.join(runs_dir, item)) and not item.startswith('.'):
                run_ids.append(item)
        
        if not run_ids:
            pytest.skip("No run IDs found")
            
        run_dirs = [os.path.join(runs_dir, run_id) for run_id in run_ids]
        run_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        
        if not run_dirs:
            pytest.skip("No valid run directories found")
            
        # Find model artifacts in the newest run
        artifacts_dir = os.path.join(run_dirs[0], "artifacts")
        if not os.path.exists(artifacts_dir):
            pytest.skip(f"No artifacts found in run directory {run_dirs[0]}")
            
        model_dir = os.path.join(artifacts_dir, "knn-model")
        if not os.path.exists(model_dir):
            pytest.skip(f"No model directory found in artifacts {artifacts_dir}")
            
        # Load model from the local path
        model = mlflow.pyfunc.load_model(model_dir)
    
    # Test the model using Iris data
    iris = load_iris()
    # Take a few samples
    X_samples = iris.data[:5]
    
    # Make predictions
    predictions = model.predict(X_samples)
    
    # Assertions
    assert predictions is not None
    assert len(predictions) == 5
    # All predictions should be one of the possible classes (0, 1, 2)
    assert all(pred in [0, 1, 2] for pred in predictions)
    
    # Check that predictions are deterministic (same input = same output)
    predictions2 = model.predict(X_samples)
    assert np.array_equal(predictions, predictions2)
    
    # Test a known sample (first iris sample should be class 0 - setosa)
    known_sample = iris.data[0:1]  # setosa
    setosa_prediction = model.predict(known_sample)
    assert setosa_prediction[0] == 0  # Should predict class 0 (setosa)

if __name__ == "__main__":
    test_model_prediction() 