import pytest
import os
import sys
import numpy as np
from sklearn.datasets import load_iris
import tempfile
import mlflow

# Add the parent directory to the path so we can import from the root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pytest.fixture(scope="session")
def iris_dataset():
    """
    Fixture that provides the Iris dataset for testing.
    """
    iris = load_iris()
    X, y = iris.data, iris.target
    return X, y, iris.feature_names, iris.target_names

@pytest.fixture(scope="function")
def temp_mlflow_tracking():
    """
    Fixture that sets up a temporary MLflow tracking directory.
    """
    # Create a temporary directory for MLflow tracking
    test_dir = tempfile.mkdtemp()
    tracking_uri = f"file:{test_dir}"
    
    # Set MLflow tracking URI to the test directory
    original_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    
    # Create a new experiment for testing
    experiment_name = "test_experiment"
    mlflow.set_experiment(experiment_name)
    
    yield tracking_uri, experiment_name
    
    # Teardown: restore original URI and clean up temp dir
    mlflow.set_tracking_uri(original_uri)
    import shutil
    shutil.rmtree(test_dir) 