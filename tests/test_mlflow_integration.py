import unittest
import os
import shutil
import tempfile
import numpy as np
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from unittest.mock import patch, MagicMock
import sys

# Add the parent directory to the path so we can import from the root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import from the pipeline
from simple_ml_pipeline import run_pipeline

class TestMLflowIntegration(unittest.TestCase):
    """Tests for MLflow tracking and model registry integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for MLflow tracking
        self.test_dir = tempfile.mkdtemp()
        self.tracking_uri = f"file:{self.test_dir}"
        
        # Set MLflow tracking URI to the test directory
        mlflow.set_tracking_uri(self.tracking_uri)
        
        # Use the experiment name from the pipeline
        self.experiment_name = "Iris Classification with Feast"
        # Ensure the experiment exists for the tests
        if mlflow.get_experiment_by_name(self.experiment_name) is None:
            mlflow.create_experiment(self.experiment_name)
        mlflow.set_experiment(self.experiment_name)
        
        # Load iris dataset for testing
        self.iris = load_iris()
        self.X, self.y = self.iris.data, self.iris.target
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Remove temporary directory
        shutil.rmtree(self.test_dir)
    
    def test_mlflow_logging(self):
        """Test basic MLflow parameter and metric logging."""
        # Start a run manually
        with mlflow.start_run() as run:
            # Log some parameters
            mlflow.log_param("test_param_1", 42)
            mlflow.log_param("test_param_2", "value")
            
            # Log some metrics
            mlflow.log_metric("test_metric_1", 0.95)
            mlflow.log_metric("test_metric_2", 0.5)
            
            # Log a model
            model = KNeighborsClassifier(n_neighbors=3)
            model.fit(self.X, self.y)
            mlflow.sklearn.log_model(model, "test_model")
            
            # Get run info
            run_id = run.info.run_id
        
        # Verify logged data using MLflow client
        client = MlflowClient()
        run_data = client.get_run(run_id)
        
        # Check parameters
        self.assertEqual(run_data.data.params["test_param_1"], "42")
        self.assertEqual(run_data.data.params["test_param_2"], "value")
        
        # Check metrics
        self.assertEqual(run_data.data.metrics["test_metric_1"], 0.95)
        self.assertEqual(run_data.data.metrics["test_metric_2"], 0.5)
        
        # Check artifacts
        artifacts = client.list_artifacts(run_id)
        artifact_paths = [a.path for a in artifacts]
        self.assertIn("test_model", artifact_paths)
    
    def test_mlflow_pipeline_integration(self):
        """Test that our ML pipeline correctly integrates with MLflow."""
        # Redirect all stdout to avoid cluttering the test output
        with patch('sys.stdout'):
            # Run the pipeline
            run_pipeline()
        
        # Get the latest run
        client = MlflowClient()
        experiment = client.get_experiment_by_name(self.experiment_name)
        self.assertIsNotNone(experiment, f"Experiment '{self.experiment_name}' not found.")
        runs = client.search_runs(experiment_ids=[experiment.experiment_id])
        
        # We should have at least one run
        self.assertTrue(len(runs) > 0, "No MLflow runs were recorded")
        
        # Get the latest run
        latest_run = runs[0]
        
        # Check that expected parameters are logged
        params = latest_run.data.params
        self.assertIn("test_size", params)
        self.assertIn("random_state", params)
        self.assertIn("n_estimators", params)
        self.assertIn("max_depth", params)
        self.assertIn("min_samples_split", params)
        
        # Check that expected metrics are logged
        metrics = latest_run.data.metrics
        self.assertIn("accuracy", metrics)
        self.assertTrue(0 <= metrics["accuracy"] <= 1, "Accuracy should be between 0 and 1")
        
        # Check that expected artifacts are logged
        artifacts = client.list_artifacts(latest_run.info.run_id)
        artifact_paths = [a.path for a in artifacts]
        self.assertIn("iris-classifier", artifact_paths)
    
    def test_model_loading(self):
        """Test that we can load a model from MLflow."""
        # First, run a pipeline and log a model
        with patch('sys.stdout'):
            run_pipeline()
        
        # Get the latest run
        client = MlflowClient()
        experiment = client.get_experiment_by_name(self.experiment_name)
        self.assertIsNotNone(experiment, f"Experiment '{self.experiment_name}' not found.")
        runs = client.search_runs(experiment_ids=[experiment.experiment_id])
        latest_run = runs[0]
        run_id = latest_run.info.run_id
        
        # Load the model
        model_uri = f"runs:/{run_id}/iris-classifier"
        loaded_model = mlflow.sklearn.load_model(model_uri)
        
        # Check that it's the right type of model
        from sklearn.ensemble import RandomForestClassifier
        self.assertIsInstance(loaded_model, RandomForestClassifier)
        
        # Check that the model can make predictions
        X_test = self.X[:5]  # Use first 5 samples as test
        predictions = loaded_model.predict(X_test)
        self.assertEqual(len(predictions), len(X_test))
    
    def test_custom_pipeline_logging(self):
        """Test logging a custom sklearn pipeline with MLflow."""
        from sklearn.pipeline import Pipeline
        
        # Create a simple pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', KNeighborsClassifier(n_neighbors=3))
        ])
        
        # Fit the pipeline
        pipeline.fit(self.X, self.y)
        
        # Log the pipeline with MLflow
        with mlflow.start_run() as run:
            mlflow.sklearn.log_model(pipeline, "pipeline_model")
            run_id = run.info.run_id
        
        # Load the pipeline
        model_uri = f"runs:/{run_id}/pipeline_model"
        loaded_pipeline = mlflow.sklearn.load_model(model_uri)
        
        # Check that it's a Pipeline
        self.assertIsInstance(loaded_pipeline, Pipeline)
        
        # Check that it has the right steps
        self.assertEqual(len(loaded_pipeline.steps), 2)
        self.assertEqual(loaded_pipeline.steps[0][0], 'scaler')
        self.assertEqual(loaded_pipeline.steps[1][0], 'classifier')
        
        # Check that the pipeline works end-to-end
        X_test = self.X[:5]
        predictions = loaded_pipeline.predict(X_test)
        self.assertEqual(len(predictions), len(X_test))
    
    def test_serving_signature(self):
        """Test logging a model with a model signature for serving."""
        from mlflow.models.signature import infer_signature
        
        # Create and train a model
        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(self.X, self.y)
        
        # Generate predictions to infer output schema
        predictions = model.predict(self.X[:5])
        
        # Infer model signature
        signature = infer_signature(self.X[:5], predictions)
        
        # Log model with signature
        with mlflow.start_run(experiment_id=mlflow.get_experiment_by_name(self.experiment_name).experiment_id) as run:
            mlflow.sklearn.log_model(
                model,
                "model_with_signature",
                signature=signature,
                registered_model_name="model_with_signature"
            )
            run_id = run.info.run_id
        
        # Check signature in model metadata
        client = MlflowClient()
        model_version = client.get_latest_versions(
            name="model_with_signature",
            stages=["None"]
        )
        
        # Load the model
        model_uri = f"runs:/{run_id}/model_with_signature"
        loaded_model = mlflow.sklearn.load_model(model_uri)
        
        # Model should work for prediction
        test_input = self.X[:1]  # Single sample
        prediction = loaded_model.predict(test_input)
        self.assertEqual(len(prediction), 1)
    
    def test_model_registry(self):
        """Test basic model registry operations."""
        # Create and train a model
        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(self.X, self.y)
        
        # Log model to tracking server
        with mlflow.start_run() as run:
            mlflow.sklearn.log_model(model, "registry_test_model")
            run_id = run.info.run_id
        
        # Register model in registry
        model_uri = f"runs:/{run_id}/registry_test_model"
        registered_model_name = "test_registry_model"
        
        # Register the model
        mlflow.register_model(model_uri, registered_model_name)
        
        # Check that model exists in registry
        client = MlflowClient()
        registered_models = client.search_registered_models()
        model_names = [rm.name for rm in registered_models]
        
        self.assertIn(registered_model_name, model_names)
        
        # Get the latest version
        model_versions = client.get_latest_versions(registered_model_name)
        self.assertTrue(len(model_versions) > 0)
        
        # Check version details
        version = model_versions[0]
        self.assertEqual(version.name, registered_model_name)
        self.assertEqual(version.current_stage, "None")
        
        # Transition to staging
        client.transition_model_version_stage(
            name=registered_model_name,
            version=version.version,
            stage="Staging"
        )
        
        # Check that stage was updated
        updated_version = client.get_model_version(
            name=registered_model_name,
            version=version.version
        )
        self.assertEqual(updated_version.current_stage, "Staging")

if __name__ == "__main__":
    unittest.main() 