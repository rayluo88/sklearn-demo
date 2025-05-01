import unittest
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from unittest.mock import patch, MagicMock
import mlflow
import os
import sys

# Add the parent directory to the path so we can import from the root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the pipeline function
from simple_ml_pipeline import run_pipeline, TEST_SIZE, RANDOM_STATE, N_NEIGHBORS

class TestSimpleMlPipeline(unittest.TestCase):
    """Unit tests for the ML pipeline components."""

    def setUp(self):
        """Set up test fixtures."""
        # Load iris dataset for testing
        self.iris = load_iris()
        self.X, self.y = self.iris.data, self.iris.target
        
    def test_data_loading(self):
        """Test that the Iris dataset loads with expected properties."""
        X, y = self.X, self.y
        
        # Check data shape and properties
        self.assertEqual(X.shape[0], 150, "Dataset should have 150 samples")
        self.assertEqual(X.shape[1], 4, "Dataset should have 4 features")
        self.assertEqual(len(np.unique(y)), 3, "Dataset should have 3 classes")
        self.assertEqual(set(np.unique(y)), {0, 1, 2}, "Classes should be 0, 1, and 2")
        
        # Check data types
        self.assertTrue(np.issubdtype(X.dtype, np.number), "Features should be numeric")
        self.assertTrue(np.issubdtype(y.dtype, np.integer), "Labels should be integers")
    
    def test_data_splitting(self):
        """Test that train-test splitting works as expected."""
        from sklearn.model_selection import train_test_split
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=self.y
        )
        
        # Check split sizes
        expected_test_size = int(len(self.X) * TEST_SIZE)
        expected_train_size = len(self.X) - expected_test_size
        
        self.assertEqual(len(X_test), expected_test_size, 
                        f"Test set should have {expected_test_size} samples")
        self.assertEqual(len(X_train), expected_train_size, 
                        f"Train set should have {expected_train_size} samples")
        
        # Check stratification (each class should have proportional representation)
        train_class_counts = np.bincount(y_train)
        test_class_counts = np.bincount(y_test)
        
        for i in range(3):  # For each class
            train_ratio = train_class_counts[i] / len(y_train)
            test_ratio = test_class_counts[i] / len(y_test)
            self.assertAlmostEqual(train_ratio, test_ratio, delta=0.1, 
                                 msg=f"Class {i} ratio should be similar in train and test")
    
    def test_scaling(self):
        """Test that feature scaling works correctly."""
        # Create a simple dataset
        X_sample = np.array([
            [5.1, 3.5, 1.4, 0.2],  # Sample 1
            [7.0, 3.2, 4.7, 1.4],  # Sample 2
            [6.3, 3.3, 6.0, 2.5]   # Sample 3
        ])
        
        # Scale the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_sample)
        
        # Check that scaled data has mean ≈ 0 and std ≈ 1 for each feature
        self.assertTrue(np.allclose(X_scaled.mean(axis=0), 0, atol=1e-10),
                      "Scaled features should have mean close to 0")
        self.assertTrue(np.allclose(X_scaled.std(axis=0), 1, atol=1e-10),
                      "Scaled features should have std dev close to 1")
        
        # Check that relative ordering is preserved after scaling
        for col in range(X_sample.shape[1]):
            original_order = np.argsort(X_sample[:, col])
            scaled_order = np.argsort(X_scaled[:, col])
            np.testing.assert_array_equal(original_order, scaled_order,
                                        "Scaling should preserve feature ordering")
    
    def test_model_training(self):
        """Test that the KNN model trains correctly."""
        # Create a simple dataset
        X_train = np.array([
            [0, 0], [0, 1], [1, 0], [1, 1]  # 4 points in a square
        ])
        y_train = np.array([0, 0, 1, 1])  # Diagonal classification
        
        # Train the model
        model = KNeighborsClassifier(n_neighbors=1)  # Using 1-NN for deterministic behavior
        model.fit(X_train, y_train)
        
        # Check that model predicts the training data correctly
        y_pred = model.predict(X_train)
        np.testing.assert_array_equal(y_pred, y_train, 
                                    "Model should perfectly predict training data")
        
        # Check prediction for a new point
        X_new = np.array([[0.1, 0.1]])  # Should be closest to (0,0)
        y_new_pred = model.predict(X_new)
        self.assertEqual(y_new_pred[0], 0, 
                       "New point [0.1, 0.1] should be classified as class 0")
    
    def test_model_prediction(self):
        """Test model prediction with varying k values."""
        # Create a simple dataset where different k values give different results
        X_train = np.array([
            [0, 0], [0.1, 0.1], [0.9, 0.9], [1, 1], [1.1, 1.1]
        ])
        y_train = np.array([0, 0, 1, 1, 1])
        
        X_test = np.array([[0.6, 0.6]])  # Test point in the middle
        
        # With k=1, should get class 1 (closest to [0.9, 0.9])
        model_k1 = KNeighborsClassifier(n_neighbors=1)
        model_k1.fit(X_train, y_train)
        y_pred_k1 = model_k1.predict(X_test)
        self.assertEqual(y_pred_k1[0], 1, "With k=1, test point should be class 1")
        
        # With k=3, should still get class 1 (2 points from class 1, 1 from class 0)
        model_k3 = KNeighborsClassifier(n_neighbors=3)
        model_k3.fit(X_train, y_train)
        y_pred_k3 = model_k3.predict(X_test)
        self.assertEqual(y_pred_k3[0], 1, "With k=3, test point should be class 1")
        
        # With k=5, should get class 1 (majority of 5 neighbors)
        model_k5 = KNeighborsClassifier(n_neighbors=5)
        model_k5.fit(X_train, y_train)
        y_pred_k5 = model_k5.predict(X_test)
        self.assertEqual(y_pred_k5[0], 1, "With k=5, test point should be class 1")
    
    @patch('mlflow.log_param')
    @patch('mlflow.log_metric')
    @patch('mlflow.log_artifact')
    @patch('mlflow.sklearn.log_model')
    @patch('mlflow.start_run')
    def test_mlflow_integration(self, mock_start_run, mock_log_model, 
                               mock_log_artifact, mock_log_metric, mock_log_param):
        """Test that MLflow tracking is called with correct arguments."""
        # Mock the mlflow.start_run context manager
        mock_start_run.return_value.__enter__ = MagicMock(return_value=None)
        mock_start_run.return_value.__exit__ = MagicMock(return_value=None)
        
        # Temporarily redirect stdout to avoid cluttering test output
        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        
        try:
            # Run the pipeline
            run_pipeline()
            
            # Check that MLflow functions were called
            mock_start_run.assert_called_once()
            
            # Check params were logged
            mock_log_param.assert_any_call("test_size", TEST_SIZE)
            mock_log_param.assert_any_call("random_state", RANDOM_STATE)
            mock_log_param.assert_any_call("n_neighbors", N_NEIGHBORS)
            
            # Check metrics were logged
            mock_log_metric.assert_any_call("accuracy", unittest.mock.ANY)
            
            # Check model was logged
            mock_log_model.assert_called_once()
            
            # Check artifact was logged
            mock_log_artifact.assert_called_once_with("classification_report.txt")
        finally:
            # Restore stdout
            sys.stdout.close()
            sys.stdout = original_stdout
    
    @patch('mlflow.start_run')
    def test_end_to_end_pipeline(self, mock_start_run):
        """Test the entire pipeline end-to-end."""
        # Mock the mlflow.start_run context manager
        mock_start_run.return_value.__enter__ = MagicMock(return_value=None)
        mock_start_run.return_value.__exit__ = MagicMock(return_value=None)
        
        # Patch all MLflow functions to avoid actual logging
        with patch('mlflow.log_param'), \
             patch('mlflow.log_metric'), \
             patch('mlflow.log_artifact'), \
             patch('mlflow.sklearn.log_model'), \
             patch('mlflow.set_tag'):
            
            # Temporarily redirect stdout
            original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
            
            try:
                # Run the pipeline
                run_pipeline()
                
                # We just check that it runs without errors
                # More specific component tests are in other test methods
                mock_start_run.assert_called_once()
            finally:
                # Restore stdout
                sys.stdout.close()
                sys.stdout = original_stdout

if __name__ == "__main__":
    unittest.main() 