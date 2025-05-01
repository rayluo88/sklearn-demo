import unittest
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import os
import sys

# Add the parent directory to the path so we can import from the root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestModelEvaluation(unittest.TestCase):
    """Tests for evaluating model quality and hyperparameter tuning."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Load iris dataset for testing
        self.iris = load_iris()
        self.X, self.y = self.iris.data, self.iris.target
        
        # Create a preprocessing and model pipeline
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier(n_neighbors=5))
        ])
    
    def test_cross_validation(self):
        """Test model performance using cross-validation."""
        # Perform 5-fold cross-validation
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.pipeline, self.X, self.y, cv=cv, scoring='accuracy')
        
        # Check that cross-validation scores are reasonable
        self.assertTrue(all(score > 0.8 for score in cv_scores), 
                      f"All CV scores should be > 0.8, got {cv_scores}")
        
        # Check mean and std of scores
        mean_score = np.mean(cv_scores)
        std_score = np.std(cv_scores)
        
        print(f"Cross-validation results: {mean_score:.4f} ± {std_score:.4f}")
        
        # For Iris with KNN, we expect very good performance
        self.assertGreater(mean_score, 0.9, 
                         f"Mean CV score should be > 0.9, got {mean_score:.4f}")
        self.assertLess(std_score, 0.1, 
                      f"Std of CV scores should be < 0.1, got {std_score:.4f}")
    
    def test_hyperparameter_tuning(self):
        """Test hyperparameter tuning with grid search."""
        # Define parameter grid
        param_grid = {
            'knn__n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15],
            'knn__weights': ['uniform', 'distance']
        }
        
        # Set up grid search
        grid_search = GridSearchCV(
            self.pipeline, param_grid, cv=5, 
            scoring='accuracy', return_train_score=True
        )
        
        # Run grid search
        grid_search.fit(self.X, self.y)
        
        # Check best parameters and score
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        # We expect the best score to be very good for Iris
        self.assertGreater(grid_search.best_score_, 0.95, 
                         "Best CV score should be > 0.95")
        
        # The best k is typically small for Iris data
        self.assertLessEqual(grid_search.best_params_['knn__n_neighbors'], 9,
                           "Best n_neighbors should be ≤ 9 for Iris data")
        
        # Plot results if this test is run directly (not through unittest)
        if __name__ == "__main__":
            self._plot_grid_search_results(grid_search)
    
    def test_prediction_probabilities(self):
        """Test that prediction probabilities sum to 1 and match predictions."""
        # Fit the pipeline
        self.pipeline.fit(self.X, self.y)
        
        # Get predictions and probabilities
        y_pred = self.pipeline.predict(self.X)
        y_proba = self.pipeline.predict_proba(self.X)
        
        # Check probability shape (n_samples, n_classes)
        self.assertEqual(y_proba.shape, (len(self.X), len(np.unique(self.y))),
                       "Probability matrix should be (n_samples, n_classes)")
        
        # Check probabilities sum to 1
        self.assertTrue(np.allclose(np.sum(y_proba, axis=1), 1.0),
                      "Probabilities should sum to 1 for each sample")
        
        # Check that predictions match class with highest probability
        y_pred_from_proba = np.argmax(y_proba, axis=1)
        np.testing.assert_array_equal(y_pred, y_pred_from_proba,
                                    "Predictions should match argmax of probabilities")
    
    def test_feature_importance(self):
        """Evaluate feature importance through permutation importance."""
        from sklearn.inspection import permutation_importance
        
        # Fit the pipeline
        self.pipeline.fit(self.X, self.y)
        
        # Calculate permutation importance
        result = permutation_importance(
            self.pipeline, self.X, self.y, n_repeats=10, random_state=42
        )
        
        # Check that we have importance scores for each feature
        self.assertEqual(len(result.importances_mean), self.X.shape[1],
                       "Should have importance scores for each feature")
        
        # Print feature importance
        feature_names = self.iris.feature_names
        importances = result.importances_mean
        std = result.importances_std
        
        print("\nFeature Importance:")
        for i in range(len(importances)):
            print(f"{feature_names[i]}: {importances[i]:.4f} ± {std[i]:.4f}")
        
        # For Iris, typically petal length and width are more important
        petal_indices = [2, 3]  # Indices of petal length and width
        sepal_indices = [0, 1]  # Indices of sepal length and width
        
        petal_importance = np.mean(importances[petal_indices])
        sepal_importance = np.mean(importances[sepal_indices])
        
        print(f"\nAverage petal importance: {petal_importance:.4f}")
        print(f"Average sepal importance: {sepal_importance:.4f}")
        
        # In Iris dataset, petal features are typically more important
        self.assertGreater(petal_importance, sepal_importance,
                         "Petal features should be more important than sepal features")
    
    def _plot_grid_search_results(self, grid_search):
        """Helper method to plot grid search results."""
        results = grid_search.cv_results_
        
        # Extract n_neighbors values
        n_neighbors_values = sorted(set([params['knn__n_neighbors'] 
                                      for params in results['params']]))
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Plot uniform weights
        uniform_scores = [results['mean_test_score'][i] 
                        for i, params in enumerate(results['params']) 
                        if params['knn__weights'] == 'uniform']
        plt.plot(n_neighbors_values, uniform_scores, 'o-', label='Uniform weights')
        
        # Plot distance weights
        distance_scores = [results['mean_test_score'][i] 
                         for i, params in enumerate(results['params']) 
                         if params['knn__weights'] == 'distance']
        plt.plot(n_neighbors_values, distance_scores, 's-', label='Distance weights')
        
        # Add labels and legend
        plt.xlabel('Number of Neighbors (k)')
        plt.ylabel('Cross-validation Accuracy')
        plt.title('KNN Performance vs. Number of Neighbors')
        plt.xticks(n_neighbors_values)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Save plot
        plot_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'knn_hyperparameter_tuning.png')
        plt.savefig(plot_path)
        print(f"Plot saved as '{plot_path}'")
        plt.close()

if __name__ == "__main__":
    unittest.main() 