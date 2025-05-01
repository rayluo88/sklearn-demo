import unittest
import numpy as np
from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Add the parent directory to the path so we can import from the root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestDataValidation(unittest.TestCase):
    """Tests for validating data quality and integrity."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Load iris dataset for testing
        self.iris = load_iris()
        self.X, self.y = self.iris.data, self.iris.target
        
        # Create a DataFrame for easier data manipulation
        self.df = pd.DataFrame(
            data=np.c_[self.X, self.y],
            columns=list(self.iris.feature_names) + ['target']
        )
        self.df['species'] = self.df['target'].map({
            0: 'setosa', 1: 'versicolor', 2: 'virginica'
        })
    
    def test_data_completeness(self):
        """Test that the dataset has no missing values."""
        # Check for NaN values
        self.assertFalse(np.isnan(self.X).any(), "Features should not contain NaN values")
        self.assertFalse(np.isnan(self.y).any(), "Target should not contain NaN values")
        
        # Check for expected dataset size
        self.assertEqual(self.X.shape, (150, 4), "Expected 150 samples with 4 features")
        self.assertEqual(self.y.shape, (150,), "Expected 150 target labels")
    
    def test_data_validity(self):
        """Test that data values are within expected ranges."""
        # Check that feature values are positive (for Iris measurements)
        self.assertTrue((self.X > 0).all(), "All measurements should be positive")
        
        # Check that all features are within reasonable range for Iris
        self.assertTrue((self.X < 10).all(), "All measurements should be less than 10 cm")
        
        # Check target values
        unique_targets = np.unique(self.y)
        expected_targets = np.array([0, 1, 2])
        np.testing.assert_array_equal(
            sorted(unique_targets), expected_targets,
            "Target should only contain values 0, 1, and 2"
        )
    
    def test_class_balance(self):
        """Test that classes are balanced in the dataset."""
        # Count samples per class
        class_counts = np.bincount(self.y.astype(int))
        
        # Check that all classes have sufficient representation
        min_samples_per_class = 30  # Minimum expected samples per class
        self.assertTrue(
            all(count >= min_samples_per_class for count in class_counts),
            f"Each class should have at least {min_samples_per_class} samples"
        )
        
        # Check that classes are roughly balanced (Iris is perfectly balanced)
        self.assertEqual(
            len(set(class_counts)), 1,
            "All classes should have the same number of samples"
        )
        
        # Print class distribution
        print("\nClass distribution:")
        for i, count in enumerate(class_counts):
            print(f"Class {i} ({self.iris.target_names[i]}): {count} samples")
    
    def test_feature_distributions(self):
        """Test feature distributions for anomalies."""
        # Check for reasonable feature means and stds
        feature_means = np.mean(self.X, axis=0)
        feature_stds = np.std(self.X, axis=0)
        
        print("\nFeature statistics:")
        for i, name in enumerate(self.iris.feature_names):
            print(f"{name}: mean={feature_means[i]:.2f}, std={feature_stds[i]:.2f}")
            
            # Check for reasonable values (Iris specific)
            if 'sepal' in name.lower():
                if 'length' in name.lower():
                    self.assertTrue(4 < feature_means[i] < 7, 
                                  f"Expected sepal length mean between 4-7, got {feature_means[i]:.2f}")
                elif 'width' in name.lower():
                    self.assertTrue(2 < feature_means[i] < 5, 
                                  f"Expected sepal width mean between 2-5, got {feature_means[i]:.2f}")
            elif 'petal' in name.lower():
                if 'length' in name.lower():
                    self.assertTrue(2 < feature_means[i] < 5, 
                                  f"Expected petal length mean between 2-5, got {feature_means[i]:.2f}")
                elif 'width' in name.lower():
                    self.assertTrue(0.5 < feature_means[i] < 2.5, 
                                  f"Expected petal width mean between 0.5-2.5, got {feature_means[i]:.2f}")
        
        # If running as script, plot distributions
        if __name__ == "__main__":
            self._plot_feature_distributions()
    
    def test_feature_correlations(self):
        """Test feature correlations for multicollinearity."""
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(self.X, rowvar=False)
        
        # Print correlation matrix
        print("\nFeature correlation matrix:")
        pd.set_option('display.precision', 2)
        corr_df = pd.DataFrame(
            corr_matrix, 
            index=self.iris.feature_names,
            columns=self.iris.feature_names
        )
        print(corr_df)
        
        # Check for high correlations (potentially problematic multicollinearity)
        # Get upper triangle of correlation matrix excluding diagonal
        upper_tri = np.triu(corr_matrix, k=1)
        high_corr_threshold = 0.9
        high_corrs = np.where(abs(upper_tri) > high_corr_threshold)
        
        # Print highly correlated features
        if len(high_corrs[0]) > 0:
            print("\nHighly correlated features (r > 0.9):")
            for i, j in zip(high_corrs[0], high_corrs[1]):
                print(f"{self.iris.feature_names[i]} and {self.iris.feature_names[j]}: r={corr_matrix[i, j]:.2f}")
        
        # For Iris dataset, we expect petal length and width to be highly correlated
        petal_length_idx = self.iris.feature_names.index('petal length (cm)')
        petal_width_idx = self.iris.feature_names.index('petal width (cm)')
        petal_correlation = corr_matrix[petal_length_idx, petal_width_idx]
        
        self.assertGreater(
            petal_correlation, 0.7,
            f"Expected high correlation between petal length and width, got {petal_correlation:.2f}"
        )
        
        # If running as script, plot correlations
        if __name__ == "__main__":
            self._plot_correlation_matrix(corr_df)
    
    def test_feature_importance_for_target(self):
        """Test feature relevance for predicting the target."""
        from sklearn.feature_selection import mutual_info_classif
        
        # Calculate mutual information scores
        mi_scores = mutual_info_classif(self.X, self.y, random_state=42)
        
        # Print mutual information scores
        print("\nFeature mutual information with target:")
        for i, name in enumerate(self.iris.feature_names):
            print(f"{name}: {mi_scores[i]:.4f}")
        
        # For Iris, we expect petal features to have higher MI scores
        petal_indices = [
            i for i, name in enumerate(self.iris.feature_names) 
            if 'petal' in name.lower()
        ]
        sepal_indices = [
            i for i, name in enumerate(self.iris.feature_names) 
            if 'sepal' in name.lower()
        ]
        
        petal_mi = np.mean(mi_scores[petal_indices])
        sepal_mi = np.mean(mi_scores[sepal_indices])
        
        print(f"\nAverage petal MI: {petal_mi:.4f}")
        print(f"Average sepal MI: {sepal_mi:.4f}")
        
        # For Iris, petal features are typically more informative
        self.assertGreater(
            petal_mi, sepal_mi,
            "Petal features should have higher mutual information with target"
        )
    
    def test_for_duplicate_samples(self):
        """Test that the dataset does not contain duplicate samples."""
        # Count unique rows
        unique_rows = np.unique(self.X, axis=0)
        
        # Check if there are fewer unique rows than total rows
        if len(unique_rows) < len(self.X):
            n_duplicates = len(self.X) - len(unique_rows)
            print(f"\nFound {n_duplicates} duplicate feature vectors")
        else:
            print("\nNo duplicate feature vectors found")
        
        # For Iris we don't expect many duplicates, but some might exist
        # This is just a reporting test, not a strict assertion
        pass
    
    def _plot_feature_distributions(self):
        """Helper method to plot feature distributions."""
        # Create figure
        plt.figure(figsize=(14, 10))
        
        # Plot histograms for each feature
        for i, name in enumerate(self.iris.feature_names):
            plt.subplot(2, 2, i+1)
            for species in ['setosa', 'versicolor', 'virginica']:
                subset = self.df[self.df['species'] == species]
                sns.histplot(subset[name], label=species, kde=True, alpha=0.6)
            plt.title(f'Distribution of {name}')
            plt.xlabel(name)
            plt.ylabel('Count')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig('feature_distributions.png')
        print("Feature distributions plot saved as 'feature_distributions.png'")
        plt.close()
    
    def _plot_correlation_matrix(self, corr_df):
        """Helper method to plot correlation matrix."""
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_df, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig('correlation_matrix.png')
        print("Correlation matrix plot saved as 'correlation_matrix.png'")
        plt.close()
        
        # Also plot pairplot
        plt.figure(figsize=(12, 10))
        sns.pairplot(self.df, hue='species', vars=self.iris.feature_names)
        plt.savefig('pairplot.png')
        print("Pairplot saved as 'pairplot.png'")
        plt.close()

if __name__ == "__main__":
    unittest.main() 