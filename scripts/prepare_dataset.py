import pandas as pd
from sklearn.datasets import load_iris
import os

def main():
    # Create data directory if it doesn't exist
    if not os.path.exists("data"):
        os.makedirs("data")

    # Load Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names # For reference, not directly in CSV usually

    # Create a DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    # Add species names for easier interpretation if needed, though target is primary
    # df['species'] = df['target'].map(lambda i: target_names[i])

    # Define output path
    output_path = "data/iris.csv"

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Iris dataset saved to {output_path}")
    print(f"Dataset dimensions: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Feature columns: {feature_names}")
    print(f"Target column: 'target' (0: {target_names[0]}, 1: {target_names[1]}, 2: {target_names[2]})")

if __name__ == "__main__":
    main() 