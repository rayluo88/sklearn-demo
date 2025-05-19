import pandas as pd
from datetime import datetime, timedelta
import numpy as np

# Define input and output paths
RAW_DATA_PATH = "data/iris.csv"
FEAST_SOURCE_PATH = "data/iris_feast_source.parquet" # This path is referenced in iris_definitions.py

def prepare_data_for_feast():
    """
    Reads the raw Iris CSV, adds iris_id and event_timestamp columns,
    and saves it as a Parquet file for Feast.
    """
    print(f"Loading raw data from {RAW_DATA_PATH}...")
    df = pd.read_csv(RAW_DATA_PATH)

    # Rename columns to be more Feast-friendly and match feature definitions
    column_name_mapping = {
        "sepal length (cm)": "sepal_length",
        "sepal width (cm)": "sepal_width",
        "petal length (cm)": "petal_length",
        "petal width (cm)": "petal_width",
        # 'target' is already a good name, will be handled later for target_class
    }
    df.rename(columns=column_name_mapping, inplace=True)
    # Ensure all other columns that might have spaces or special chars are also sanitized if necessary,
    # but for now, we are focusing on the feature columns and target.

    # Add an entity ID (iris_id)
    df['iris_id'] = range(len(df))

    # Add an event_timestamp column.
    # For demo purposes, we'll create a series of timestamps starting from some point in the past.
    now = datetime.utcnow()
    df['event_timestamp'] = [now - timedelta(days=len(df) - i) for i in range(len(df))]

    # The Iris dataset from sklearn has string targets, but if using a CSV that
    # already has integer targets (like the one provided), this mapping might be different
    # or not needed if the target column is already suitable.
    # For this example, assuming 'variety' is the column with 'Setosa', 'Versicolor', 'Virginica'
    # If your CSV has a numeric target (e.g., 0, 1, 2), adjust accordingly.
    # class_mapping = {
    # 'Setosa': 0,
    # 'Versicolor': 1,
    # 'Virginica': 2
    # }
    # df['target_class'] = df['variety'].map(class_mapping)
    
    # The CSV has a 'target' column with already encoded integers (0, 1, 2)
    # We will use this directly for 'target_class'
    if 'target' in df.columns:
        df['target_class'] = df['target']
    elif 'variety' in df.columns:
        # Fallback for a differently named column, assuming it needs mapping
        class_mapping = {
            'Setosa': 0,
            'Versicolor': 1,
            'Virginica': 2
        }
        df['target_class'] = df['variety'].map(class_mapping)
    else:
        raise ValueError("CSV must contain either a 'target' column with integer classes or a 'variety' column with string classes.")

    print(f"Columns in DataFrame after loading: {df.columns.tolist()}")

    # Select and reorder columns to match FeatureView (plus iris_id and event_timestamp)
    # The FileSource in Feast will pick up columns defined in the FeatureView schema
    # plus the timestamp_field and inferred entity join keys.
    # So the parquet file should contain: iris_id, event_timestamp, sepal_length, sepal_width, petal_length, petal_width, target_class
    
    df_feast = df[['iris_id', 'event_timestamp', 'sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target_class']]

    print(f"Saving prepared data to {FEAST_SOURCE_PATH}...")
    df_feast.to_parquet(FEAST_SOURCE_PATH, index=False)
    print("Data preparation for Feast complete.")
    print(f"Generated Parquet file schema:\n{df_feast.info()}")
    print(f"First 5 rows of generated Parquet file:\n{df_feast.head()}")

if __name__ == "__main__":
    prepare_data_for_feast() 