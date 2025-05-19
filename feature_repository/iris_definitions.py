from google.protobuf.duration_pb2 import Duration
from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.types import Float32, Int64

# Define an entity for the Iris flower sample. In a real scenario, this would be a unique identifier.
iris_flower = Entity(name="iris_id", value_type=ValueType.INT64, description="Unique ID for an Iris flower sample")

# Define the source of the Iris data
# This will point to a Parquet file generated from iris.csv, with added 'event_timestamp' and 'iris_id'
iris_data_source = FileSource(
    name="iris_source",
    path="../data/iris_feast_source.parquet",  # Path relative to feature_store.yaml
    timestamp_field="event_timestamp",
    description="Iris dataset with event timestamps and IDs for Feast.",
)

# Define a FeatureView for the Iris features
iris_feature_view = FeatureView(
    name="iris_features",
    entities=[iris_flower],
    ttl=timedelta(days=7),  # Changed to timedelta
    schema=[
        Field(name="sepal_length", dtype=Float32),
        Field(name="sepal_width", dtype=Float32),
        Field(name="petal_length", dtype=Float32),
        Field(name="petal_width", dtype=Float32),
        Field(name="target_class", dtype=Int64)  # Including target here for simplicity in training data retrieval
    ],
    online=True, # Make features available in the online store
    source=iris_data_source,
    tags={"dataset": "iris"},
) 