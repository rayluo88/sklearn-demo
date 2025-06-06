# ML Pipeline with MLflow, DVC, Feature Store (Feast) and Deployment

This project demonstrates a complete ML workflow from development to deployment:

1. Data loading and preprocessing
2. Feature Engineering and Management with a Feature Store (Feast)
3. Model training with MLflow or Weights & Biases (W&B) tracking
4. Model versioning in MLflow Registry or W&B Artifacts/Model Registry
5. Model serving via a REST API (Flask) with online feature lookup (conceptual)
6. CI/CD pipeline for automated deployment (GitHub Actions)
7. Continuous Model Monitoring

## Project Structure

- `simple_ml_pipeline.py` - Main ML pipeline script (uses Feast for feature retrieval, tracks with MLflow)
- `wandb_ml_pipeline.py` - Alternative ML pipeline script (uses Feast, tracks with Weights & Biases)
- `model_registry.py` - Registers models in MLflow Model Registry
- `model_api.py` - Flask API for model predictions (integrates with Feast and Model Monitoring)
- `model_monitoring.py` - Handles continuous model monitoring (drift, performance)
- `prepare_feast_data.py` - Script to prepare raw data for Feast ingestion.
- `feature_repository/` - Contains Feast feature store definitions:
  - `feature_store.yaml` - Feast store configuration (local provider, Parquet offline, SQLite online).
  - `iris_definitions.py` - Defines Iris entities and feature views for Feast.
  - `__init__.py`
- `data/`
  - `iris.csv` - Raw Iris dataset.
  - `iris_feast_source.parquet` - Processed data for Feast, generated by `prepare_feast_data.py`.
  - `registry.db` - Feast local registry (SQLite DB, created by `feast apply`).
  - `online_store.db` - Feast local online store (SQLite DB, created by `feast materialize`).
- `monitoring_data/` - Stores data related to model monitoring (predictions, feedback, metrics).
- `monitoring_visualizations/` - Stores charts generated by the monitoring system.
- `model_monitoring_report.html` - HTML report from the monitoring system.
- `tests/` - Test directory containing:
  - `test_simple_ml_pipeline.py` - Unit tests for ML pipeline components
  - `test_model_evaluation.py` - Tests for model evaluation and hyperparameter tuning
  - `test_data_validation.py` - Tests for data quality and validation
  - `test_mlflow_integration.py` - Tests for MLflow tracking and registry integration
  - `test_model.py` - Tests for model loading and prediction
  - `conftest.py` - Common test fixtures and utilities
- `Dockerfile` - Container for training pipeline
- `Dockerfile.api` - Container for prediction API
- `.github/workflows/model-deployment.yml` - CI/CD workflow
- `examples/` - Extended examples including:
  - `distributed_training/` - Ray-based distributed model training examples
  - `large_datasets/` - Techniques for processing datasets that don't fit in memory
  - `parallel_computing/` - Parallel computing with Dask and Ray for ML workloads
    - `dask_pipeline.py` - Example of parallel processing with Dask
    - `ray_pipeline.py` - Example of hyperparameter tuning and batch inference with Ray (compatible with Ray 2.46.0)

## Getting Started

### Prerequisites

- Python 3.9+ (preferably managed with Conda)
- Conda (for environment management)
- Docker (for containerized execution)
- Git (optional, for versioning)
- DVC initialized in the project, with `data/iris.csv` (or a similar dataset) tracked.
- For the MLflow pipeline (`simple_ml_pipeline.py`): MLflow Tracking Server running (e.g., `mlflow ui --host 0.0.0.0`)
- For the Weights & Biases pipeline (`wandb_ml_pipeline.py`):
    - `wandb` library installed (`pip install wandb`).
    - Logged into W&B (run `wandb login` in your terminal and follow instructions).

### Installation

```bash
# Clone the repository (if you're using Git)
git clone <repository-url>
cd <repository-directory>

# Install dependencies
# Consider adding 'wandb' to your requirements.txt if you plan to use the W&B pipeline
pip install -r requirements.txt
# If wandb is not in requirements.txt, install it separately:
# pip install wandb
```

#### Conda Environment Setup (Recommended)

It's highly recommended to use a Conda environment to manage dependencies for this project.

1.  **Create the Conda environment:**
    ```bash
    conda create --name sklearn_demo_env python=3.12 -y
    ```

2.  **Activate the environment:**
    ```bash
    conda activate sklearn_demo_env
    ```
    Your terminal prompt should now start with `(sklearn_demo_env)`.

3.  **Install dependencies into the Conda environment:**
    ```bash
    # Ensure requirements.txt includes mlflow, feast, scikit-learn, pandas, etc.
    # Add 'wandb' if you intend to use the Weights & Biases pipeline.
    pip install -r requirements.txt
    # Or, if not in requirements.txt: pip install wandb
    ```

### Data Setup

Ensure the DVC-tracked data is available. If you have already configured DVC and added `data/iris.csv`:
```bash
# Pull the latest version of the data tracked by DVC
dvc pull
```
If `data/iris.csv` is not found after running the pipeline, this command should fetch it, assuming it has been previously added and committed to DVC storage.

### Feature Store Setup (Feast)

Before running the main ML pipeline, you need to set up the Feast feature store:

1.  **Prepare Data for Feast:**
    Run the script to convert `iris.csv` to the Parquet format Feast will use as a source. This adds necessary `iris_id` and `event_timestamp` columns.
    ```bash
    python prepare_feast_data.py
    ```
    This will create `data/iris_feast_source.parquet`.

2.  **Initialize and Apply Feature Definitions:**
    Navigate to the feature repository directory and apply the feature definitions. This registers your features with Feast and creates the `registry.db`.
    ```bash
    cd feature_repository/
    feast apply
    cd .. 
    ```

3.  **Materialize Features to Online Store:**
    Load features from your offline store (the Parquet file) into the online store (SQLite DB). This is necessary for the API to perform online feature lookups.
    ```bash
    cd feature_repository/
    # Materialize features into the online store.
    # For the initial load, you need to specify a start and end time that covers your data.
    # Your 'prepare_feast_data.py' script generates timestamps going back from the current date.
    # For the Iris dataset (150 rows), data spans ~150 days.
    # Example: Materialize from 200 days ago until tomorrow.
    # Adjust these dates based on when you ran 'prepare_feast_data.py'.
    FEAST_START_DATE=$(date -u -d "200 days ago" +"%Y-%m-%dT%H:%M:%S")
    FEAST_END_DATE=$(date -u -d "1 day" +"%Y-%m-%dT%H:%M:%S")
    echo "Materializing from $FEAST_START_DATE to $FEAST_END_DATE"
    feast materialize $FEAST_START_DATE $FEAST_END_DATE

    # For ongoing incremental materialization (after the initial load):
    # feast materialize-incremental $(date -u +"%Y-%m-%dT%H:%M:%S")
    cd ..
    ```
    This creates `data/online_store.db`.

### Running the Pipeline

This section describes running the original MLflow-based pipeline. See the next section for Weights & Biases.

With the feature store set up and MLflow server running:

```bash
# Train a model using features from Feast and log to MLflow
python simple_ml_pipeline.py

# Register the best model in MLflow Registry (if not done by pipeline)
# python model_registry.py # The pipeline now registers the model directly

# Start the MLflow UI to view experiments (if not already running)
# mlflow ui --host 0.0.0.0
```

### Running the Pipeline with Weights & Biases

This project includes an alternative pipeline script, `wandb_ml_pipeline.py`, that uses [Weights & Biases (W&B)](https://wandb.ai) for experiment tracking, model versioning, and visualization.

**Key Features of the W&B Pipeline:**
- Connects to the same Feast feature store for feature management
- Uses RandomForestClassifier with GridSearchCV for hyperparameter optimization
- Logs metrics, parameters, model artifacts, and training data samples to W&B
- Supports complete experiment tracking and result visualization

**Prerequisites for W&B:**
1.  **Install W&B Client**:
    ```bash
    pip install wandb
    ```
    (Ensure this is added to your `requirements.txt` if you use it regularly).
2.  **Login to W&B**:
    You'll need a free W&B account. Run the following in your terminal and follow the prompts to authenticate:
    ```bash
    wandb login
    ```

**Execution:**

Ensure your Feast feature store is set up as described in the "Feature Store Setup (Feast)" section (i.e., `prepare_feast_data.py` has been run, and `feast apply` & `feast materialize` have been executed in the `feature_repository/` directory).

Then, run the W&B pipeline script:
```bash
# Train a model using features from Feast and log to Weights & Biases
python wandb_ml_pipeline.py
```
This will execute the pipeline, and all configurations, metrics, and the trained model (as an artifact) will be logged to your W&B project (default: `iris-feast-wandb-demo`). You can view and analyze your runs on the W&B dashboard.

**Benefits of the W&B Pipeline:**
- Rich visualization dashboard for experiment tracking
- Easy model versioning and comparison
- Automatic hyperparameter visualization from GridSearchCV
- Collaborative features for team settings
- Integrated model registry

### Testing the ML Components

The project includes comprehensive tests for different aspects of the ML pipeline:

```bash
# Run all tests with pytest
pytest tests/

# Run all tests with unittest
python -m unittest discover tests

# Run specific test modules with pytest
pytest tests/test_simple_ml_pipeline.py    # Basic pipeline components
pytest tests/test_model_evaluation.py      # Model evaluation and tuning
pytest tests/test_data_validation.py       # Data quality checks
pytest tests/test_mlflow_integration.py    # MLflow integration
pytest tests/test_model.py                 # Model loading and predictions
```

#### Test Coverage

- **Pipeline Tests**: Test data loading, preprocessing, model training, and MLflow logging
- **Model Evaluation**: Test cross-validation, hyperparameter tuning, and feature importance
- **Data Validation**: Test data quality, distributions, correlations, and feature relevance
- **MLflow Integration**: Test experiment tracking, model registry, and model loading

Running the following test scripts directly will generate visualizations:

```bash
# Generate evaluation visualizations
python tests/test_model_evaluation.py
python tests/test_data_validation.py
```

### Serving Predictions (with Feast integration)

Ensure the model is trained, registered, and features are materialized in Feast's online store.

```bash
# Start the Flask API server
python model_api.py

# Or with Gunicorn (production)
gunicorn --bind 0.0.0.0:5001 model_api:app
```

### Making API Requests

```bash
# Example prediction request (API uses these raw features for prediction)
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [[5.1, 3.5, 1.4, 0.2]]}'
```

### Model Monitoring

The API integrates with `model_monitoring.py`. Predictions and feedback are logged, and a monitoring dashboard is available:

-   Access the monitoring report: `http://localhost:5001/monitoring`
-   Submit feedback via the UI or the `/feedback` endpoint.
-   Optionally trigger retraining via `/trigger-retraining` (POST request, e.g., with curl `curl -X POST http://localhost:5001/trigger-retraining`).

## CI/CD Pipeline

The GitHub Actions workflow automates:
1. Model training and testing (currently configured for the MLflow pipeline)
2. Model registration in MLflow Registry
3. Building and deploying the prediction API

## Model Versioning

The project uses MLflow Model Registry by default for versioning, with stages:
- Staging: Models under evaluation
- Production: Models deployed to production
- Archived: Previous model versions

Alternatively, if using the `wandb_ml_pipeline.py` script, Weights & Biases (W&B) is used. W&B Artifacts provide robust model versioning, and W&B also offers a Model Registry for managing model lifecycles. The `wandb_ml_pipeline.py` script logs the trained model as a W&B artifact.

## Feature Store (Feast)

This project uses Feast for feature management:
-   **Definitions**: Features (`sepal_length`, `petal_width`, etc.) and entities (`iris_id`) are defined in `feature_repository/iris_definitions.py`.
-   **Offline Store**: A Parquet file (`data/iris_feast_source.parquet`) serves as the offline data source.
-   **Online Store**: SQLite (`data/online_store.db`) is used for low-latency access during model serving.
-   **Workflow**:
    1.  `prepare_feast_data.py`: Processes raw data into the format Feast expects.
    2.  `feast apply` (in `feature_repository/`): Registers feature definitions.
    3.  `simple_ml_pipeline.py` or `wandb_ml_pipeline.py`: Retrieves historical features from Feast for model training.
    4.  `feast materialize-incremental <date>` (in `feature_repository/`): Loads features into the online store.
    5.  `model_api.py`: (Conceptually) retrieves online features for predictions.

## Examples Directory

The project includes several example directories showing advanced ML techniques:

### Parallel Computing with Dask and Ray

This project demonstrates how to use distributed computing frameworks like Dask and Ray to handle larger datasets and more computationally intensive ML tasks. The `examples/parallel_computing/` directory contains complete, runnable scripts:

*   **Dask Integration (`examples/parallel_computing/dask_pipeline.py`)**: Illustrates parallel data processing and model training with Dask.
*   **Ray Integration (`examples/parallel_computing/ray_pipeline.py`)**: Shows distributed hyperparameter tuning and batch inference with Ray (compatible with Ray 2.46.0).

These examples showcase techniques for:
1.  **Large Datasets**: Processing data that doesn't fit in memory on a single machine.
2.  **Hyperparameter Tuning**: Running multiple model configurations in parallel.
3. **Feature Engineering**: Distributing complex feature transformations.
4. **Cross-Validation**: Parallelizing k-fold validation for faster execution.

To integrate these frameworks into your own ML pipelines:

### Dask Integration
[Dask](https://dask.org/) provides parallel computing capabilities that integrate well with scikit-learn.

1.  Install Dask and related packages:
    ```bash
    pip install dask distributed dask-ml scikit-learn
    ```
2.  Modify your pipeline to use Dask's parallel capabilities. For example:
    ```python
    # Example: Using Dask with scikit-learn
    from dask.distributed import Client
    import dask.array as da
    from dask_ml.model_selection import train_test_split as dask_train_test_split
    from dask_ml.wrappers import ParallelPostFit
    from sklearn.neighbors import KNeighborsClassifier # Or any other sklearn estimator

    # Set up Dask client (local or connect to a cluster)
    client = Client() 

    # Load and distribute data with Dask (e.g., for large_dataset)
    # X = da.from_array(large_dataset, chunks="auto") 

    # Use a parallel model wrapper with a scikit-learn estimator
    parallel_model = ParallelPostFit(KNeighborsClassifier(n_neighbors=3))
    
    # MLflow tracking works the same way with these Dask-wrapped models.
    ```
    Refer to `examples/parallel_computing/dask_pipeline.py` for a full example.

### Ray Integration
[Ray](https://ray.io/) is a Python-first distributed computing framework. The examples in this project are compatible with Ray 2.46.0.

1.  Install Ray (with extras for data, tune, and train):
    ```bash
    pip install "ray[data,tune,train]==2.46.0" scikit-learn
    ```
2.  Modify your pipeline for distributed model training or hyperparameter tuning. For Ray 2.46.0 and newer, using Ray's joblib backend for scikit-learn integration is often recommended for hyperparameter tuning, as shown in `examples/parallel_computing/ray_pipeline.py`.
    ```python
    # Example: Using Ray's joblib backend for scikit-learn (from ray_pipeline.py)
    import ray
    from ray.util.joblib import register_ray
    from sklearn.model_selection import GridSearchCV
    from sklearn.neighbors import KNeighborsClassifier # Or any other sklearn estimator
    
    ray.init()
    register_ray() # Registers Ray as the joblib backend

    # Your GridSearchCV (or similar scikit-learn operation) will now use Ray for parallelism
    # param_grid = {"n_neighbors": [1, 5, 10]}
    # grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=3, n_jobs=-1) # n_jobs=-1 uses all Ray workers
    # grid_search.fit(X_train, y_train)
    
    # For distributed training, refer to examples in examples/distributed_training/
    ```
    Refer to `examples/parallel_computing/ray_pipeline.py` for a complete hyperparameter tuning example and `examples/distributed_training/` for distributed training examples with Ray.

See `examples/parallel_computing/` and `examples/distributed_training/` directories for more detailed implementation examples and their respective READMEs.

### Distributed Training (`examples/distributed_training/`)

Examples of training models in a distributed fashion:

- **Basic Distributed Training** (`ray_distributed_training.py`): Ensemble training across multiple workers
- **Large Dataset Training** (`ray_large_dataset_training.py`): Distributed training with NYC Taxi dataset

```bash
# Run basic distributed training example
python examples/distributed_training/ray_distributed_training.py

# Run large dataset distributed training
python examples/distributed_training/ray_large_dataset_training.py --sample 200000 --partitions 8
```

### Large Dataset Processing (`examples/large_datasets/`)

Examples of handling datasets that don't fit in memory:

- **Chunked Processing** (`chunked_processing.py`): Process large CSV files in memory-efficient chunks
- **Dask DataFrames** (`dask_dataframes.py`): Process pandas-like operations on larger-than-memory data
- **Ray Datasets** (`ray_datasets.py`): Distributed data processing with Ray
- **File Format Conversion** (`file_conversion.py`): Convert between file formats for better performance

```bash
# Download sample datasets
python examples/large_datasets/download_datasets.py

# Run chunked processing example
python examples/large_datasets/chunked_processing.py
```

Each example directory contains its own detailed README with more information.

## Large Dataset Processing

The project also includes examples for handling large datasets efficiently in the `examples/large_datasets` directory:

1. **Chunked Processing**: Process large files in manageable chunks
2. **Memory-mapped Files**: Access file data without loading everything into RAM
3. **Dask DataFrames**: Process pandas-like operations on larger-than-memory data
4. **Ray Datasets**: Distributed processing across multiple cores/machines
5. **Efficient File Formats**: Converting between formats (CSV, Parquet, HDF5)

## License

This project is licensed under the MIT License - see the LICENSE file for details. 