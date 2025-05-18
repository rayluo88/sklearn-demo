# ML Pipeline with MLflow and Deployment

This project demonstrates a complete ML workflow from development to deployment:

1. Data loading and preprocessing
2. Model training with MLflow tracking
3. Model versioning in the MLflow Registry
4. Model serving via a REST API
5. CI/CD pipeline for automated deployment

## Project Structure

- `simple_ml_pipeline.py` - Main ML pipeline script (uses DVC-tracked data from `data/iris.csv`)
- `model_registry.py` - Registers models in MLflow Model Registry
- `model_api.py` - Flask API for model predictions
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

## Getting Started

### Prerequisites

- Python 3.9+
- Docker (for containerized execution)
- Git (optional, for versioning)
- DVC initialized in the project, with `data/iris.csv` (or a similar dataset) tracked.

### Installation

```bash
# Clone the repository (if you're using Git)
git clone <repository-url>
cd <repository-directory>

# Install dependencies
pip install -r requirements.txt
```

### Data Setup

Ensure the DVC-tracked data is available. If you have already configured DVC and added `data/iris.csv`:
```bash
# Pull the latest version of the data tracked by DVC
dvc pull
```
If `data/iris.csv` is not found after running the pipeline, this command should fetch it, assuming it has been previously added and committed to DVC storage.

### Running the Pipeline

```bash
# Train a model and log to MLflow
python simple_ml_pipeline.py

# Register the best model in MLflow Registry
python model_registry.py

# Start the MLflow UI to view experiments
mlflow ui
```

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

### Serving Predictions

```bash
# Start the Flask API server
python model_api.py

# Or with Gunicorn (production)
gunicorn --bind 0.0.0.0:5001 model_api:app
```

### Docker Containers

```bash
# Build and run training container
docker build -t sklearn-mlflow-app .
docker run --rm -v "$(pwd)/mlruns:/app/mlruns" sklearn-mlflow-app

# Build and run API container
docker build -t iris-prediction-api -f Dockerfile.api .
docker run -p 5001:5001 iris-prediction-api
```

### Making API Requests

```bash
# Example prediction request
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [[5.1, 3.5, 1.4, 0.2]]}'
```

## CI/CD Pipeline

The GitHub Actions workflow automates:
1. Model training and testing
2. Model registration in MLflow Registry
3. Building and deploying the prediction API

## Model Versioning

The project uses MLflow Model Registry for versioning, with stages:
- Staging: Models under evaluation
- Production: Models deployed to production
- Archived: Previous model versions

## Examples Directory

The project includes several example directories showing advanced ML techniques:

### Parallel Computing (`examples/parallel_computing/`)

Examples of enhancing the ML pipeline with parallel computing frameworks:

- **Dask Integration** (`dask_pipeline.py`): Parallel data processing and model training with Dask
- **Ray Integration** (`ray_pipeline.py`): Distributed hyperparameter tuning and batch inference with Ray

```bash
# Run the Dask example
python examples/parallel_computing/dask_pipeline.py

# Run the Ray example
python examples/parallel_computing/ray_pipeline.py
```

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

## Parallel Computing with Dask or Ray

This project can be extended to use distributed computing frameworks for handling larger datasets and more computationally intensive models:

### Integration with Dask

[Dask](https://dask.org/) provides parallel computing capabilities that integrate well with scikit-learn. To add Dask:

1. Install Dask and related packages:
   ```bash
   pip install dask distributed dask-ml scikit-learn
   ```

2. Modify `simple_ml_pipeline.py` to use Dask's parallel capabilities:
   ```python
   # Example code for using Dask with scikit-learn
   from dask.distributed import Client
   import dask.array as da
   from dask_ml.model_selection import train_test_split as dask_train_test_split
   from dask_ml.wrappers import ParallelPostFit

   # Set up Dask client
   client = Client()  # For local cluster
   # client = Client("scheduler-address")  # For connecting to existing cluster

   # Load and distribute data with Dask
   # For larger datasets, you could use:
   # X = da.from_array(large_dataset, chunks="auto")

   # Use parallel model with scikit-learn estimator
   from sklearn.neighbors import KNeighborsClassifier
   parallel_model = ParallelPostFit(KNeighborsClassifier(n_neighbors=3))
   
   # MLflow tracking works the same way with these models
   ```

### Integration with Ray

[Ray](https://ray.io/) provides a Python-first distributed computing framework. To integrate Ray:

1. Install Ray and related packages:
   ```bash
   pip install ray ray[tune] ray[air] scikit-learn
   ```

2. Modify the pipeline for distributed model training:
   ```python
   # Example code for using Ray with scikit-learn
   import ray
   from ray.air import session
   from ray.air.config import ScalingConfig
   from ray.train.sklearn import SklearnTrainer
   
   ray.init()
   
   # Define model training function
   def train_model_func():
       from sklearn.neighbors import KNeighborsClassifier
       model = KNeighborsClassifier(n_neighbors=3)
       # ... load data and train model ...
       return model
   
   # Use Ray to distribute training
   trainer = SklearnTrainer(
       train_loop_per_worker=train_model_func,
       scaling_config=ScalingConfig(num_workers=4),
   )
   result = trainer.fit()
   
   # Retrieve and log model with MLflow as usual
   ```

### Distributed Training Examples

The project now includes complete examples of distributed model training in the `examples/distributed_training` directory:

1. **Basic Distributed Training**: `ray_distributed_training.py` demonstrates how to:
   - Distribute model training across multiple workers
   - Create an ensemble from distributed models
   - Compare performance with regular training

2. **Large Dataset Training**: `ray_large_dataset_training.py` shows how to:
   - Process larger-than-memory datasets
   - Distribute the training across a Ray cluster
   - Track distributed experiments with MLflow

Run the distributed training examples:

```bash
# Basic distributed training example
python examples/distributed_training/ray_distributed_training.py

# Large dataset training (downloads NYC Taxi dataset if needed)
python examples/distributed_training/ray_large_dataset_training.py --sample 200000 --partitions 8
```

To run on a Ray cluster:
```bash
# Start a Ray cluster in one terminal
ray start --head

# Connect to the local Ray cluster
python examples/distributed_training/ray_distributed_training.py --address="auto"
```

### Performance Benefits

Integrating these frameworks is particularly useful for:

1. **Large Datasets**: Process data that doesn't fit in memory on a single machine
2. **Hyperparameter Tuning**: Run multiple model configurations in parallel
3. **Feature Engineering**: Distribute complex feature transformations
4. **Cross-Validation**: Parallelize k-fold validation for faster execution

See `examples/parallel_computing` and `examples/distributed_training` directories for complete implementation examples.

## Large Dataset Processing

The project also includes examples for handling large datasets efficiently in the `examples/large_datasets` directory:

1. **Chunked Processing**: Process large files in manageable chunks
2. **Memory-mapped Files**: Access file data without loading everything into RAM
3. **Dask DataFrames**: Process pandas-like operations on larger-than-memory data
4. **Ray Datasets**: Distributed processing across multiple cores/machines
5. **Efficient File Formats**: Converting between formats (CSV, Parquet, HDF5)

## License

This project is licensed under the MIT License - see the LICENSE file for details. 