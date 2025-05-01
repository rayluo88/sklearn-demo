# Parallel Computing Examples

This directory contains examples of how to enhance the ML pipeline with parallel computing frameworks.

## Overview

Two popular frameworks are demonstrated:

1. **Dask**: A flexible library for parallel computing that integrates well with NumPy, pandas, and scikit-learn
2. **Ray**: A more general-purpose framework for distributed computing with specialized ML capabilities

## Examples

### Dask Integration (`dask_pipeline.py`)

This example shows:
- Setting up a local Dask cluster
- Converting data to Dask arrays
- Using parallel computation for data splitting
- Implementing parallel model prediction

```bash
# Install dependencies
pip install dask distributed dask-ml scikit-learn mlflow

# Run the example
python dask_pipeline.py
```

### Ray Integration (`ray_pipeline.py`)

This example demonstrates two use cases:

1. **Hyperparameter Tuning**: Using Ray Tune to distribute the search for optimal model parameters
2. **Batch Inference**: Distributing prediction workloads across workers

```bash
# Install dependencies
pip install ray==2.38.0 ray[tune]==2.38.0 ray[data]==2.38.0 optuna scikit-learn mlflow

# Run the example
python ray_pipeline.py
```

## Key Differences

| Feature | Dask | Ray |
|---------|------|-----|
| **Focus** | Data parallelism with NumPy/pandas-like API | General distributed computing |
| **ML Support** | Parallelize scikit-learn | Dedicated ML libraries (Tune, Train) |
| **Learning Curve** | Lower (familiar NumPy-like API) | Moderate (requires understanding Ray concepts) |
| **Use Case** | Large datasets, direct scikit-learn integration | Hyperparameter tuning, training, serving |
| **Scaling** | Good for medium-large clusters | Excellent for large clusters |

## Integration with MLflow

Both examples demonstrate MLflow integration:
- Experiment tracking works the same way with distributed models
- Models can be logged to the MLflow Model Registry
- Parallel results are tracked in organized MLflow runs

## When to Use Parallel Computing

Consider these frameworks when:
1. Your dataset is too large for memory
2. Training is taking too long
3. You need to run many model configurations (hyperparameter search)
4. Inference needs to handle large batches quickly

## Note on Ray Versions

Ray's API changes frequently with new releases. The examples in this project use Ray 2.38.0, which may have different API structures compared to other versions. If you encounter issues or use a different Ray version, you may need to adapt the code accordingly. 