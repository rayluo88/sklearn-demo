# Tests for ML Pipeline Project

This directory contains tests for the ML Pipeline project.

## Test Structure

- `test_simple_ml_pipeline.py`: Tests for data loading, preprocessing, model training, and basic pipeline functionality
- `test_model_evaluation.py`: Tests for cross-validation, hyperparameter tuning, and model quality evaluation
- `test_data_validation.py`: Tests for data quality, distributions, correlations, and feature importance
- `test_mlflow_integration.py`: Tests for MLflow experiment tracking and model registry integration
- `test_model.py`: Tests for model loading and prediction

## Running Tests

### Running all tests

```bash
# From the project root directory
pytest tests/

# With verbose output
pytest -v tests/

# With test coverage
pytest --cov=. tests/
```

### Running specific test files

```bash
# Run a specific test file
pytest tests/test_simple_ml_pipeline.py

# Run a specific test
pytest tests/test_simple_ml_pipeline.py::TestSimpleMlPipeline::test_data_loading
```

### Test Dependencies

Make sure you have the required dependencies installed:

```bash
pip install pytest pytest-cov
```

## Test Fixtures

Common test fixtures are defined in `conftest.py`:

- `iris_dataset`: Provides the Iris dataset for testing
- `temp_mlflow_tracking`: Sets up a temporary MLflow tracking directory for tests 