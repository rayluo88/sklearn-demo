# ML Pipeline with MLflow and Deployment

This project demonstrates a complete ML workflow from development to deployment:

1. Data loading and preprocessing
2. Model training with MLflow tracking
3. Model versioning in the MLflow Registry
4. Model serving via a REST API
5. CI/CD pipeline for automated deployment

## Project Structure

- `simple_ml_pipeline.py` - Main ML pipeline script
- `model_registry.py` - Registers models in MLflow Model Registry
- `model_api.py` - Flask API for model predictions
- `test_model.py` - Tests for the trained model
- `Dockerfile` - Container for training pipeline
- `Dockerfile.api` - Container for prediction API
- `.github/workflows/model-deployment.yml` - CI/CD workflow

## Getting Started

### Prerequisites

- Python 3.9+
- Docker (for containerized execution)
- Git (optional, for versioning)

### Installation

```bash
# Clone the repository (if you're using Git)
git clone <repository-url>
cd <repository-directory>

# Install dependencies
pip install -r requirements.txt
```

### Running the Pipeline

```bash
# Train a model and log to MLflow
python simple_ml_pipeline.py

# Register the best model in MLflow Registry
python model_registry.py

# Start the MLflow UI to view experiments
mlflow ui
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

## License

This project is licensed under the MIT License - see the LICENSE file for details. 