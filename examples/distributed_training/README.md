# Distributed Training Examples

This directory contains examples that demonstrate distributed training techniques for machine learning models. Distributed training allows you to scale model training across multiple machines or cores, reducing training time and enabling training on larger datasets.

## Techniques Demonstrated

1. **Ray-based Distributed Training**: Train models in parallel across multiple workers
2. **Model Ensembling**: Combine distributed models into a powerful ensemble
3. **Performance Comparison**: Compare distributed vs. non-distributed approaches
4. **MLflow Integration**: Track experiments, metrics, and models

## Getting Started

### Prerequisites

To run the distributed training examples, you need:

```bash
pip install ray sklearn pandas numpy matplotlib mlflow
```

These dependencies should already be included in your project's `requirements.txt`.

### Running Locally (Single Machine)

The simplest way to run the distributed training example is in local mode:

```bash
python examples/distributed_training/ray_distributed_training.py
```

This will:
1. Initialize Ray on your local machine
2. Partition the training data
3. Train models in parallel across CPU cores
4. Ensemble the results
5. Compare with non-distributed training
6. Log everything to MLflow

### Running on a Ray Cluster

For true distributed training, you can connect to a Ray cluster:

```bash
# Start a Ray cluster (in a separate terminal)
ray start --head

# Connect to the local Ray cluster
python examples/distributed_training/ray_distributed_training.py --address="auto"

# To specify number of partitions/workers
python examples/distributed_training/ray_distributed_training.py --partitions=8
```

## How It Works

1. **Data Partitioning**: The training data is split into N partitions
2. **Parallel Training**: Ray spawns N workers, each training a model on one partition
3. **Model Combination**: Individual models are combined into a voting ensemble
4. **Evaluation**: The ensemble is compared against a single model trained on all data

## Benefits of Distributed Training

1. **Speed**: Training completes faster by utilizing multiple CPU cores or machines
2. **Scalability**: Can handle larger datasets by distributing the workload
3. **Ensemble Effects**: Often achieves better performance through ensemble learning
4. **Resource Utilization**: Makes better use of available computing resources

## Extending the Example

You can extend this example to:

1. Train on a real distributed cluster
2. Use different base models (SVMs, neural networks, etc.)
3. Implement more sophisticated ensembling techniques
4. Add distributed hyperparameter tuning
5. Scale to very large datasets 