# Large Dataset Processing Examples

This directory contains examples that extend the ML pipeline to handle large datasets efficiently. These techniques are essential when working with datasets that don't fit into memory.

## Techniques Demonstrated

1. **Chunked Processing**: Process large files in manageable chunks
2. **Memory-mapped Files**: Access file data without loading everything into RAM
3. **Dask DataFrames**: Process pandas-like operations on larger-than-memory data
4. **Ray Datasets**: Distributed processing across multiple cores/machines
5. **Efficient File Formats**: Converting between formats (CSV, Parquet, HDF5)

## Examples

### Chunked Processing (`chunked_processing.py`)

Shows how to:
- Read large CSV files in chunks using pandas
- Process each chunk independently
- Aggregate results across chunks
- Apply scikit-learn transformations to chunks

### Dask DataFrames (`dask_dataframes.py`)

Demonstrates how to:
- Load data as Dask DataFrames
- Perform complex transformations lazily
- Scale to larger-than-memory datasets
- Execute parallel computations with minimal code changes

### Ray Datasets (`ray_datasets.py`)

Shows how to:
- Create and transform Ray Datasets
- Apply preprocessing to distributed data
- Train ML models on large datasets
- Use Ray's distributed preprocessing capabilities

### File Format Conversion (`file_conversion.py`)

Demonstrates how to:
- Convert CSV files to columnar formats like Parquet and Arrow
- Compare loading times and memory usage between formats
- Configure compression and chunking parameters
- Benchmark query performance on different formats

## Getting Started with Large Datasets

### Download a Sample Large Dataset

```bash
# Download the NYC Taxi dataset (about 2GB raw)
python examples/large_datasets/download_datasets.py
```

### Run the Examples

```bash
# Chunked processing example
python examples/large_datasets/chunked_processing.py

# Dask DataFrames example
python examples/large_datasets/dask_dataframes.py

# Ray Datasets example  
python examples/large_datasets/ray_datasets.py

# File format conversion example
python examples/large_datasets/file_conversion.py
```

## Memory-Efficiency Tips

1. **Use generators** instead of lists when processing large volumes of data
2. **Process data incrementally** rather than all at once
3. **Choose appropriate file formats** - parquet and HDF5 are better than CSV for large data
4. **Use sparse matrices** when dealing with high-dimensional sparse features
5. **Apply feature selection early** in your pipeline to reduce dimensionality
6. **Set `low_memory=True`** in pandas functions when possible
7. **Monitor memory usage** with tools like `memory_profiler` 