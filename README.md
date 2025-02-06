# UTAML

A toolkit for processing and analyzing Telescope Array (TA) cosmic ray physics data, with a focus on machine learning preparation and modern data analysis workflows.

## Author

Created by Zane Gerber ([ZGerber](https://github.com/ZGerber))

This project provides a bridge between legacy ROOT-based data formats and modern data science tools, enabling both ML research and physics analysis. It features:

- Efficient conversion of TA ROOT files to more accessible formats (Parquet/NPZ)
- Memory-optimized processing of large cosmic ray datasets
- ML-focused data preparation pipelines
- Tools for both experienced physicists and newcomers to cosmic ray physics

The toolkit is designed to lower the barrier to entry for TA data analysis by:
- Eliminating the need to learn complex DST and ROOT data structures
- Providing pythonic access to cosmic ray data
- Enabling the use of modern data science libraries (pandas, numpy, scikit-learn, etc.)
- Supporting both quick exploratory analysis and production ML pipelines

⚠️ **Note: This project is under active development. Some features may not be fully tested.**

## Overview

This toolkit provides three main utilities:
- `root2parquet`: Convert ROOT files to Parquet format
- `parquet-prep`: Process Parquet files for machine learning applications
- `parquet-split`: Split large Parquet datasets with memory efficiency

## Installation

```bash
# Using pip
pip install utaml

# Optional HDF5 support
pip install utaml[hdf5]
```

## Tools

### 1. root2parquet

Convert ROOT files to Parquet format with parallel processing and memory optimization.

```bash
root2parquet input.root --tree_name taTree -o output_dir \
    --compression ZSTD --row_group_size 10000 -j 4 \
    pt eta phi m tau_pt tau_eta  # specify desired columns
```

Key features:
- Parallel processing with configurable workers
- Memory-efficient chunked processing
- Selective column reading (recommended to avoid memory issues)
- Multiple compression options (SNAPPY, GZIP, BROTLI, ZSTD)
- Progress tracking and resource monitoring
- Configurable row group sizes

⚠️ **Important:** It's highly recommended to explicitly specify which columns you want to convert. 
Processing an entire ROOT tree without column selection can lead to:
- Excessive memory usage
- Slow processing times
- Potential issues with complex branch structures
- Unnecessary data conversion

Example with column selection:
```bash
# Good: explicitly specify needed columns
root2parquet input.root --tree_name taTree \
    pt eta phi m tau_pt tau_eta \
    --compression ZSTD -j 4

# Not recommended: converting entire tree
root2parquet input.root --tree_name taTree  # may cause issues
```

### 2. parquet-prep

Process Parquet files for machine learning applications with various data transformations.

```bash
# Basic usage
parquet-prep input.parquet -o output.parquet

# Apply filters and transformations
parquet-prep input.parquet \
    -f "pt>20" "abs(eta)<2.4" \
    --features "theta:np.sin(theta):sin_theta" \
    --scale "pt:standard" "eta:minmax"
```

Features:
- Event filtering with Python expressions
- Feature engineering
- Jagged array processing
- Column scaling (standard, minmax)
- Column renaming
- Multiple output formats (parquet, npz)

### 3. parquet-split

Split large Parquet datasets with memory efficiency.

```bash
parquet-split input.parquet output_dir \
    --split-proportions 0.8 0.2 \
    --shuffle --seed 42 \
    --output-format parquet
```

Features:
- Memory-efficient streaming splits
- Row group-based splitting
- Column-based splitting
- Configurable split proportions
- Optional shuffling
- Multiple output formats

## Advanced Usage

### Memory Management

All tools include memory monitoring and protection:

```bash
# Set maximum memory usage (80% of system RAM)
parquet-split input.parquet output_dir --max-memory-usage 0.8

# Control batch size for processing
root2parquet input.root -o output --chunk_size 5000
```

### Parallel Processing

```bash
# Specify number of worker processes
root2parquet input.root -j 8
parquet-split input.parquet output_dir --num-workers 4
```

### Custom Feature Engineering

```bash
parquet-prep input.parquet --features \
    "theta:np.sin(theta):sin_theta" \
    "pt:np.log(pt):log_pt" \
    "eta:abs(eta):abs_eta"
```

### Jagged Array Processing

```bash
parquet-prep input.parquet \
    --max-length 10 \
    --pad-value -999
```

## Dependencies

Core dependencies:
- awkward (≥2.6.10)
- pyarrow (≥10.0)
- uproot (≥5.0)
- numpy (≥1.21)
- colorlog (≥6.7)
- psutil (≥5.9)

Optional:
- h5py (for HDF5 support)

## Known Issues

- Large file processing may require significant memory despite optimizations
- Some scaling operations may not handle NaN values gracefully
- Parallel processing can occasionally lead to deadlocks with very large datasets
- Not all combinations of options have been thoroughly tested

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License
