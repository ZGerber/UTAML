# UTAML

A toolkit for processing and analyzing Telescope Array (TA) cosmic ray physics data, with a focus on machine learning preparation and modern data analysis workflows.

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

⚠️ **Note: This project is under active development. Many features may not be fully implemented or tested.**

## Overview

This toolkit provides three main utilities:
- `root2parquet`: Convert ROOT files to Parquet format
- `parquet-prep`: Process Parquet files for machine learning applications
- `parquet-split`: Split large Parquet datasets with memory efficiency

## Installation

Clone the repository:
```bash
git clone https://github.com/ZGerber/UTAML.git
cd UTAML
```

```bash
# Using pip
pip install .

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

⚠️ **Important:** If your input data contains custom branches, it's highly recommended to explicitly specify which columns you want to convert.
Processing an entire ROOT tree without column selection can lead to:
- Excessive memory usage
- Slow processing times
- Potential issues with complex branch structures
- Unnecessary data conversion

Example with column selection:
```bash
# Good: explicitly specify needed columns
root2parquet input.root --tree_name taTree \
    energy theta psi --compression ZSTD -j 4

# Not recommended: converting entire tree
root2parquet input.root --tree_name taTree  # may cause issues if 'taTree' is too complicated.
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
    --scale "theta:standard" "energy:minmax"
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

### Understanding File Formats

#### Parquet

Parquet is a columnar storage file format optimized for use with big data processing frameworks. It offers several advantages:

- **Efficient Compression**: Parquet files are highly compressed, reducing storage requirements and improving I/O performance.
- **Columnar Storage**: This format allows for efficient reading of specific columns, which is ideal for analytical queries and machine learning workflows.
- **Schema Evolution**: Parquet supports schema evolution, allowing you to add new columns without breaking existing data.
- **Interoperability**: Widely supported by modern data processing tools like Apache Spark, Pandas, and Dask.

#### Comparison with Other Formats

- **ROOT**: 
  - Designed for high-energy physics data, ROOT files are complex and can be difficult for newcomers to use.
  - ROOT provides powerful data analysis capabilities but requires learning its unique data structures and C++-based language.
  - Parquet offers a more accessible, Python-friendly alternative for data analysis and machine learning.

- **NPZ**:
  - NPZ is a simple, compressed archive format for storing NumPy arrays.
  - It is easy to use with Python but lacks the advanced features of Parquet, such as efficient columnar storage and schema evolution.
  - NPZ is suitable for smaller datasets and quick prototyping but may not scale well for large datasets.

- **CSV**:
  - CSV is a plain text format that is easy to read and write.
  - It is widely supported and can be opened in spreadsheet applications.
  - However, CSV lacks support for complex data types and is inefficient for large datasets.

- **HDF5**:
  - HDF5 is a versatile format that supports large, complex datasets and hierarchical data structures.
  - It is well-suited for scientific data and supports parallel I/O.
  - HDF5 can be more complex to use and requires additional libraries for full functionality.

#### File Format Comparison Table

| Format | Compression | Columnar | Schema Evolution | Complexity | Best Use Case |
|--------|-------------|----------|------------------|------------|---------------|
| Parquet | Yes         | Yes      | Yes              | Moderate   | Large-scale data processing, ML |
| ROOT   | Yes         | No       | Limited          | High       | High-energy physics analysis |
| NPZ    | Yes         | No       | No               | Low        | Small datasets, quick prototyping |
| CSV    | No          | No       | No               | Low        | Simple data exchange, spreadsheets |
| HDF5   | Yes         | No       | Yes              | High       | Scientific data, hierarchical data |

#### When to Use Each Format

- **Parquet**: Best for large-scale data processing, machine learning, and scenarios where efficient columnar access is needed.
- **ROOT**: Ideal for traditional high-energy physics analysis where ROOT's specialized tools are required.
- **NPZ**: Useful for small to medium-sized datasets, quick data sharing, and when working exclusively within Python.
- **CSV**: Suitable for simple data exchange and when compatibility with spreadsheet software is needed.
- **HDF5**: Best for complex scientific data and when hierarchical data structures are required.

By converting ROOT files to Parquet, UTAML enables users to leverage modern data science tools and workflows, making cosmic ray data more accessible and easier to analyze.

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
    "theta:abs(theta):abs_theta"
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

- **awkward (≥2.6.10)**: A library for nested, variable-sized data, enabling efficient manipulation of complex data structures.
- **pyarrow (≥10.0)**: Provides a Python interface to the Apache Arrow project, enabling efficient in-memory columnar data storage and fast data interchange between systems.
- **uproot (≥5.0)**: A library for reading and writing ROOT files in pure Python, allowing seamless access to high-energy physics data without the need for the ROOT framework.
- **numpy (≥1.21)**: A fundamental package for scientific computing in Python, providing support for large, multi-dimensional arrays and matrices.
- **colorlog (≥6.7)**: Enhances Python's logging module with colored output, improving readability of log messages.
- **psutil (≥5.9)**: Provides an interface for retrieving information on system utilization (CPU, memory, disks, network, sensors) and system uptime.

Optional:
- h5py (for HDF5 support)

### Key Packages Explained

- **uproot**: 
  - Uproot is a pure Python package that allows you to read and write ROOT files without needing the ROOT framework. 
  - It is particularly useful for high-energy physics applications, where ROOT is a standard format for data storage.
  - Uproot provides a Pythonic interface to access and manipulate ROOT data, making it easier to integrate with modern data science workflows.

- **pyarrow**:
  - PyArrow is part of the Apache Arrow project, which provides a language-independent columnar memory format for flat and hierarchical data.
  - It enables efficient data interchange between different systems and is optimized for performance, making it ideal for big data applications.
  - PyArrow is used in this project to handle Parquet files, allowing for efficient reading and writing of columnar data.

## Known Issues

- Large file processing may require significant memory despite optimizations
- Some scaling operations may not handle NaN values gracefully
- Parallel processing can occasionally lead to deadlocks with very large datasets
- Not all combinations of options have been thoroughly tested


## Author

Created by Zane Gerber ([ZGerber](https://github.com/ZGerber))

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License
