import argparse
import awkward as ak
import pyarrow.parquet as pq
import pyarrow as pa
import logging
import colorlog


def setup_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    formatter = colorlog.ColoredFormatter(
        '%(log_color)s%(levelname)-8s%(reset)s %(white)s%(message)s',
        datefmt=None,
        reset=True,
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'bold_red',
        }
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


logger = setup_logger()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process TA data in Parquet format with multiple transformation options"
    )
    parser.add_argument("input_file", help="Input Parquet file")
    parser.add_argument("-o", "--output", default="processed_data.parquet",
                        help="Output file name (default: processed_data.parquet)")
    parser.add_argument("--filter", type=str,
                        help="Filter expression (e.g., 'Energy > 50 & NHits > 3')")
    parser.add_argument("--add-features", nargs="+",
                        help="New features to add from jagged columns (e.g., 'MIP0:mean,max')")
    parser.add_argument("--truncate", nargs="+", type=lambda x: x.split(":"),
                        help="Truncate/pad jagged columns (e.g., 'MIP0:100' for max 100 elements)")
    parser.add_argument("--scale", nargs="+", type=lambda x: x.split(":"),
                        help="Scale numeric columns (e.g., 'Energy:zscore' or 'Xmax:minmax')")
    parser.add_argument("--rename", nargs="+", type=lambda x: x.split(":"),
                        help="Rename columns (e.g., 'Energy_mc:true_energy')")
    parser.add_argument("--format", choices=["parquet", "npz", "hdf5"], default="parquet",
                        help="Output format (default: parquet)")
    return parser.parse_args()


def process_data(data, args):
    """Apply all requested transformations to the data."""

    # 1. Filter events
    if args.filter:
        mask = eval(args.filter, {"ak": ak}, data)
        data = data[mask]

    # 2. Process jagged arrays
    if args.truncate:
        for col_spec in args.truncate:
            col, max_len = col_spec
            max_len = int(max_len)
            padded = ak.pad_none(data[col], max_len, clip=True)
            data[col] = ak.fill_none(padded, 0)

    # 3. Add derived features
    if args.add_features:
        for feature_spec in args.add_features:
            col, ops = feature_spec.split(":")
            jagged = data[col]
            for op in ops.split(","):
                if op == "mean":
                    data[f"{col}_mean"] = ak.mean(jagged, axis=1)
                elif op == "max":
                    data[f"{col}_max"] = ak.max(jagged, axis=1)
                elif op == "sum":
                    data[f"{col}_sum"] = ak.sum(jagged, axis=1)
                elif op == "count":
                    data[f"{col}_count"] = ak.num(jagged, axis=1)

    # 4. Normalization
    if args.scale:
        for col, method in args.scale:
            values = data[col]
            if method == "zscore":
                mean = ak.mean(values)
                std = ak.std(values)
                data[col] = (values - mean) / std
            elif method == "minmax":
                _min = ak.min(values)
                _max = ak.max(values)
                data[col] = (values - _min) / (_max - _min)

    # 5. Rename columns
    if args.rename:
        for old_name, new_name in args.rename:
            data[new_name] = data[old_name]
            del data[old_name]

    return data


def process_and_save_by_row_groups(input_file, args):
    """Process data row group by row group for large files."""
    # Get metadata from the Parquet file
    metadata = ak.metadata_from_parquet(input_file)
    num_row_groups = metadata.get("num_row_groups", 0)

    if num_row_groups == 0:
        raise ValueError(f"No row groups found in the file {input_file}")

    logger.info(f"Total row groups: {num_row_groups}")

    output_path = args.output
    parquet_writer = None  # Initialize ParquetWriter outside loop

    try:
        for i in range(num_row_groups):
            logger.info(f"Processing row group {i + 1}/{num_row_groups}")

            # Load data for the current row group
            data = ak.from_parquet(input_file, row_groups=[i])

            # Process the data
            processed_data = process_data(data, args)

            # Convert Awkward Array to PyArrow Table
            arrow_table = ak.to_arrow_table(processed_data)

            # Initialize ParquetWriter for the first row group
            if parquet_writer is None:
                parquet_writer = pq.ParquetWriter(
                    output_path,
                    schema=arrow_table.schema,
                    use_dictionary=True,
                    compression="SNAPPY",
                )

            # Write the processed data
            parquet_writer.write_table(arrow_table)
            logger.info(f"Processed and saved row group {i + 1}/{num_row_groups}")

    finally:
        # Ensure the writer is closed properly
        if parquet_writer is not None:
            parquet_writer.close()


def main():
    args = parse_args()

    # Log input and output paths
    logger.info(f"Input file: {args.input_file}")
    logger.info(f"Output file: {args.output}")

    # Process and save data by row groups
    process_and_save_by_row_groups(args.input_file, args)

    logger.info(f"Processing completed. Saved output to: {args.output}")


if __name__ == "__main__":
    main()
