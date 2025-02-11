import argparse
import uproot
import awkward as ak
from pathlib import Path
from multiprocessing import Pool
import pyarrow.parquet as pq
import pyarrow as pa
import logging
import colorlog
import psutil
import signal
import sys


def setup_logger():
    """Set up the logger with color formatting and timestamps."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    handler = logging.StreamHandler()

    formatter = colorlog.ColoredFormatter(
        '%(log_color)s%(asctime)s %(levelname)-8s%(reset)s %(white)s%(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
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


def signal_handler(sig, frame):
    """Handle interrupt signals for graceful shutdown."""
    logger.info("Interrupt received. Performing cleanup...")
    sys.exit(1)


signal.signal(signal.SIGINT, signal_handler)


def parse_args():
    """Parse command-line arguments with new compression and row group options."""
    parser = argparse.ArgumentParser(
        description="Convert ROOT file to a Parquet file.")
    parser.add_argument("file_path", type=str, help="Path to the ROOT file")
    parser.add_argument("column_names", type=str, nargs="*",
                        help="Columns to convert (default: all -- not recommended)")
    parser.add_argument("--tree_name", type=str, default="taTree",
                        help="TTree name (default: 'taTree')")
    parser.add_argument("-o", "--output_dir", type=str, default=".",
                        help="Output directory (default: current)")
    parser.add_argument("-x", "--omit_columns", type=str, nargs="*",
                        help="Columns to exclude (overrides column_names)")
    parser.add_argument("-j", "--num_workers", type=int, default=4,
                        help="Number of worker processes (default: 4)")
    parser.add_argument("-c", "--chunk_size", type=int, default=10000,
                        help="Number of rows per chunk (default: 10000)")
    parser.add_argument("--compression", type=str, default="SNAPPY",
                        choices=["SNAPPY", "GZIP", "BROTLI", "ZSTD", "NONE"],
                        help="Compression algorithm (default: SNAPPY)")
    parser.add_argument("--row_group_size", type=int, default=10000,
                        help="Row group size for Parquet files (default: 10000)")
    return parser.parse_args()


def get_columns(tree, args):
    """Resolve column selection logic."""
    all_columns = tree.keys()

    if args.omit_columns:
        return [col for col in all_columns if col not in args.omit_columns]
    if args.column_names:
        return [col for col in args.column_names if col in all_columns]
    return all_columns


def process_chunk(args):
    """Process a single chunk of data with configurable compression."""
    (file_path, tree_name, columns, entry_start, entry_stop, temp_dir,
     compression, row_group_size) = args
    try:
        with uproot.open(file_path) as file:
            tree = file[tree_name]
            data = tree.arrays(columns, entry_start=entry_start, entry_stop=entry_stop)

            chunk_path = temp_dir / f"chunk_{entry_start}-{entry_stop}.parquet"
            ak.to_parquet(data, chunk_path,
                          row_group_size=row_group_size,
                          compression=compression)
            logger.info(f"Processed chunk: {entry_start}-{entry_stop}")
            return chunk_path
    except Exception as e:
        logger.error(f"Failed to process chunk {entry_start}-{entry_stop}: {e}")
        raise


def validate_schema(chunk_files):
    """Ensure all chunks have identical schemas."""
    with pq.ParquetFile(chunk_files[0]) as pf:
        reference_schema = pf.schema_arrow

    for chunk in chunk_files[1:]:
        with pq.ParquetFile(chunk) as pf:
            if pf.schema_arrow != reference_schema:
                raise ValueError(f"Schema mismatch in chunk: {chunk}")


def save_parquet_multiprocessing(file_path, output_dir, tree_name, columns,
                                 chunk_size, num_workers, compression, row_group_size):
    """Main conversion workflow with enhanced error handling."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{Path(file_path).stem}.parquet"

    if output_path.exists():
        logger.warning(f"Output file {output_path} already exists. Overwrite? (y/n)")
        if input().lower() != "y":
            logger.info("Aborting conversion.")
            return None

    temp_dir = output_dir / "temp_chunks"
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        with uproot.open(file_path) as file:
            tree = file[tree_name]
            total_entries = tree.num_entries

            tasks = [
                (file_path, tree_name, columns, start, min(start + chunk_size, total_entries),
                 temp_dir, compression, row_group_size)
                for start in range(0, total_entries, chunk_size)
            ]

        logger.info("Starting parallel processing of chunks...")
        with Pool(num_workers) as pool:
            chunk_files = pool.map(process_chunk, tasks)

        logger.info("Validating chunk schemas...")
        validate_schema(chunk_files)

        logger.info("Merging chunks into final Parquet file...")
        merge_parquet_files(chunk_files, output_path, compression)

        return output_path

    finally:
        logger.info("Performing cleanup of temporary files...")
        for chunk_file in temp_dir.glob("*.parquet"):
            chunk_file.unlink()
        temp_dir.rmdir()


def merge_parquet_files(chunk_files, output_path, compression):
    """Merge chunks with progress tracking and configurable compression."""
    try:
        with pq.ParquetFile(chunk_files[0]) as pf:
            schema = pf.schema_arrow
            writer = pq.ParquetWriter(output_path, schema, compression=compression)

        for i, chunk_file in enumerate(chunk_files):
            table = pq.read_table(chunk_file)
            writer.write_table(table)
            logger.info(f"Merged chunk {i + 1}/{len(chunk_files)}")

        writer.close()
        logger.info(f"Final file created: {output_path}")
    except Exception as e:
        logger.error(f"Merging failed: {e}")
        if 'writer' in locals():
            writer.close()
        raise


def monitor_resources():
    """Log system resource utilization."""
    memory = psutil.virtual_memory()
    logger.info(f"Memory usage: {memory.percent}% ({memory.used / 1e9:.2f}GB used)")
    logger.info(f"CPU usage: {psutil.cpu_percent()}%")


def validate_row_groups(output_path):
    """Verify Parquet file structure."""
    metadata = pq.read_metadata(output_path)
    logger.info(f"Parquet file has {metadata.num_row_groups} row groups")
    logger.info(f"Total rows: {metadata.num_rows}")


def main():
    args = parse_args()

    with uproot.open(args.file_path) as file:
        tree = file[args.tree_name]
        columns = get_columns(tree, args)
        if not columns:
            raise ValueError("No valid columns selected for conversion")

    logger.info("Starting conversion with parameters:")
    logger.info(f"Input: {args.file_path}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Chunk size: {args.chunk_size}")
    logger.info(f"Compression: {args.compression}")
    logger.info(f"Workers: {args.num_workers}")

    monitor_resources()

    output_path = save_parquet_multiprocessing(
        args.file_path, args.output_dir, args.tree_name, columns,
        args.chunk_size, args.num_workers, args.compression, args.row_group_size
    )

    if output_path:
        validate_row_groups(output_path)
        logger.info(f"Conversion complete: {output_path}")
        monitor_resources()


if __name__ == "__main__":
    main()
