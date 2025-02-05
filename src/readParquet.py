import argparse
import awkward as ak
import numpy as np
import pyarrow.parquet as pq
import logging
import colorlog
from pathlib import Path
from multiprocessing import Pool, cpu_count
import psutil
import os
import math
from contextlib import suppress


def setup_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    formatter = colorlog.ColoredFormatter(
        '%(log_color)s%(levelname)-8s%(reset)s %(white)s%(message)s',
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
        description="Memory-efficient Parquet file splitter for large datasets")
    parser.add_argument("input_file", type=Path, help="Input Parquet file")
    parser.add_argument("output_dir", type=Path, help="Output directory")
    parser.add_argument("--split-proportions", nargs="+", type=float, default=[0.8, 0.2],
                        help="Split proportions (sum must be 1)")
    parser.add_argument("--shuffle", action="store_true", help="Enable shuffling")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--output-format", choices=["parquet", "npz"], default="parquet")
    parser.add_argument("--split-by-columns", nargs="+", help="Columns to split by")
    parser.add_argument("--split-by-row-groups", action="store_true",
                        help="Split using Parquet row groups")
    parser.add_argument("--non-equal-splits", nargs="+", type=int,
                        help="Specify exact row counts for splits")
    parser.add_argument("--row-group-batch-size", type=int, default=1,
                        help="Row groups per batch for memory control")
    parser.add_argument("--max-memory-usage", type=float, default=0.8,
                        help="Max memory fraction to use (0-1)")
    parser.add_argument("--num-workers", type=int, default=cpu_count(),
                        help="Number of worker processes")
    return parser.parse_args()


class MemoryGuard:
    def __init__(self, max_usage=0.8):
        self.max_usage = max_usage
        self.process = psutil.Process(os.getpid())

    def __enter__(self):
        if self.check_memory():
            raise MemoryError("Insufficient memory to proceed")

    def __exit__(self, *args):
        pass

    def check_memory(self):
        mem = psutil.virtual_memory()
        used = mem.used / mem.total
        return used > self.max_usage


def calculate_batch_size(parquet_file, max_mem):
    """Calculate safe number of row groups to process at once"""
    row_group_bytes = sum(rg.total_byte_size for rg in parquet_file.metadata.row_groups)
    total_bytes = row_group_bytes
    available_mem = psutil.virtual_memory().available * max_mem
    return max(1, int(available_mem // (total_bytes / parquet_file.num_row_groups)))


def split_by_row_groups(args):
    """Memory-efficient row group splitting"""
    with MemoryGuard(args.max_memory_usage):
        parquet_file = pq.ParquetFile(args.input_file)
        batch_size = calculate_batch_size(parquet_file, args.max_memory_usage)

        for batch_start in range(0, parquet_file.num_row_groups, batch_size):
            batch_rgs = range(batch_start, min(batch_start + batch_size, parquet_file.num_row_groups))

            with Pool(args.num_workers) as pool:
                pool.starmap(process_row_group_batch, [
                    (args.input_file, args.output_dir, args.output_format,
                     list(batch_rgs), args.shuffle, args.seed)
                ])


def process_row_group_batch(input_file, output_dir, output_format, row_groups, shuffle, seed):
    """Process a batch of row groups"""
    try:
        data = ak.from_parquet(input_file, row_groups=row_groups)

        if shuffle:
            data = data[np.random.default_rng(seed).permutation(len(data))]

        for i, rg in enumerate(row_groups):
            output_file = output_dir / f"rowgroup_{rg}.{output_format}"
            save_split(data[i::len(row_groups)], output_file, output_format)

    except Exception as e:
        logger.error(f"Failed processing row groups {row_groups}: {e}")


def split_by_columns(args):
    """Column splitting"""
    with MemoryGuard(args.max_memory_usage):
        parquet_file = pq.ParquetFile(args.input_file)
        all_columns = parquet_file.schema.names

        for group in args.split_by_columns:
            cols = [c.strip() for c in group.split(",")]
            missing = set(cols) - set(all_columns)
            if missing:
                raise ValueError(f"Missing columns: {missing}")

            output_file = args.output_dir / f"columns_{'-'.join(cols)}.{args.output_format}"
            data = ak.from_parquet(args.input_file, columns=cols)
            save_split(data, output_file, args.output_format)


def streaming_split(args):
    """Streaming split for large datasets"""
    parquet_file = pq.ParquetFile(args.input_file)
    split_writers = create_split_writers(args)

    try:
        for i in range(parquet_file.num_row_groups):
            data = ak.from_parquet(args.input_file, row_groups=[i])

            if args.shuffle:
                data = data[np.random.default_rng(args.seed).permutation(len(data))]

            indices = calculate_split_indices(len(data), args)

            # Apply the indices to the data
            for split_idx, writer in enumerate(split_writers.values()):  # Use integer index
                split_data = data[indices[split_idx]]  # Apply split index to access correct portion of data

                if args.output_format == "parquet":
                    writer.append(split_data)
                elif args.output_format == "npz":
                    for field in split_data.fields:
                        if field not in writer:
                            writer[field] = []
                        writer[field].append(split_data[field].to_numpy())

        close_writers(split_writers, args)

    except Exception as e:
        logger.error(f"Streaming split failed: {e}")
        raise


def calculate_split_indices(n_rows, args):
    """Calculate split indices"""
    if args.non_equal_splits:
        splits = np.cumsum(args.non_equal_splits)
        if splits[-1] > n_rows:
            raise ValueError(f"Non-equal splits sum {splits[-1]} exceeds row count {n_rows}")
        return np.split(np.arange(n_rows), splits[:-1])
    else:
        split_points = (np.cumsum(args.split_proportions) * n_rows).astype(int)[:-1]
        return np.split(np.random.permutation(n_rows) if args.shuffle else np.arange(n_rows), split_points)


def create_split_writers(args):
    """Create file writers for each split"""
    writers = {}
    split_names = ["train", "val", "test"][:len(args.split_proportions)]

    for name in split_names:
        path = args.output_dir / f"{name}.{args.output_format}"
        if args.output_format == "parquet":
            writers[name] = ak.ArrayBuilder()
        elif args.output_format == "npz":
            writers[name] = {}
    return writers


def close_writers(writers, args):
    """Finalize and save all writers"""
    for name, writer in writers.items():
        path = args.output_dir / f"{name}.{args.output_format}"
        if args.output_format == "parquet":
            array = writer.snapshot()
            ak.to_parquet(array, path, compression="ZSTD", row_group_size=10000)
        elif args.output_format == "npz":
            np.savez_compressed(path, **{k: np.concatenate(v) for k, v in writer.items()})
        logger.info(f"Finalized {path}")


def save_split(data, path, format):
    """Save data with memory-efficient writes"""
    if format == "parquet":
        ak.to_parquet(data, path, compression="ZSTD", row_group_size=10000)
    elif format == "npz":
        np.savez_compressed(path, **{
            field: data[field].to_numpy() for field in data.fields
        })
    logger.info(f"Saved {path} ({len(data)} rows)")


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting split with memory limit: {args.max_memory_usage * 100:.0f}%")

    try:
        if args.split_by_row_groups:
            split_by_row_groups(args)
        elif args.split_by_columns:
            split_by_columns(args)
        else:
            streaming_split(args)

    except MemoryError as e:
        logger.critical(f"Memory error: {e}")
        logger.info("Try reducing --row-group-batch-size or increasing --max-memory-usage")
        raise

    logger.info("Split completed successfully")


if __name__ == "__main__":
    main()
