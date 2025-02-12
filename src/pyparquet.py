import sys
import awkward as ak
import numpy as np
from pathlib import Path
import pyarrow.parquet as pq

def main():
    if len(sys.argv) < 2:
        print("Usage: pyparquet <input_parquet> [variable_name]")
        sys.exit(1)

    input_file = sys.argv[1]
    variable_name = sys.argv[2] if len(sys.argv) > 2 else "data"

    try:
        # Open the Parquet file
        parquet_file = pq.ParquetFile(input_file)
        
        # Read the file in chunks
        data_chunks = []
        for batch in parquet_file.iter_batches():
            # Convert each batch to an awkward array
            data_chunk = ak.from_arrow(batch)
            data_chunks.append(data_chunk)
        
        # Concatenate all chunks into a single awkward array
        data = ak.concatenate(data_chunks)
        print(f"Loaded data from {input_file} into variable '{variable_name}'")
    except Exception as e:
        print(f"Failed to load parquet file: {e}")
        sys.exit(1)

    # Start an IPython interactive session
    try:
        from IPython import embed
        local_vars = {variable_name: data, 'ak': ak, 'np': np}
        embed(user_ns=local_vars)
    except ImportError:
        print("IPython is not installed. Please install it to use this feature.")
        sys.exit(1)

if __name__ == "__main__":
    main() 