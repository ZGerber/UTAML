import sys
import awkward as ak
import numpy as np
from pathlib import Path
import pyarrow.parquet as pq

def main():
    if len(sys.argv) < 2:
        print("Usage: pyparquet <input_parquet1> [input_parquet2 ...]")
        sys.exit(1)

    # Collect all input files
    input_files = sys.argv[1:]

    try:
        # Dictionary to hold data variables
        data_dict = {}
        
        for idx, input_file in enumerate(input_files, start=1):
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
            data_var_name = f"data{idx}"
            data_dict[data_var_name] = data
            print(f"Loaded data from {input_file} into variable '{data_var_name}'")
    except Exception as e:
        print(f"Failed to load parquet file: {e}")
        sys.exit(1)

    # Start an IPython interactive session
    try:
        from IPython import embed
        local_vars = {**data_dict, 'ak': ak, 'np': np}
        embed(user_ns=local_vars)
    except ImportError:
        print("IPython is not installed. Please install it to use this feature.")
        sys.exit(1)

if __name__ == "__main__":
    main() 