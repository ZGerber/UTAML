import sys
import awkward as ak
import numpy as np
from pathlib import Path

def main():
    if len(sys.argv) < 2:
        print("Usage: pyparquet <input_parquet> [variable_name]")
        sys.exit(1)

    input_file = sys.argv[1]
    variable_name = sys.argv[2] if len(sys.argv) > 2 else "data"

    try:
        data = ak.from_parquet(input_file)
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