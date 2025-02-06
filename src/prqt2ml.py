import argparse
import numpy as np
import awkward as ak
# import pyarrow.parquet as pq
import re
import logging
import colorlog
from typing import List
from pathlib import Path
import os
import sys
import site


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

class ParquetProcessor:
    def __init__(self, input_file: str):
        """Initialize the processor with input file path."""
        self.input_file = Path(input_file)
        logger.info(f"Reading input file: {input_file}")
        try:
            self.data = ak.from_parquet(input_file)
            logger.debug(f"Available fields: {ak.fields(self.data)}")
        except Exception as e:
            logger.error(f"Failed to read parquet file: {str(e)}")
            raise
            
        self.jagged_columns = self._identify_jagged_columns()
        logger.info(f"Identified jagged columns: {self.jagged_columns}")

    def _identify_jagged_columns(self) -> List[str]:
        """Identify columns containing jagged arrays."""
        jagged_cols = []
        for field in ak.fields(self.data):
            try:
                # Convert to numpy array first
                num_elements = ak.to_numpy(ak.num(getattr(self.data, field), axis=-1))
                if num_elements.max() > 1:
                    jagged_cols.append(field)
            except Exception as e:
                continue
        
        logger.debug(f"Found {len(jagged_cols)} jagged columns")
        return jagged_cols

    def filter_data(self, filters: List[str]):
        """Apply filters to the data."""
        mask = ak.Array(np.ones(len(self.data), dtype=bool))
        for filter_expr in filters:
            logger.info(f"Applying filter: {filter_expr}")
            try:
                # Check if this is a jagged array index filter (contains colon)
                if ":" in filter_expr:
                    # Split into field, index, and condition parts
                    parts = filter_expr.split(":", 1)
                    field = parts[0]
                    # Extract the index and condition
                    index_str = re.match(r'(\d+)\s*(.*)', parts[1])
                    if not index_str:
                        raise ValueError(f"Invalid jagged array filter format: {filter_expr}")
                    index = int(index_str.group(1))
                    condition = index_str.group(2)

                    # Get values at the specified index where possible
                    has_index = ak.num(getattr(self.data, field)) > index
                    indexed_values = ak.mask(
                        ak.firsts(getattr(self.data, field)[:, index:index+1]), 
                        has_index
                    )
                    # Replace None with nan
                    field_values = ak.fill_none(indexed_values, np.nan)
                else:
                    # Handle regular filter expressions by extracting field name and condition
                    field_match = re.match(r'([a-zA-Z_][a-zA-Z0-9_]*)(.*)', filter_expr)
                    if not field_match:
                        raise ValueError(f"Invalid filter format: {filter_expr}")
                    field = field_match.group(1)
                    condition = field_match.group(2)
                    field_values = getattr(self.data, field)

                # Create safe evaluation environment
                safe_dict = {
                    'ak': ak,
                    'np': np,
                    'abs': abs,
                    field: field_values
                }
                # Evaluate the condition
                filter_mask = eval(f"{field} {condition}", {"__builtins__": {}}, safe_dict)
                
                mask = mask & filter_mask
            except Exception as e:
                logger.error(f"Failed to apply filter '{filter_expr}': {str(e)}")
                continue
                
        initial_len = len(self.data)
        self.data = self.data[mask]
        final_len = len(self.data)
        logger.info(f"Filtered {initial_len - final_len} events ({final_len/initial_len:.1%} remaining)")

    def add_feature(self, feature_spec: str):
        """Add new features based on existing columns."""
        try:
            input_field, expression, new_field = feature_spec.split(':')
            logger.info(f"Adding feature '{new_field}' using expression: {expression}")
            
            if input_field not in ak.fields(self.data):
                logger.error(f"Input field '{input_field}' not found in data")
                return
                
            safe_dict = {
                'ak': ak,
                'np': np,
                input_field: getattr(self.data, input_field)
            }
            result = eval(expression, {"__builtins__": {}}, safe_dict)
            self.data = ak.with_field(self.data, result, new_field)
        except Exception as e:
            logger.error(f"Failed to add feature '{feature_spec}': {str(e)}")

    def process_jagged(self, max_length: int = None, pad_value: float = 0.0):
        """Process jagged arrays by truncating or padding."""
        if max_length is None:
            return
            
        logger.info(f"Processing jagged arrays to length {max_length} with pad value {pad_value}")
        processed = {}
        for field in ak.fields(self.data):
            if field in self.jagged_columns:
                try:
                    # Truncate if needed
                    arr = ak.pad_none(getattr(self.data, field), max_length, clip=True)
                    # Replace None with pad_value
                    arr = ak.fill_none(arr, pad_value)
                    processed[field] = arr
                except Exception as e:
                    logger.error(f"Failed to process jagged field '{field}': {str(e)}")
                    processed[field] = getattr(self.data, field)
            else:
                processed[field] = getattr(self.data, field)
        
        self.data = ak.Array(processed)
        logger.info("Completed jagged array processing")

    def scale_column(self, column: str, method: str = 'standard'):
        """Scale/normalize a column using various methods."""
        from sklearn.preprocessing import StandardScaler, MinMaxScaler
        
        logger.info(f"Scaling column '{column}' using method: {method}")
        
        scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler()
        }
        
        if method not in scalers:
            logger.error(f"Unknown scaling method: {method}")
            return
            
        try:
            if column in self.jagged_columns:
                flat_data = ak.flatten(getattr(self.data, column))
                scaler = scalers[method].fit(flat_data.to_numpy().reshape(-1, 1))
                
                scaled = ak.Array([
                    scaler.transform(subarr.to_numpy().reshape(-1, 1)).flatten()
                    for subarr in getattr(self.data, column)
                ])
                self.data = ak.with_field(self.data, scaled, column)
            else:
                values = getattr(self.data, column)
                scaled = scalers[method].fit_transform(values.to_numpy().reshape(-1, 1)).flatten()
                self.data = ak.with_field(self.data, scaled, column)
            logger.info(f"Successfully scaled column '{column}'")
        except Exception as e:
            logger.error(f"Failed to scale column '{column}': {str(e)}")

    def rename_column(self, old_name: str, new_name: str):
        """Rename a column."""
        logger.info(f"Renaming column '{old_name}' to '{new_name}'")
        try:
            self.data = ak.with_field(self.data, getattr(self.data, old_name), new_name)
            self.data = ak.without_field(self.data, old_name)
            logger.debug(f"Successfully renamed column '{old_name}' to '{new_name}'")
        except Exception as e:
            logger.error(f"Failed to rename column '{old_name}': {str(e)}")

    def save(self, output_file: str, format: str = 'parquet'):
        """Save the processed data in the specified format."""
        # Convert relative path to absolute path
        output_path = Path(output_file).resolve()
        
        # Ensure the parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving output to: {output_path} (format: {format})")
        
        try:
            if format == 'parquet':
                ak.to_parquet(self.data, output_path)
            elif format == 'npz':
                save_dict = {field: ak.to_numpy(getattr(self.data, field)) 
                            for field in ak.fields(self.data)}
                np.savez(output_path, **save_dict)
            logger.info("Successfully saved output file")
        except Exception as e:
            logger.error(f"Failed to save output file: {str(e)}")
            raise

def main():
    parser = argparse.ArgumentParser(
        description='Process parquet files for ML applications',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    %(prog)s input.parquet -o output.parquet
    
    # Apply filters (multiple conditions supported)
    %(prog)s input.parquet -f "pt>20" "abs(eta)<2.4" "nJets>=2"
    
    # Add new features
    %(prog)s input.parquet --features "theta:np.sin(theta):sin_theta" "pt:np.log(pt):log_pt"
    
    # Process jagged arrays to fixed length
    %(prog)s input.parquet --max-length 10 --pad-value -999
    
    # Scale columns
    %(prog)s input.parquet --scale "pt:standard" "eta:minmax"
    
    # Rename columns
    %(prog)s input.parquet --rename "oldName:newName" "pt:transverse_momentum"
    
    # Save as npz format
    %(prog)s input.parquet --format npz
    """)
    
    parser.add_argument('input_file', 
                       help='Input parquet file path')
    
    parser.add_argument('--output', '-o',
                       help='Output file path. If not specified, will prepend "processed_" to input filename')
    
    parser.add_argument('--filters', '-f', nargs='+',
                       help='''Filters to apply to the dataset. Each filter should be a valid Python expression.
Examples:
  "pt>20"            - Select events with pt > 20
  "abs(eta)<2.4"     - Select events with |eta| < 2.4
  "nJets>=2"         - Select events with 2 or more jets
Multiple filters can be specified and will be applied sequentially.''')
    
    parser.add_argument('--features', '-F', nargs='+',
                       help='''Add new features using the format: field:expression:newField
The expression can use numpy functions (accessed via 'np.')
Examples:
  "theta:np.sin(theta):sin_theta"     - Add sine of theta
  "pt:np.log(pt):log_pt"             - Add natural log of pt
  "eta:abs(eta):abs_eta"             - Add absolute value of eta
  "phi:np.cos(phi):cos_phi"          - Add cosine of phi''')
    
    parser.add_argument('--max-length', type=int,
                       help='''Maximum length for jagged arrays. Arrays longer than this will be truncated,
shorter arrays will be padded with --pad-value''')
    
    parser.add_argument('--pad-value', type=float, default=0.0,
                       help='Value to use for padding jagged arrays (default: 0.0)')
    
    parser.add_argument('--scale', nargs='+',
                       help='''Scale columns using format: column:method
Available scaling methods:
  standard - Zero mean and unit variance scaling
  minmax   - Scale to range [0,1]
Examples:
  "pt:standard"    - Apply standard scaling to pt
  "eta:minmax"     - Apply min-max scaling to eta''')
    
    parser.add_argument('--rename', nargs='+',
                       help='''Rename columns using format: old_name:new_name
Examples:
  "pt:transverse_momentum"
  "eta:pseudorapidity"''')
    
    parser.add_argument('--format', choices=['parquet', 'npz'],
                       default='parquet',
                       help='''Output format (default: parquet)
  parquet - Save as parquet file (maintains column structure)
  npz     - Save as numpy compressed archive (good for ML frameworks)''')

    args = parser.parse_args()

    # Modify how output_file is constructed
    if args.output:
        output_file = args.output
    else:
        # Get the input file name and add "processed_" prefix in the same directory
        input_path = Path(args.input_file)
        output_file = input_path.parent / f"processed_{input_path.name}"
    
    processor = ParquetProcessor(args.input_file)

    if args.filters:
        processor.filter_data(args.filters)

    if args.features:
        for feature_spec in args.features:
            processor.add_feature(feature_spec)

    if args.max_length is not None:
        processor.process_jagged(args.max_length, args.pad_value)

    if args.scale:
        for scale_spec in args.scale:
            column, method = scale_spec.split(':')
            processor.scale_column(column, method)

    if args.rename:
        for rename_spec in args.rename:
            old_name, new_name = rename_spec.split(':')
            processor.rename_column(old_name, new_name)

    processor.save(output_file, args.format)

if __name__ == "__main__":
    main()

