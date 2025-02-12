import argparse
import numpy as np
import awkward as ak
import re
import logging
import colorlog
from typing import List
from pathlib import Path
import tempfile
import shutil
from asteval import Interpreter



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
        
        logger.info(f"Found {len(jagged_cols)} columns")
        return jagged_cols

    def filter_data(self, filters: List[str], handle_none: str = 'nan') -> ak.Array:
        """Apply filters to the data and return a filtered copy."""
        if len(self.data) == 0:
            logger.warning("Data is empty. No filtering applied.")
            return self.data

        mask = ak.Array(np.ones(len(self.data), dtype=bool))
        aeval = Interpreter()

        for filter_expr in filters:
            logger.info(f"Applying filter: {filter_expr}")
            try:
                if ":" in filter_expr:
                    parts = filter_expr.split(":", 1)
                    field = parts[0]
                    index_str = re.match(r'(\d+)\s*(.*)', parts[1])
                    if not index_str:
                        raise ValueError(f"Invalid jagged array filter format: {filter_expr}")
                    index = int(index_str.group(1))
                    condition = index_str.group(2)

                    has_index = ak.num(getattr(self.data, field)) > index
                    indexed_values = ak.mask(
                        ak.firsts(getattr(self.data, field)[:, index:index+1]), 
                        has_index
                    )
                    if handle_none == 'nan':
                        field_values = ak.fill_none(indexed_values, np.nan)
                    elif handle_none == 'zero':
                        field_values = ak.fill_none(indexed_values, 0)
                    else:
                        field_values = indexed_values
                else:
                    field_match = re.match(r'([a-zA-Z_][a-zA-Z0-9_]*)(.*)', filter_expr)
                    if not field_match:
                        raise ValueError(f"Invalid filter format: {filter_expr}")
                    field = field_match.group(1)
                    condition = field_match.group(2)
                    field_values = getattr(self.data, field)

                safe_dict = {
                    'ak': ak,
                    'np': np,
                    'abs': abs,
                    field: field_values
                }
                aeval.symtable.update(safe_dict)
                filter_mask = aeval(f"{field} {condition}")
                mask = mask & filter_mask
            except Exception as e:
                logger.error(f"Failed to apply filter '{filter_expr}': {str(e)}")
                continue

        initial_len = len(self.data)
        filtered_data = self.data[mask]
        final_len = len(filtered_data)
        logger.info(f"Filtered {initial_len - final_len} events ({final_len/initial_len:.1%} remaining)")
        return filtered_data

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
            logger.info(f"Successfully renamed column '{old_name}' to '{new_name}'")
        except Exception as e:
            logger.error(f"Failed to rename column '{old_name}': {str(e)}")

    def delete_columns(self, columns: List[str]):
        """Delete specified columns from the data."""
        logger.info(f"Deleting columns: {columns}")
        try:
            self.data = ak.without_field(self.data, columns)
            logger.info(f"Successfully deleted columns: {columns}")
        except Exception as e:
            logger.error(f"Failed to delete columns '{columns}': {str(e)}")

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

    def slice_field(self, slice_spec: str):
        """Slice a field to create a new field."""
        try:
            field, index_str, new_field = slice_spec.split(':')
            index = int(index_str)
            logger.info(f"Slicing field '{field}' at index {index} to create '{new_field}'")
            
            if field not in ak.fields(self.data):
                logger.error(f"Field '{field}' not found in data")
                return
            
            if new_field in ak.fields(self.data):
                logger.error(f"New field name '{new_field}' already exists. Choose a different name.")
                return
            
            # Get values at the specified index where possible
            has_index = ak.num(getattr(self.data, field)) > index
            indexed_values = ak.mask(
                ak.firsts(getattr(self.data, field)[:, index:index+1]), 
                has_index
            )
            # Replace None with nan
            field_values = ak.fill_none(indexed_values, np.nan)
            
            self.data = ak.with_field(self.data, field_values, new_field)
            logger.info(f"Successfully created new field '{new_field}'")
        except Exception as e:
            logger.error(f"Failed to slice field '{slice_spec}': {str(e)}")

def main():
    parser = argparse.ArgumentParser(
        description='Process parquet files for ML applications',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    %(prog)s cosmic_data.parquet -o processed_data.parquet
    
    # Apply filters (multiple conditions supported)
    %(prog)s cosmic_data.parquet -f "energy>1e3" "zenith_angle<45" "num_particles>=10"
    
    # Add new features
    %(prog)s cosmic_data.parquet --features "zenith_angle:np.sin(zenith_angle):sin_zenith" "energy:np.log(energy):log_energy"
    
    # Process jagged arrays to fixed length
    %(prog)s cosmic_data.parquet --max-length 10 --pad-value -999
    
    # Scale columns
    %(prog)s cosmic_data.parquet --scale "energy:standard" "zenith_angle:minmax"
    
    # Rename columns
    %(prog)s cosmic_data.parquet --rename "oldName:newName" "energy:kinetic_energy"
    
    # Delete columns
    %(prog)s cosmic_data.parquet --delete "unnecessary_column" "another_column"
    
    # Save as npz format
    %(prog)s cosmic_data.parquet --format npz
    
    # Slice fields
    %(prog)s cosmic_data.parquet --slice "particle_energy:1:second_particle_energy" "arrival_time:0:first_arrival_time"
    """)
    
    parser.add_argument('input_file', 
                       help='Input parquet file path containing cosmic ray data')
    
    parser.add_argument('--output', '-o',
                       help='Output file path. If not specified, will overwrite the input file')
    
    parser.add_argument('--filters', '-f', nargs='+',
                       help='''Filters to apply to the dataset. Each filter should be a valid Python expression.
Examples:
  "energy>10"            - Select events with energy > 10 EeV
  "zenith_angle<45"       - Select events with zenith angle < 45 degrees
  "num_sds>=10"     - Select events with 10 or more SDs.
Multiple filters can be specified and will be applied sequentially.''')
    
    parser.add_argument('--features', '-F', nargs='+',
                       help='''Add new features using the format: field:expression:newField
The expression can use numpy functions (accessed via 'np.')
Examples:
  "zenith_angle:np.sin(zenith_angle):sin_zenith"     - Add sine of zenith angle
  "energy:np.log(energy):log_energy"                 - Add natural log of energy
  "arrival_time:abs(arrival_time):abs_arrival_time"  - Add absolute value of arrival time
  "azimuth:np.cos(azimuth):cos_azimuth"              - Add cosine of azimuth angle''')
    
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
  "energy:standard"    - Apply standard scaling to energy
  "zenith_angle:minmax" - Apply min-max scaling to zenith angle''')
    
    parser.add_argument('--rename', nargs='+',
                       help='''Rename columns using format: old_name:new_name
Examples:
  "energy:kinetic_energy"
  "zenith_angle:zenith"''')
    
    parser.add_argument('--format', choices=['parquet', 'npz'],
                       default='parquet',
                       help='''Output format (default: parquet)
  parquet - Save as parquet file (maintains column structure)
  npz     - Save as numpy compressed archive (good for ML frameworks)''')

    parser.add_argument('--slice', '-s', nargs='+',
                       help='''Slice fields to create new fields using the format: field:index:newField
Examples:
  "particle_energy:1:second_particle_energy"     - Create a new field 'second_particle_energy' from the second element of 'particle_energy'
  "arrival_time:0:first_arrival_time"            - Create a new field 'first_arrival_time' from the first element of 'arrival_time"''')

    parser.add_argument('--delete', nargs='+',
                       help='''Delete specified columns from the dataset.
Examples:
  "unnecessary_column"
  "another_column"''')

    parser.add_argument('--handle-none', choices=['nan', 'zero', 'none'], default='nan',
                       help='''Specify how to handle None values in jagged arrays (default: nan).
  nan  - Replace None with NaN
  zero - Replace None with 0
  none - Keep None values as is''')

    args = parser.parse_args()

    processor = ParquetProcessor(args.input_file)

    if args.filters:
        processor.filter_data(args.filters, handle_none=args.handle_none)

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

    if args.delete:
        processor.delete_columns(args.delete)

    if args.slice:
        for slice_spec in args.slice:
            processor.slice_field(slice_spec)

    # Use a temporary file to ensure safe overwriting
    if args.output:
        output_file = args.output
    else:
        # Use a temporary file in the same directory as the input file
        input_path = Path(args.input_file)
        with tempfile.NamedTemporaryFile(delete=False, dir=input_path.parent, suffix=input_path.suffix) as tmp_file:
            temp_output_file = tmp_file.name
        output_file = temp_output_file

    try:
        processor.save(output_file, args.format)
        if not args.output:
            # Overwrite the original file with the temporary file
            shutil.move(temp_output_file, args.input_file)
    except Exception as e:
        logger.error(f"Failed to save the processed data: {str(e)}")
        if not args.output:
            # Clean up the temporary file if an error occurred
            Path(temp_output_file).unlink(missing_ok=True)
        raise

if __name__ == "__main__":
    main()

