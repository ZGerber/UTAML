#!/usr/bin/env python3
import os
import sys
import subprocess
from pathlib import Path

def find_tldir():
    # Start from the script's directory
    current_path = Path(__file__).resolve().parent
    
    # Try up to 2 parent directories
    for _ in range(3):
        if (current_path / "tluti" / "load_tluti.C").exists():
            return str(current_path)
        current_path = current_path.parent
    
    print("ERROR: TLDIR not found; re-install tlanalysis", file=sys.stderr)
    sys.exit(2)

def verify_environment():
    tldir = os.environ.get('TLDIR', '')
    
    # If TLDIR not set, try to find it
    if not tldir:
        tldir = find_tldir()
        os.environ['TLDIR'] = tldir
    
    # Verify required files exist
    tluti_path = Path(tldir) / "tluti" / "load_tluti.C"
    tlsdfit_path = Path(tldir) / "tlsdfit" / "load_tlsdfit.C"
    
    if not tluti_path.exists():
        print(f"ERROR: TLDIR/tluti/load_tluti.C ({tluti_path}) not found", file=sys.stderr)
        print("reinstall tlanalysis and be sure to source tlanalysis/bin/this_tlanalysis.sh", file=sys.stderr)
        sys.exit(2)
    
    if not tlsdfit_path.exists():
        print(f"ERROR: TLDIR/tlsdfit/load_tlsdfit.C ({tlsdfit_path}) not found", file=sys.stderr)
        print("reinstall tlanalysis and be sure to source tlanalysis/bin/this_tlanalysis.sh", file=sys.stderr)
        sys.exit(2)

def setup_root_env():
    tdstio = os.environ.get('TDSTio', '')
    if not tdstio:
        print("ERROR: TDSTio environment variable not set", file=sys.stderr)
        sys.exit(2)
    
    # Construct ROOT macro loading command
    load_command = f".L {tdstio}/macro/load_TDSTio.C"
    load_command += f"\n.L {os.environ['TLDIR']}/tluti/load_tluti.C"
    
    # Set this for ROOT to find the correct environment
    os.environ['ROOTSYS'] = os.environ.get('ROOTSYS', '')
    os.environ['LD_LIBRARY_PATH'] = os.environ.get('LD_LIBRARY_PATH', '')
    
    return load_command

def main():
    # Verify and setup environment
    verify_environment()
    root_load_command = setup_root_env()
    
    # Import necessary packages
    try:
        import uproot
        import awkward as ak
        import IPython
    except ImportError as e:
        print(f"ERROR: Required package not found: {e}", file=sys.stderr)
        print("Please install required packages: pip install uproot awkward IPython", file=sys.stderr)
        sys.exit(2)
    
    # Create a welcome message with instructions
    welcome_msg = """
    ROOT environment configured.
    
    Available variables:
    - TLDIR: {}
    - TDSTio: {}
    
    You can now use uproot to read ROOT files, for example:
    >>> file = uproot.open("your_file.root")
    >>> tree = file["tree_name"]
    >>> arrays = tree.arrays()
    """.format(os.environ.get('TLDIR', ''), os.environ.get('TDSTio', ''))
    
    # Start IPython session
    IPython.embed(header=welcome_msg)

if __name__ == "__main__":
    main()
    