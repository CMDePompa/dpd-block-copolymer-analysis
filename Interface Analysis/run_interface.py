#!/usr/bin/env python3
"""Entry point for interface_analysis V2.

Examples:
    python run_interface.py --make-template interface_config_v2.yaml
    python run_interface.py --config interface_config_v2.yaml
    python run_interface.py --config interface_config_v2.yaml --test-mode
    python run_interface.py --config lamella.yaml --calibrate

See Int_README.md for full documentation.
"""

import importlib.util
from pathlib import Path
import sys

# Load this folder as a package even though its display name contains a space.
package_dir = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "interface_analysis_v2", package_dir / "__init__.py",
    submodule_search_locations=[str(package_dir)],
)
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
spec.loader.exec_module(package)
from interface_analysis_v2.cli import main

if __name__ == "__main__":
    main()
