"""Load the moved analysis folder under its import name for tests."""
import importlib.util
from pathlib import Path
import sys

package_dir = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "interface_analysis_v2", package_dir / "__init__.py",
    submodule_search_locations=[str(package_dir)],
)
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
spec.loader.exec_module(package)
