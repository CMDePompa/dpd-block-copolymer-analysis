"""Read-only bridge to the project's existing S(q) pipeline.

Two responsibilities:

1. Locate and load the S(q) pipeline's ``trajectory_reader`` module by file
   path (via importlib, under a private module name) so its ``iter_frames``
   can serve as the REFERENCE parser in equivalence tests. The S(q) package
   is never modified and never imported under its own top-level name
   (``pipeline`` is too generic to put on sys.path safely).

2. Read per-simulation S(q) RESULTS (q_star, L0, accepted equilibration
   window) from an S(q) output directory. These provide the default
   ``analysis_start_step`` and the q*/L0 cross-checks. All reads are
   tolerant: missing files return partial results with explanatory notes,
   never exceptions.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Optional

_READER_RELPATH = Path("pipeline") / "trajectory_reader.py"
_LOADED_READER: Optional[ModuleType] = None


def find_sq_pipeline_dir(
    explicit: Optional[str | Path] = None,
    search_from: Optional[str | Path] = None,
) -> Optional[Path]:
    """Locate an S(q) pipeline directory containing pipeline/trajectory_reader.py.

    Search order: the explicit config value; then the sibling
    ``Structure Factor`` directory and older pipeline names under
    *search_from* (default: cwd), its parent, and the parent of this package.
    Returns None if not found.
    """
    candidates = []
    if explicit is not None:
        candidates.append(Path(explicit))
    roots = []
    if search_from is not None:
        roots.append(Path(search_from))
    roots.append(Path.cwd())
    roots.append(Path(__file__).resolve().parent.parent)  # project dir holding this package
    for root in roots:
        for name in ("Structure Factor", "Sq_pipeline_v4", "freud_Sq_pipeline"):
            candidates.append(root / name)
            candidates.append(root.parent / name)
    for cand in candidates:
        try:
            if (cand / _READER_RELPATH).is_file():
                return cand.resolve()
        except OSError:
            continue
    return None


def load_reference_reader(sq_pipeline_dir: Optional[str | Path] = None) -> Optional[ModuleType]:
    """Load the S(q) pipeline's trajectory_reader module by file path.

    Returns the module (with ``iter_frames``, ``FrameData``, ``count_frames``)
    or None when the pipeline directory cannot be located. The module is
    registered under the private name ``_sq_reference_trajectory_reader`` so
    nothing collides with an installed package.
    """
    global _LOADED_READER
    if _LOADED_READER is not None:
        return _LOADED_READER
    base = find_sq_pipeline_dir(explicit=sq_pipeline_dir)
    if base is None:
        return None
    path = base / _READER_RELPATH
    spec = importlib.util.spec_from_file_location("_sq_reference_trajectory_reader", path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    # Must be registered BEFORE exec_module: dataclasses resolves string
    # annotations via sys.modules[cls.__module__].
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _LOADED_READER = module
    return module


# ---------------------------------------------------------------------------
# S(q) results
# ---------------------------------------------------------------------------
def load_sq_results(sq_results_dir: Optional[str | Path], sim_id: str) -> Dict[str, Any]:
    """Read q_star, L0, and the accepted window for one simulation.

    Looks for ``<sq_results_dir>/<sim_id>/analysis_summary.csv`` and
    ``accepted_windows.json`` (the S(q) pipeline's per-sim outputs).

    Returns a dict with keys:
      available (bool), q_star, L0, window_start_step, window_end_step,
      n_accepted_frames, stable_window_found, source, notes (list of str).
    Missing/invalid entries are None with a note; never raises.
    """
    out: Dict[str, Any] = {
        "available": False,
        "q_star": None,
        "L0": None,
        "window_start_step": None,
        "window_end_step": None,
        "n_accepted_frames": None,
        "stable_window_found": None,
        "source": None,
        "notes": [],
    }
    if sq_results_dir is None:
        out["notes"].append("sq_results_dir not configured; S(q) results not used.")
        return out
    sim_dir = Path(sq_results_dir) / sim_id
    if not sim_dir.is_dir():
        out["notes"].append(f"No S(q) output directory found at {sim_dir}.")
        return out
    out["source"] = str(sim_dir)

    summary_csv = sim_dir / "analysis_summary.csv"
    if summary_csv.is_file():
        try:
            import pandas as pd
            df = pd.read_csv(summary_csv)
            if len(df) >= 1:
                row = df.iloc[0]
                # V4 names the publishable composition peak q_star_sq/L0_sq;
                # accept the V1 names for previously generated result trees.
                q_star = float(row.get(
                    "q_star_sq", row.get("q_star_freud", float("nan")),
                ))
                L0 = float(row.get(
                    "L0_sq", row.get("L0_freud", float("nan")),
                ))
                if math.isfinite(q_star) and q_star > 0:
                    out["q_star"] = q_star
                    out["L0"] = L0 if math.isfinite(L0) and L0 > 0 else 2.0 * math.pi / q_star
                else:
                    out["notes"].append(
                        "S(q) summary present but q_star_sq/q_star_freud is not finite."
                    )
        except Exception as exc:  # tolerate malformed files, but say so
            out["notes"].append(f"Failed to parse {summary_csv.name}: {exc}")
    else:
        out["notes"].append("analysis_summary.csv not found in S(q) output.")

    windows_json = sim_dir / "accepted_windows.json"
    if windows_json.is_file():
        try:
            with windows_json.open("r") as fh:
                doc = json.load(fh)
            out["window_start_step"] = doc.get("window_start_step")
            out["window_end_step"] = doc.get("window_end_step")
            out["n_accepted_frames"] = doc.get("n_accepted_frames")
            out["stable_window_found"] = doc.get("stable_found")
        except Exception as exc:
            out["notes"].append(f"Failed to parse {windows_json.name}: {exc}")
    else:
        out["notes"].append("accepted_windows.json not found in S(q) output.")

    out["available"] = out["q_star"] is not None or out["window_start_step"] is not None
    return out
