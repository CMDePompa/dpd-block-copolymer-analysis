"""Output, logging, and provenance utilities for interface_analysis_v2.

Conventions follow the S(q) pipeline: per-simulation output directories, a
logger-callable passed down the stack, JSON metadata files, and explicit
recording of every resolved parameter. Additions: a timestamped log FILE per
simulation (under ``<sim_out>/logs/``) and a reproducibility snapshot of
package versions.
"""

from __future__ import annotations

import datetime
import json
import platform
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TextIO

import numpy as np

LogFn = Callable[[str], None]


class SimLogger:
    """Timestamped logger writing to a file and echoing to a parent callable.

    Use as a context manager or call :meth:`close` explicitly. The instance
    itself is callable so it can be passed anywhere a plain ``log(msg)``
    callable is expected (the S(q) pipeline convention).
    """

    def __init__(self, log_path: Path, echo: Optional[LogFn] = print) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh: Optional[TextIO] = log_path.open("a", encoding="utf-8")
        self._echo = echo
        self.path = log_path
        self.warnings: List[str] = []   # WARN/ERROR lines collected for interface_warnings.json

    def __call__(self, msg: str) -> None:
        stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{stamp}] {msg}"
        if self._fh is not None and not self._fh.closed:
            self._fh.write(line + "\n")
            self._fh.flush()
        if self._echo is not None:
            self._echo(msg)
        stripped = msg.lstrip()
        if stripped.startswith(("WARN", "ERROR")):
            self.warnings.append(stripped)

    def warn(self, msg: str) -> None:
        self(f"WARN: {msg}")

    def error(self, msg: str) -> None:
        self(f"ERROR: {msg}")

    def close(self) -> None:
        if self._fh is not None and not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> "SimLogger":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


class _NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder that understands numpy scalars/arrays and Paths.

    NaN/Inf floats are serialized as null (strict JSON compatibility).
    """

    def default(self, o: Any) -> Any:
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            v = float(o)
            return v if np.isfinite(v) else None
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        return super().default(o)

    def iterencode(self, o: Any, _one_shot: bool = False):
        # json emits bare NaN/Infinity for python floats, which is invalid
        # JSON; sanitize them to None recursively first.
        return super().iterencode(_sanitize_floats(o), _one_shot)


def _sanitize_floats(o: Any) -> Any:
    if isinstance(o, float) and not np.isfinite(o):
        return None
    if isinstance(o, dict):
        return {k: _sanitize_floats(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_sanitize_floats(v) for v in o]
    return o


def write_json(obj: Dict[str, Any], path: str | Path) -> None:
    """Write a dict to JSON, handling numpy types and non-finite floats."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, cls=_NumpyJSONEncoder, default=str)


def package_versions() -> Dict[str, str]:
    """Versions of every package the pipeline depends on (reproducibility)."""
    versions: Dict[str, str] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for name in ("numpy", "scipy", "pandas", "matplotlib", "skimage", "yaml", "freud"):
        try:
            mod = __import__(name)
            versions[name] = str(getattr(mod, "__version__", "unknown"))
        except ImportError:
            versions[name] = "not installed"
    return versions


def provenance_block(cfg_hash: str) -> Dict[str, Any]:
    """Standard reproducibility metadata attached to every summary file."""
    from . import __version__
    return {
        "interface_analysis_version": __version__,
        "config_hash": cfg_hash,
        "run_timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "command_line": " ".join(sys.argv),
        "package_versions": package_versions(),
    }


def make_sim_output_dirs(output_root: Path, sim_id: str) -> Dict[str, Path]:
    """Create <output_root>/<sim_id>/{plots,logs} and return the paths."""
    sim_out = output_root / sim_id
    dirs = {
        "root": sim_out,
        "plots": sim_out / "plots",
        "logs": sim_out / "logs",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs
