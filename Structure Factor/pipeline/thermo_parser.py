"""Parse LAMMPS out/log files for DPD simulations run on LSF.

Handles:
  - Multiple thermo blocks per file (each preceded by a 'Setting up Verlet run' header)
  - Stage markers of the form: STAGE N: segment M/K setting a12=VALUE
  - LSF job-scheduler footer (submitted/started/terminated times, exit code, resource use)
  - Abnormal termination detection (TERM_RUNLIMIT, nonzero exit code)
"""

from __future__ import annotations

import json
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------
RE_STAGE = re.compile(
    r"STAGE\s+(?P<stage>\d+):\s+segment\s+(?P<seg>\d+)/(?P<total_segs>\d+)"
    r"\s+setting\s+a12=(?P<a12>[0-9.eE+\-]+)",
    re.IGNORECASE,
)
RE_THERMO_HEADER = re.compile(r"^\s+Step\s+", re.IGNORECASE)
RE_NUMERIC_ROW = re.compile(r"^\s*\d")
RE_LOOP_TIME = re.compile(r"^Loop time")
RE_TOTAL_WALLTIME = re.compile(r"Total wall time", re.IGNORECASE)

# LSF footer patterns
RE_SUBMITTED = re.compile(r"was submitted from host.*?at (.+)$")
RE_STARTED = re.compile(r"Job was executed on host.*?at (.+)$")
RE_TERMINATED = re.compile(r"Terminated at (.+)$")
RE_RESULTS = re.compile(r"Results reported at (.+)$")
RE_EXIT_CODE = re.compile(r"Exited with exit code (\d+)")
RE_TERM_RUNLIMIT = re.compile(r"TERM_RUNLIMIT")
RE_CPU_TIME = re.compile(r"CPU time\s*:\s*([\d.]+)\s*sec", re.IGNORECASE)
RE_MAX_MEM = re.compile(r"Max Memory\s*:\s*(.*)", re.IGNORECASE)
RE_AVG_MEM = re.compile(r"Average Memory\s*:\s*(.*)", re.IGNORECASE)
RE_REQ_MEM = re.compile(r"Total Requested Memory\s*:\s*(.*)", re.IGNORECASE)
RE_WORKDIR = re.compile(r"<(.+?)> was used as the working directory")
RE_JOB_NAME = re.compile(r"Job <(.+?)> was submitted")
RE_QUEUE = re.compile(r"in queue <(.+?)>")
RE_HOST = re.compile(r"Job was executed on host\(s\) <(.+?)>")
RE_RUN_TIME = re.compile(r"Run time\s*:\s*([\d.]+)\s*sec", re.IGNORECASE)


@dataclass
class ThermoResult:
    """All thermo data and job metadata parsed from one out/log file."""

    sim_id: str
    source_file: str
    thermo_df: pd.DataFrame       # one row per thermo step
    job_metadata: Dict[str, Any]  # scheduler/run info
    warnings: List[str] = field(default_factory=list)


def find_out_file(input_dir: str | Path, sim_id: str, out_prefix: str = "out.") -> Optional[Path]:
    """Return the matching out file path, or None if not found."""
    candidate = Path(input_dir) / f"{out_prefix}{sim_id}"
    return candidate if candidate.is_file() else None


def parse_out_file(path: str | Path, sim_id: str = "") -> ThermoResult:
    """Parse a LAMMPS out/log file and return a :class:`ThermoResult`.

    Multiple thermo blocks in the same file are concatenated. Each row gets
    metadata columns: sim_id, source_file, stage, segment, total_segments,
    a12, thermo_block_index, line_number.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Out file not found: {path}")

    rows: List[Dict[str, Any]] = []
    job_meta: Dict[str, Any] = {
        "sim_id": sim_id,
        "source_file": str(path),
        "submitted_time": None,
        "started_time": None,
        "terminated_time": None,
        "results_reported_time": None,
        "exit_code": None,
        "normal_completion": False,
        "term_runlimit": False,
        "cpu_time_sec": None,
        "max_memory": None,
        "avg_memory": None,
        "requested_memory": None,
        "run_time_sec": None,
        "working_directory": None,
        "job_name": None,
        "queue": None,
        "host": None,
    }
    parse_warnings: List[str] = []

    # Current stage context
    current_stage: Optional[int] = None
    current_seg: Optional[int] = None
    current_total_segs: Optional[int] = None
    current_a12: Optional[float] = None
    thermo_block_idx: int = -1
    current_header: Optional[List[str]] = None
    in_footer = False

    with path.open("r", errors="replace") as fh:
        lines = fh.readlines()

    for lineno, raw in enumerate(lines, start=1):
        line = raw.rstrip("\n")

        # --- footer detection ---
        if "Sender: LSF System" in line or "----" in line and "Sender:" in "".join(lines[max(0, lineno):lineno+3]):
            in_footer = True

        if in_footer:
            _parse_footer_line(line, job_meta)
            continue

        # --- stage marker ---
        m = RE_STAGE.search(line)
        if m:
            current_stage = int(m.group("stage"))
            current_seg = int(m.group("seg"))
            current_total_segs = int(m.group("total_segs"))
            current_a12 = float(m.group("a12"))
            continue

        # --- normal completion ---
        if RE_TOTAL_WALLTIME.search(line):
            job_meta["normal_completion"] = True
            continue

        # --- thermo header ---
        if RE_THERMO_HEADER.match(line):
            thermo_block_idx += 1
            current_header = line.split()
            continue

        # --- thermo data row ---
        if current_header and RE_NUMERIC_ROW.match(line):
            parts = line.split()
            if len(parts) < len(current_header):
                continue  # malformed row
            try:
                row: Dict[str, Any] = {
                    col: float(parts[i]) for i, col in enumerate(current_header)
                }
            except ValueError:
                continue
            # Cast Step to int
            if "Step" in row:
                row["Step"] = int(row["Step"])
            # Metadata
            row["sim_id"] = sim_id
            row["source_file"] = str(path)
            row["stage"] = current_stage
            row["segment"] = current_seg
            row["total_segments"] = current_total_segs
            row["a12_target"] = current_a12
            row["thermo_block_index"] = thermo_block_idx
            row["line_number"] = lineno
            rows.append(row)
            continue

        # --- end of thermo block ---
        if RE_LOOP_TIME.match(line):
            current_header = None
            continue

    # Build DataFrame
    if rows:
        df = pd.DataFrame(rows)
        # Ensure Step is the first column
        cols = ["Step"] + [c for c in df.columns if c != "Step"]
        df = df[[c for c in cols if c in df.columns]]
    else:
        df = pd.DataFrame()
        parse_warnings.append(f"No thermo data found in {path}")

    # Check for incomplete last stage
    if current_total_segs is not None and current_seg is not None:
        if current_seg < current_total_segs and not job_meta["normal_completion"]:
            parse_warnings.append(
                f"Last parsed segment {current_seg}/{current_total_segs} — "
                "simulation likely did not complete all stages."
            )

    return ThermoResult(
        sim_id=sim_id,
        source_file=str(path),
        thermo_df=df,
        job_metadata=job_meta,
        warnings=parse_warnings,
    )


def _parse_footer_line(line: str, meta: Dict[str, Any]) -> None:
    """Extract scheduler metadata from a single LSF footer line."""
    if m := RE_SUBMITTED.search(line):
        meta["submitted_time"] = m.group(1).strip()
    if m := RE_STARTED.search(line):
        meta["started_time"] = m.group(1).strip()
    if m := RE_TERMINATED.search(line):
        meta["terminated_time"] = m.group(1).strip()
    if m := RE_RESULTS.search(line):
        meta["results_reported_time"] = m.group(1).strip()
    if m := RE_EXIT_CODE.search(line):
        meta["exit_code"] = int(m.group(1))
    if RE_TERM_RUNLIMIT.search(line):
        meta["term_runlimit"] = True
    if m := RE_CPU_TIME.search(line):
        meta["cpu_time_sec"] = float(m.group(1))
    if m := RE_MAX_MEM.search(line):
        meta["max_memory"] = m.group(1).strip()
    if m := RE_AVG_MEM.search(line):
        meta["avg_memory"] = m.group(1).strip()
    if m := RE_REQ_MEM.search(line):
        meta["requested_memory"] = m.group(1).strip()
    if m := RE_RUN_TIME.search(line):
        meta["run_time_sec"] = float(m.group(1))
    if m := RE_WORKDIR.search(line):
        meta["working_directory"] = m.group(1).strip()
    if m := RE_JOB_NAME.search(line):
        meta["job_name"] = m.group(1).strip()
    if m := RE_QUEUE.search(line):
        meta["queue"] = m.group(1).strip()
    if m := RE_HOST.search(line):
        meta["host"] = m.group(1).strip()


def termination_status(meta: Dict[str, Any]) -> str:
    """Return a one-line termination status string."""
    if meta.get("normal_completion"):
        return "normal"
    if meta.get("term_runlimit"):
        return "TERM_RUNLIMIT (wall-time kill)"
    ec = meta.get("exit_code")
    if ec is not None and ec != 0:
        return f"nonzero_exit_code_{ec}"
    return "incomplete_or_unknown"


def save_job_metadata(meta: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    with path.open("w") as fh:
        json.dump(meta, fh, indent=2, default=str)
