"""Batch orchestration: discover trajectories, run per-simulation analysis, aggregate outputs."""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from .analysis_pipeline import analyze_simulation
from .config import config_hash, parse_sim_id, save_config


def run_batch(
    cfg: Dict[str, Any],
    logger: Optional[Callable[[str], None]] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Path:
    """Discover all trajectories in cfg['input_dir'] and run the full pipeline.

    Parameters
    ----------
    cfg:
        Loaded and validated config dict.
    logger:
        Called with each progress/info string.
    progress_callback:
        Called as (done, total, message) after each simulation completes.

    Returns
    -------
    Path to the batch-level analysis_summary.csv.
    """
    input_dir = Path(cfg["input_dir"]).expanduser().resolve()
    output_root = Path(cfg["output_dir"]).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    suffix = cfg.get("traj_suffix", ".prod.lammpstrj")

    def log(msg: str) -> None:
        if logger:
            logger(msg)

    # Save config for this run
    run_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_config(cfg, output_root / f"run_config_{run_ts}.yaml")

    # Discover trajectories
    trajs = sorted([p for p in input_dir.iterdir() if p.is_file() and p.name.endswith(suffix)])
    if not trajs:
        raise FileNotFoundError(
            f"No files ending in '{suffix}' found in {input_dir}. "
            "Check input_dir and traj_suffix in your config."
        )
    log(f"Found {len(trajs)} trajectories in {input_dir}")

    all_summaries: List[Dict[str, Any]] = []
    all_thermo_rows: List[pd.DataFrame] = []
    all_job_meta: List[Dict[str, Any]] = []
    done = 0

    for traj in trajs:
        sim_id = parse_sim_id(traj.name, suffix)
        log(f"\n--- [{done+1}/{len(trajs)}] {sim_id} ---")
        try:
            summary = analyze_simulation(traj, cfg, output_root, logger=logger)
        except Exception as exc:
            log(f"  FAILED: {exc}")
            summary = {
                "sim_id": sim_id,
                "traj_file": str(traj),
                "analysis_complete": False,
                "warnings": f"Uncaught error: {exc}",
                "q_star_sq": float("nan"),
                "L0_sq": float("nan"),
            }
        all_summaries.append(summary)

        # Collect thermo for batch aggregation
        thermo_file = output_root / sim_id / "thermo_timeseries.csv"
        if thermo_file.is_file():
            try:
                df = pd.read_csv(thermo_file)
                if "sim_id" not in df.columns:
                    df.insert(0, "sim_id", sim_id)
                all_thermo_rows.append(df)
            except Exception:
                pass

        # Collect job metadata
        meta_file = output_root / sim_id / "job_metadata.json"
        if meta_file.is_file():
            try:
                with meta_file.open() as fh:
                    meta = json.load(fh)
                all_job_meta.append(meta)
            except Exception:
                pass

        done += 1
        if progress_callback:
            progress_callback(done, len(trajs), f"Done: {sim_id}")

    # Write batch-level outputs
    batch_summary_df = pd.DataFrame(all_summaries)
    batch_summary_path = output_root / "analysis_summary.csv"
    batch_summary_df.to_csv(batch_summary_path, index=False)
    log(f"\nWrote batch summary: {batch_summary_path}")

    if all_thermo_rows:
        all_thermo_df = pd.concat(all_thermo_rows, ignore_index=True)
        thermo_batch_path = output_root / "all_thermo_timeseries.csv"
        all_thermo_df.to_csv(thermo_batch_path, index=False)
        log(f"Wrote all_thermo_timeseries.csv ({len(all_thermo_df)} rows)")

    if all_job_meta:
        job_status_rows = []
        for meta in all_job_meta:
            from .thermo_parser import termination_status
            row = {
                "sim_id": meta.get("sim_id", ""),
                "source_file": meta.get("source_file", ""),
                "termination_status": termination_status(meta),
                "exit_code": meta.get("exit_code"),
                "term_runlimit": meta.get("term_runlimit", False),
                "normal_completion": meta.get("normal_completion", False),
                "submitted_time": meta.get("submitted_time"),
                "started_time": meta.get("started_time"),
                "terminated_time": meta.get("terminated_time"),
                "cpu_time_sec": meta.get("cpu_time_sec"),
                "max_memory": meta.get("max_memory"),
                "avg_memory": meta.get("avg_memory"),
                "run_time_sec": meta.get("run_time_sec"),
                "working_directory": meta.get("working_directory"),
                "job_name": meta.get("job_name"),
                "queue": meta.get("queue"),
                "host": meta.get("host"),
            }
            job_status_rows.append(row)
        job_status_df = pd.DataFrame(job_status_rows)
        job_status_path = output_root / "all_job_status.csv"
        job_status_df.to_csv(job_status_path, index=False)
        log(f"Wrote all_job_status.csv ({len(job_status_df)} rows)")

    failed = [
        s.get("sim_id", "unknown") for s in all_summaries
        if not bool(s.get("analysis_complete", False))
    ]
    if failed:
        raise RuntimeError(
            f"{len(failed)} simulation(s) failed analysis: {failed}. "
            f"The error rows were written to {batch_summary_path}."
        )

    return batch_summary_path


def aggregate_existing_outputs(output_root: str | Path, logger: Optional[Callable[[str], None]] = None) -> Path:
    """Aggregate per-simulation analysis_summary.csv files already in output_root.

    This is useful after running many independent single-trajectory jobs, for
    example from an LSF job array. Per-simulation summaries live in
    output_root/<sim_id>/analysis_summary.csv.
    """
    output_root = Path(output_root).expanduser().resolve()

    def log(msg: str) -> None:
        if logger:
            logger(msg)

    summary_files = sorted(
        p for p in output_root.glob("*/analysis_summary.csv")
        if p.is_file()
    )
    if not summary_files:
        raise FileNotFoundError(f"No per-simulation analysis_summary.csv files found in {output_root}")

    rows = []
    for path in summary_files:
        try:
            rows.append(pd.read_csv(path))
        except Exception as exc:
            log(f"Skipping unreadable summary {path}: {exc}")

    if not rows:
        raise RuntimeError(f"Found summaries in {output_root}, but none could be read.")

    batch_summary_df = pd.concat(rows, ignore_index=True)
    batch_summary_path = output_root / "analysis_summary.csv"
    batch_summary_df.to_csv(batch_summary_path, index=False)
    log(f"Wrote aggregate summary: {batch_summary_path} ({len(batch_summary_df)} rows)")
    return batch_summary_path
