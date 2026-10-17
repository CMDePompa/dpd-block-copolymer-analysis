#!/usr/bin/env python
"""Run the pipeline over many trajectories in parallel, one process per file.

The per-file cost is dominated by single-threaded text parsing of the dump
(~0.6 s/frame, ~6 MB/s) while the disk sustains ~1 GB/s, so throughput scales
with worker count until the cores run out. Each worker is an independent
``run_pipeline.py --traj-file`` invocation writing into the shared output root;
a final aggregation pass merges the per-simulation summaries.

    python run_parallel.py --config cfg1.yaml [--config cfg2.yaml ...] -j 8

Progress and per-file logs go to <output_dir>/logs/<sim_id>.log.
"""

from __future__ import annotations

import argparse
import datetime
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import csv

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))


def discover(cfg_path: Path):
    """Return (cfg_path, traj_path, output_dir) for every trajectory in a config."""
    from pipeline.config import load_config

    cfg = load_config(cfg_path)
    input_dir = Path(cfg["input_dir"]).expanduser().resolve()
    output_dir = Path(cfg["output_dir"]).expanduser().resolve()
    suffix = cfg.get("traj_suffix", ".prod.lammpstrj")
    trajs = sorted(p for p in input_dir.iterdir() if p.is_file() and p.name.endswith(suffix))
    if not trajs:
        raise FileNotFoundError(f"No '{suffix}' files in {input_dir}")
    return [(cfg_path, t, output_dir, suffix) for t in trajs]


def run_one(job, python_exe: str, log_dir: Path, env: dict):
    cfg_path, traj, output_dir, suffix = job
    sim_id = traj.name[: -len(suffix)] if traj.name.endswith(suffix) else traj.stem
    log_path = log_dir / f"{sim_id}.log"
    cmd = [
        python_exe, str(HERE / "run_pipeline.py"),
        "--config", str(cfg_path),
        "--traj-file", str(traj),
    ]
    t0 = time.time()
    with log_path.open("w", encoding="utf-8", errors="replace") as fh:
        proc = subprocess.run(
            cmd, cwd=str(HERE), stdout=fh, stderr=subprocess.STDOUT,
            env=env, text=True,
        )
    return sim_id, proc.returncode, time.time() - t0, log_path


def summary_is_complete(path: Path) -> bool:
    """Return True only for a readable summary that explicitly completed."""
    try:
        with path.open(newline="", encoding="utf-8") as fh:
            row = next(csv.DictReader(fh), None)
        if not row or "analysis_complete" not in row:
            return False
        value = row["analysis_complete"]
        return str(value).strip().lower() in {"true", "1", "yes"}
    except Exception:
        return False


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", action="append", required=True,
                    help="Config YAML; repeat for multiple input directories.")
    ap.add_argument("-j", "--jobs", type=int, default=8,
                    help="Concurrent trajectories (default 8).")
    ap.add_argument("--threads-per-job", type=int, default=4,
                    help="Cap threads inside each worker so the parallel S(q) "
                         "phase does not oversubscribe the machine (default 4).")
    ap.add_argument("--python", default=sys.executable, help="Interpreter for workers.")
    ap.add_argument("--dry-run", action="store_true", help="List the work and exit.")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip trajectories that already have a per-simulation "
                         "analysis_summary.csv (resume an interrupted batch).")
    args = ap.parse_args(argv)

    jobs = []
    for c in args.config:
        jobs.extend(discover(Path(c).resolve()))

    if args.skip_existing:
        keep, skipped = [], 0
        for job in jobs:
            _, traj, output_dir, suffix = job
            sim_id = traj.name[: -len(suffix)] if traj.name.endswith(suffix) else traj.stem
            summary_path = output_dir / sim_id / "analysis_summary.csv"
            if summary_path.is_file() and summary_is_complete(summary_path):
                skipped += 1
            else:
                keep.append(job)
        print(f"--skip-existing: {skipped} already complete, {len(keep)} to run")
        jobs = keep
        if not jobs:
            print("Nothing to do.")
            return 0

    output_dirs = sorted({j[2] for j in jobs})
    log_dir = output_dirs[0] / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    total_bytes = sum(j[1].stat().st_size for j in jobs)
    print(f"{len(jobs)} trajectories, {total_bytes/1e9:.0f} GB total")
    print(f"output root(s): {', '.join(str(d) for d in output_dirs)}")
    print(f"workers: {args.jobs}   logs: {log_dir}")
    if args.dry_run:
        for _, t, _, _ in jobs:
            print(f"  {t.name}  ({t.stat().st_size/1e9:.1f} GB)")
        return 0

    env = dict(os.environ)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS",
                "OPENBLAS_NUM_THREADS", "TBB_NUM_THREADS"):
        env[var] = str(args.threads_per_job)

    # Largest files first: keeps the tail of the run from being one straggler.
    jobs.sort(key=lambda j: j[1].stat().st_size, reverse=True)

    t_start = time.time()
    done = failed = 0
    bytes_done = 0
    bytes_total = sum(j[1].stat().st_size for j in jobs)
    results = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(run_one, j, args.python, log_dir, env): j for j in jobs}
        for fut in as_completed(futures):
            sim_id, rc, dt, log_path = fut.result()
            done += 1
            bytes_done += futures[fut][1].stat().st_size
            status = "ok" if rc == 0 else f"FAILED rc={rc}"
            if rc != 0:
                failed += 1
            # ETA from bytes processed, not file count: files complete in waves,
            # so elapsed/done badly overestimates while the first wave lands.
            elapsed = time.time() - t_start
            rate = bytes_done / elapsed
            remaining = (bytes_total - bytes_done) / rate if rate > 0 else 0
            eta = datetime.timedelta(seconds=int(remaining))
            print(f"[{done}/{len(jobs)}] {sim_id}: {status} ({dt/60:.1f} min)  "
                  f"{rate/1e6:.0f} MB/s agg  ETA {eta}", flush=True)
            results.append((sim_id, rc, dt, log_path))

    print(f"\nAll workers finished in {(time.time()-t_start)/3600:.2f} h; {failed} failure(s).")
    for sim_id, rc, _, log_path in sorted(results):
        if rc != 0:
            print(f"  FAILED {sim_id} -> {log_path}")

    # Aggregate per-simulation summaries into one batch CSV per output root.
    from pipeline.batch_runner import aggregate_existing_outputs
    for out in output_dirs:
        try:
            path = aggregate_existing_outputs(out, logger=print)
            print(f"Aggregated: {path}")
        except Exception as exc:
            print(f"Aggregation failed for {out}: {exc}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
