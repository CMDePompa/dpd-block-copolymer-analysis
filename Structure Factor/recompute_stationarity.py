#!/usr/bin/env python
"""Re-evaluate the stationary-window verdict from saved per-frame metrics.

Window detection consumes only per-frame quantities that are already written to
``<sim>/frame_metrics.csv``, so a different choice of gating metrics can be
scored without re-reading trajectories or recomputing any S(q).

Nothing existing is overwritten. Results go to:
  <sim>/window_diagnostics_regated.csv     per-block metrics under the new gate
  <output_root>/stationarity_regated.csv   one row per simulation, old vs new

    python recompute_stationarity.py --output-dir <root> --metrics phi_rms thermo
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.resolve()))
from pipeline.config import load_config  # noqa: E402
from pipeline.window_detection import detect_stable_window  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", required=True, help="Pipeline output root.")
    ap.add_argument("--metrics", nargs="+", default=["phi_rms", "thermo"],
                    help="Gating metrics for the re-scored verdict.")
    ap.add_argument("--config", help="Config supplying the heuristic parameters "
                                     "(block size, tolerances). Defaults apply if omitted.")
    args = ap.parse_args(argv)

    root = Path(args.output_dir).resolve()
    cfg = load_config(args.config) if args.config else {}

    def p(key, default):
        return cfg.get(key, default)

    rows = []
    sims = sorted(d for d in root.iterdir() if d.is_dir() and d.name != "logs")
    for sim in sims:
        fm = sim / "frame_metrics.csv"
        summ = sim / "analysis_summary.csv"
        if not fm.is_file() or not summ.is_file():
            continue
        f = pd.read_csv(fm)
        s = pd.read_csv(summ).iloc[0]

        thermo_path = sim / "thermo_timeseries.csv"
        thermo_df = pd.read_csv(thermo_path) if thermo_path.is_file() else None

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            w = detect_stable_window(
                timesteps=f["timestep"].tolist(),
                q_star_fft=f["q_star_fft"].tolist(),
                phi_rms=f["phi_rms"].tolist(),
                peak_height_fft=f["peak_height_fft"].tolist(),
                burn_in_fraction=float(p("burn_in_fraction", 0.2)),
                block_size_steps=int(p("block_size_steps", 25000)),
                min_blocks_accepted=int(p("min_blocks_accepted", 4)),
                metric_slope_tolerance=float(p("metric_slope_tolerance", 0.02)),
                metric_cv_tolerance=float(p("metric_cv_tolerance", 0.05)),
                fallback_fraction=float(p("fallback_fraction", 0.30)),
                thermo_df=thermo_df,
                thermo_stationarity_cols=list(p("thermo_stationarity_cols",
                                                ["PotEng", "TotEng", "Press", "Temp"])),
                stationarity_metrics=list(args.metrics),
            )

        if not w.block_diagnostics.empty:
            w.block_diagnostics.to_csv(sim / "window_diagnostics_regated.csv", index=False)

        # Does the pinned Sq window now sit inside a genuinely detected window?
        fw_start = s.get("sq_window_start_step")
        if w.fallback_used or fw_start is None or pd.isna(fw_start):
            within = None
        else:
            within = bool(w.window_start_step <= fw_start)

        rows.append({
            "sim_id": s["sim_id"],
            "gating_metrics": ",".join(args.metrics),
            # original verdict (all four metrics gating)
            "orig_stable_window_found": bool(s["stable_window_found"]),
            "orig_fallback_used": bool(s["fallback_used"]),
            "orig_window_start_step": s["window_start_step"],
            "orig_window_end_step": s["window_end_step"],
            # re-scored verdict
            "regated_stable_window_found": bool(w.stable_found),
            "regated_fallback_used": bool(w.fallback_used),
            "regated_window_start_step": w.window_start_step,
            "regated_window_end_step": w.window_end_step,
            "regated_n_accepted_frames": w.n_accepted,
            # the S(q) results themselves are untouched by re-gating
            "sq_method_used": s.get("sq_method_used"),
            "sq_window_start_step": fw_start,
            "sq_window_end_step": s.get("sq_window_end_step"),
            "sq_window_within_stable_window_regated": within,
            "q_star_sq": s["q_star_sq"],
            "L0_sq": s["L0_sq"],
        })

    if not rows:
        print(f"No per-simulation outputs found under {root}")
        return 1

    out = pd.DataFrame(rows)
    path = root / "stationarity_regated.csv"
    out.to_csv(path, index=False)

    n = len(out)
    print(f"{n} simulations re-scored with gate = {args.metrics}")
    print(f"  stable window found: {int(out.orig_stable_window_found.sum())}/{n} "
          f"(original)  ->  {int(out.regated_stable_window_found.sum())}/{n} (re-gated)")
    print(f"  fallback used:       {int(out.orig_fallback_used.sum())}/{n} "
          f"(original)  ->  {int(out.regated_fallback_used.sum())}/{n} (re-gated)")
    inside = out.sq_window_within_stable_window_regated
    print(f"  pinned Sq window inside a detected stationary window: "
          f"{int((inside == True).sum())}/{n}; "
          f"unverifiable (fallback): {int(inside.isna().sum())}; "
          f"outside: {int((inside == False).sum())}")
    print(f"\nWrote {path}")
    print("Per-block detail: <sim>/window_diagnostics_regated.csv")
    print("Original outputs were not modified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
