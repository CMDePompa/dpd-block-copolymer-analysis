#!/usr/bin/env python
"""Regenerate v4 diagnostic plots from saved CSV outputs, without rereading dumps.

Use this after plot-only code changes:

    python refresh_saved_plots.py --output-dir /path/to/sq4_output
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from pipeline.plotting import plot_composition_sq, plot_log_sq_screening, plot_window_selection
from pipeline.config import load_config


def _number(row, key, default=np.nan):
    value = row.get(key, default)
    return float(default) if pd.isna(value) else float(value)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, help="Existing v4 output root")
    parser.add_argument("--config", help="Config supplying stationarity tolerances")
    args = parser.parse_args(argv)
    root = Path(args.output_dir).expanduser().resolve()
    cfg = load_config(args.config) if args.config else {}

    refreshed = 0
    reassessed = []
    for sim_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        required = [
            sim_dir / "analysis_summary.csv",
            sim_dir / "frame_metrics.csv",
            sim_dir / "sq_composition.csv",
        ]
        if not all(path.is_file() for path in required):
            continue

        summary = pd.read_csv(required[0]).iloc[0]
        frames = pd.read_csv(required[1])
        comp = pd.read_csv(required[2])
        thermo_path = sim_dir / "thermo_timeseries.csv"
        thermo = pd.read_csv(thermo_path) if thermo_path.is_file() else None
        sim_id = str(summary["sim_id"])
        q = comp["q"].to_numpy(float)
        raw = comp["S_comp_contrast"].to_numpy(float)
        normalized = comp["S_cc_normalized"].to_numpy(float)
        q_star = _number(summary, "q_star_sq")
        L0 = _number(summary, "L0_sq")
        q_lo = _number(summary, "resolved_q_compute_min", np.nanmin(q))
        q_hi = _number(summary, "resolved_q_peak_search_max", np.nanmax(q))

        plot_dir = sim_dir / "diagnostic_plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_composition_sq(
            q, raw, q_star, L0,
            plot_dir / f"{sim_id}_composition_sq.png",
            sim_id=sim_id,
            q_plot_min=q_lo,
            q_plot_max=q_hi,
            peak_detection_method=str(summary.get("peak_detection_method", "")),
            n_peak_candidates=int(summary.get("n_peak_candidates", 0)),
            sq_method=str(summary.get("sq_method_used", "")),
        )

        window = SimpleNamespace(
            window_start_step=int(summary["window_start_step"]),
            window_end_step=int(summary["window_end_step"]),
            fallback_used=bool(summary["fallback_used"]),
        )
        plot_window_selection(
            frames["timestep"].astype(int).tolist(),
            frames["q_star_fft"].tolist(),
            frames["phi_rms"].tolist(),
            window, None,
            plot_dir / f"{sim_id}_window_selection.png",
            sim_id=sim_id,
            thermo_df=thermo,
            peak_height_fft=frames["peak_height_fft"].tolist(),
        )

        plot_log_sq_screening(
            q, raw, normalized,
            root / "plots" / f"{sim_id}_sq_log.png",
            q_star=q_star,
            L0=L0,
            peak_detected=bool(summary.get("peak_detected", False)),
            sim_id=sim_id,
            sq_method=str(summary.get("sq_method_used", "")),
            q_plot_min=q_lo,
            q_plot_max=q_hi,
        )

        block_path = sim_dir / "sq_block_spectra.csv"
        metric_path = sim_dir / "sq_block_metrics.csv"
        if block_path.is_file() and metric_path.is_file():
            blocks_long = pd.read_csv(block_path)
            metrics = pd.read_csv(metric_path).sort_values("block_id")
            grouped = [
                group.sort_values("q")
                for _, group in blocks_long.groupby("block_id", sort=True)
            ]
            min_blocks = int(cfg.get("sq_stationarity_min_blocks", 3))
            distance_tolerance = float(
                cfg.get("sq_spectral_distance_tolerance", 0.10)
            )
            peak_slope_tolerance = float(
                cfg.get("sq_peak_intensity_slope_tolerance", 0.02)
            )
            general_slope_tolerance = float(cfg.get("metric_slope_tolerance", 0.02))
            peak_values = []
            block_spectra = []
            for group in grouped:
                bq = group["q"].to_numpy(float)
                bs = group["S_cc_normalized"].to_numpy(float)
                block_spectra.append(bs)
                if np.isfinite(q_star):
                    peak_values.append(float(bs[np.nanargmin(np.abs(bq - q_star))]))

            peak_slope = np.nan
            if len(peak_values) >= min_blocks:
                x = metrics["block_mid_step"].to_numpy(float)
                y = np.asarray(peak_values, dtype=float)
                slope, _ = np.polyfit(x, y, 1)
                peak_slope = abs(slope) * (x.max() - x.min()) / abs(y.mean())

            edge_distance = np.nan
            if len(block_spectra) >= min_blocks:
                edge_n = max(1, min(min_blocks, len(block_spectra) // 2))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    first = np.nanmean(block_spectra[:edge_n], axis=0)
                    last = np.nanmean(block_spectra[-edge_n:], axis=0)
                finite = np.isfinite(first) & np.isfinite(last)
                edge_distance = float(np.sqrt(np.mean(
                    (np.log1p(np.maximum(first[finite], 0.0))
                     - np.log1p(np.maximum(last[finite], 0.0))) ** 2
                )))

            recent = metrics.tail(min_blocks)
            max_recent_distance = float(
                recent["log_spectrum_distance_from_previous"].iloc[1:].max()
            )
            x = recent["block_mid_step"].to_numpy(float)
            y = recent["low_q_integrated_S_cc_normalized"].to_numpy(float)
            slope, _ = np.polyfit(x, y, 1)
            low_q_slope = abs(slope) * (x.max() - x.min()) / abs(y.mean())
            peak_required = bool(summary.get("peak_detected", False))
            spectral_stationary = bool(
                len(grouped) >= min_blocks
                and max_recent_distance <= distance_tolerance
                and edge_distance <= distance_tolerance
                and low_q_slope <= general_slope_tolerance
                and (not peak_required or peak_slope <= peak_slope_tolerance)
            )
            combined = bool(summary["stable_window_found"]) and spectral_stationary
            record = {
                "sim_id": sim_id,
                "a12": summary.get("a12"),
                "original_window_stable": bool(summary["stable_window_found"]),
                "reassessed_sq_spectral_stationary": spectral_stationary,
                "stationarity_supported_by_all_checks": combined,
                "sq_peak_intensity_relative_slope": peak_slope,
                "sq_peak_intensity_slope_tolerance": peak_slope_tolerance,
                "sq_whole_window_edge_log_spectrum_distance": edge_distance,
                "sq_max_recent_log_spectrum_distance": max_recent_distance,
                "sq_recent_low_q_intensity_relative_slope": low_q_slope,
            }
            reassessed.append(record)
            with (sim_dir / "sq_stationarity_reassessed.json").open("w") as fh:
                json.dump(record, fh, indent=2, default=str)
            # Overwrite the window figure once more so the saved-data
            # reassessment is visible on the diagnostic itself.
            plot_window_selection(
                frames["timestep"].astype(int).tolist(),
                frames["q_star_fft"].tolist(),
                frames["phi_rms"].tolist(),
                window, None,
                plot_dir / f"{sim_id}_window_selection.png",
                sim_id=sim_id,
                thermo_df=thermo,
                peak_height_fft=frames["peak_height_fft"].tolist(),
                spectral_stationary=spectral_stationary,
            )
        refreshed += 1

    if reassessed:
        pd.DataFrame(reassessed).to_csv(
            root / "stationarity_reassessment.csv", index=False
        )
    print(f"Refreshed diagnostic plots for {refreshed} simulation(s) under {root}")
    if reassessed:
        print(f"Wrote stationarity reassessment for {len(reassessed)} simulation(s)")
    return 0 if refreshed else 1


if __name__ == "__main__":
    raise SystemExit(main())
