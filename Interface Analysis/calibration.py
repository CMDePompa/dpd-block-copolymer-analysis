"""Calibration mode: parameter-sensitivity scan on a few frames.

Runs the frame analysis over a small parameter grid on the LAST
``calibration_frames`` selected frames of each trajectory and reports
whether the headline metrics are stable:

  * full grid over (target_voxel_size x smoothing_sigma) — the two
    parameters that shape the field;
  * one-at-a-time scans of density_cutoff_fraction and phi_hysteresis from
    the base configuration.

For each metric the report gives the value at the base configuration, the
range over the scan, and a coefficient of variation; metrics with CV above
``UNSTABLE_CV`` are flagged. Widths are EXPECTED to grow with sigma (the
smoothing PSF); the report therefore also fits d(width)/d(sigma) so the
smoothing contribution is quantified rather than hidden.

Outputs (per simulation, under <output>/<sim_id>/):
  calibration_results.csv   one row per (frame, parameter combo)
  calibration_report.json   stability verdicts per metric
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .config import config_hash
from .frame_analysis import analyze_frame
from .io_utils import SimLogger, make_sim_output_dirs, provenance_block, write_json
from .pipeline import _resolve_start_step
from .sq_bridge import load_sq_results
from .trajectory import FrameReadReport, get_index, iter_selected_frames, select_records

UNSTABLE_CV = 0.15

METRICS = ["waviness_rms", "width_lam_corrected_mean", "width_10_90",
           "width_clean_column_fraction", "spacing_mean", "spacing_cv",
           "spacing_clean_coverage", "excess_area_mean", "S_n",
           "bad_column_fraction", "f_mix_voxel_0.5", "P_A_to_B", "P_B_to_A",
           "penetration_clean_coverage", "overlap_O", "topo_A0_beta0",
           "topo_A0_connection_score", "topo_A0_excess_connections"]


def calibrate_simulation(
    traj_path: str | Path,
    cfg: Dict[str, Any],
    output_root: str | Path,
    echo=print,
) -> Dict[str, Any]:
    """Run the sensitivity grid for one trajectory; returns the report dict."""
    from .config import parse_sim_id
    traj_path = Path(traj_path)
    output_root = Path(output_root)
    sim_id = parse_sim_id(traj_path.name, cfg["traj_suffix"])
    dirs = make_sim_output_dirs(output_root, sim_id)
    log = SimLogger(dirs["logs"] / "calibration.log",
                    echo=(lambda m: echo(f"[{sim_id}|cal] {m}")) if echo else None)

    try:
        # frame selection: the LAST calibration_frames of the normal selection
        sq = load_sq_results(cfg.get("sq_results_dir"), sim_id)
        index = get_index(traj_path, cache_dir=output_root / "_traj_index",
                          use_cache=bool(cfg.get("use_frame_index", True)), log=log)
        report = FrameReadReport(traj_path=str(traj_path))
        start_step = _resolve_start_step(cfg, sq, index, log)
        records = select_records(index, start_step, int(cfg["frame_stride"]),
                                 int(cfg["calibration_frames"]), report,
                                 last_step_window=cfg.get("last_step_window"))
        frames = list(iter_selected_frames(traj_path, records, report,
                                           int(cfg["A_type_id"]), int(cfg["B_type_id"]),
                                           log=log))
        if not frames:
            raise RuntimeError("no frames available for calibration")
        log(f"calibrating on {len(frames)} frame(s): "
            f"{[f.timestep for f in frames]}")

        combos = _build_combos(cfg)
        log(f"{len(combos)} parameter combinations per frame")

        rows: List[Dict[str, Any]] = []
        for frame in frames:
            for combo in combos:
                trial_cfg = dict(cfg)
                trial_cfg.update(combo["params"])
                fa = analyze_frame(frame, trial_cfg, sq_q_star=sq.get("q_star"),
                                   keep_detail=False, log=None)
                row = {"timestep": frame.timestep, "scan": combo["scan"],
                       **combo["params"]}
                for m in METRICS:
                    row[m] = fa.row.get(m, float("nan"))
                rows.append(row)
                log(f"  step {frame.timestep} {combo['scan']}: "
                    + ", ".join(f"{k}={v}" for k, v in combo["params"].items()))

        df = pd.DataFrame(rows)
        df.to_csv(dirs["root"] / "calibration_results.csv", index=False)

        rep = _stability_report(df, cfg, log)
        rep["provenance"] = provenance_block(config_hash(cfg))
        write_json(rep, dirs["root"] / "calibration_report.json")
        log("calibration complete.")
        return rep
    finally:
        log.close()


def _build_combos(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    base = {
        "target_voxel_size": float(cfg["target_voxel_size"]),
        "smoothing_sigma": float(cfg["smoothing_sigma"]),
        "density_cutoff_fraction": float(cfg["density_cutoff_fraction"]),
        "phi_hysteresis": float(cfg["phi_hysteresis"]),
    }
    combos: List[Dict[str, Any]] = []
    # full voxel x sigma grid
    for vox in cfg["calibration_voxel_sizes"]:
        for sig in cfg["calibration_sigmas"]:
            combos.append({"scan": "voxel_x_sigma",
                           "params": {**base, "target_voxel_size": float(vox),
                                      "smoothing_sigma": float(sig)}})
    # one-at-a-time scans from base
    for cut in cfg["calibration_cutoffs"]:
        combos.append({"scan": "cutoff",
                       "params": {**base, "density_cutoff_fraction": float(cut)}})
    for hyst in cfg["calibration_hysteresis"]:
        combos.append({"scan": "hysteresis",
                       "params": {**base, "phi_hysteresis": float(hyst)}})
    return combos


def _stability_report(df: pd.DataFrame, cfg: Dict[str, Any], log) -> Dict[str, Any]:
    base_vox = float(cfg["target_voxel_size"])
    base_sig = float(cfg["smoothing_sigma"])
    base_mask = (
        np.isclose(df["target_voxel_size"], base_vox)
        & np.isclose(df["smoothing_sigma"], base_sig)
        & np.isclose(df["density_cutoff_fraction"], float(cfg["density_cutoff_fraction"]))
        & np.isclose(df["phi_hysteresis"], float(cfg["phi_hysteresis"]))
    )
    report: Dict[str, Any] = {"unstable_cv_threshold": UNSTABLE_CV, "metrics": {}}
    for m in METRICS:
        vals = df[m].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        base_vals = df.loc[base_mask, m].to_numpy(dtype=float)
        base_vals = base_vals[np.isfinite(base_vals)]
        entry: Dict[str, Any] = {
            "base_value": float(np.mean(base_vals)) if base_vals.size else float("nan"),
            "scan_min": float(np.min(vals)) if vals.size else float("nan"),
            "scan_max": float(np.max(vals)) if vals.size else float("nan"),
            "scan_cv": (float(np.std(vals) / abs(np.mean(vals)))
                        if vals.size and np.mean(vals) != 0 else float("nan")),
        }
        entry["stable"] = bool(np.isfinite(entry["scan_cv"]) and entry["scan_cv"] <= UNSTABLE_CV)
        # width vs sigma slope (smoothing broadening, expected physical trend)
        if m in ("width_lam_corrected_mean", "width_10_90"):
            sub = df[df["scan"] == "voxel_x_sigma"]
            g = sub.groupby("smoothing_sigma")[m].mean()
            if len(g) >= 2 and np.all(np.isfinite(g.to_numpy())):
                slope = np.polyfit(g.index.to_numpy(dtype=float),
                                   g.to_numpy(dtype=float), 1)[0]
                entry["d_width_d_sigma"] = float(slope)
                entry["note"] = ("width grows with sigma by construction (smoothing PSF); "
                                 "compare systems at FIXED sigma only")
        report["metrics"][m] = entry
    n_unstable = sum(1 for e in report["metrics"].values() if not e["stable"])
    report["n_unstable_metrics"] = n_unstable
    report["verdict"] = (
        "conclusions stable across the scanned parameter grid"
        if n_unstable == 0 else
        f"{n_unstable} metric(s) vary by more than {UNSTABLE_CV:.0%} across the grid — "
        "inspect calibration_results.csv before comparing systems on those metrics")
    if log:
        log(f"calibration verdict: {report['verdict']}")
    return report
