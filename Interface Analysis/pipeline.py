"""Per-simulation orchestration: frames -> per-frame analysis -> outputs.

Output layout per simulation (never touches S(q) outputs):

  <output_root>/<sim_id>/
      interface_summary.json      frame-aggregated metrics + provenance
      interface_summary.csv       one-row summary (batch-friendly)
      interface_per_frame.csv     per-frame scalar metrics
      interface_per_surface.csv   per-frame x per-interface metrics
      interface_parameters.json   resolved config (every parameter logged)
      interface_resolution.json   grid/slab/orientation resolution provenance
      interface_warnings.json     all warnings (per frame + summary)
      interface_profiles.npz      aligned/naive profiles, rho(s), spectra
      density_profiles.csv/.npz   A_head/A_tail/B_tail profiles (optional)
      density_profile_metrics.csv per-frame profile diagnostics (optional)
      interface_maps.npz          last-frame maps (heights, spacing, width, ...)
      topology_by_threshold.csv   per-frame threshold sweep
      frame_read_report.json      selected/parsed/failed frame accounting
      plots/                      diagnostic figures
      logs/                       timestamped run log
"""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from .config import config_hash, is_auto, parse_filename_metadata, parse_sim_id
from .density_profiles import aggregate_density_profiles, plot_density_profile_summary
from .frame_analysis import FrameAnalysis, analyze_frame
from .io_utils import SimLogger, make_sim_output_dirs, provenance_block, write_json
from .metrics import summarize_frame_rows
from .sq_bridge import load_sq_results
from .trajectory import (
    FrameReadReport,
    get_index,
    iter_selected_frames,
    select_records,
)

AUTOCORR_KEYS = ["waviness_rms", "width_lam_corrected_mean", "spacing_mean",
                 "bad_column_fraction", "P_A_to_B"]


def analyze_simulation_interfaces(
    traj_path: str | Path,
    cfg: Dict[str, Any],
    output_root: str | Path,
    echo: Optional[Callable[[str], None]] = print,
) -> Dict[str, Any]:
    """Run the interface pipeline for one trajectory; returns the summary row."""
    traj_path = Path(traj_path)
    output_root = Path(output_root)
    sim_id = parse_sim_id(traj_path.name, cfg["traj_suffix"])
    sim_meta = parse_filename_metadata(sim_id)
    dirs = make_sim_output_dirs(output_root, sim_id)
    log = SimLogger(dirs["logs"] / "interface_analysis_v2.log",
                    echo=(lambda m: echo(f"[{sim_id}] {m}")) if echo else None)

    try:
        return _analyze(traj_path, cfg, sim_id, sim_meta, dirs, output_root, log)
    except Exception as exc:
        log.error(f"simulation analysis FAILED: {exc}")
        log(traceback.format_exc())
        return {"sim_id": sim_id, **sim_meta, "traj_file": str(traj_path),
                "n_frames_analyzed": 0, "status": "error",
                "error": str(exc), "warnings": "; ".join(log.warnings)}
    finally:
        log.close()


def _stack_profiles_on_common_grid(grids: List[np.ndarray],
                                   profiles: List[np.ndarray]):
    """Interpolate frame profiles onto one physical-coordinate grid.

    ``dw`` changes slightly when each frame's normal extent is rounded to an
    integer sample count.  Index-aligned padding therefore misregisters the
    profiles.  This helper preserves physical s/w coordinates and uses NaN
    outside each frame's measured support.
    """
    pairs = [(np.asarray(g, float), np.asarray(p, float))
             for g, p in zip(grids, profiles) if len(g) == len(p) and len(g) > 0]
    if not pairs:
        return np.zeros(0), np.zeros((0, 0))
    steps = [float(np.median(np.diff(g))) for g, _ in pairs if len(g) > 1]
    step = min(abs(s) for s in steps if s != 0) if steps else 1.0
    lo = min(float(g[0]) for g, _ in pairs)
    hi = max(float(g[-1]) for g, _ in pairs)
    ilo = int(np.floor(lo / step))
    ihi = int(np.ceil(hi / step))
    common = np.arange(ilo, ihi + 1, dtype=float) * step
    stack = np.full((len(pairs), len(common)), np.nan)
    for i, (grid, profile) in enumerate(pairs):
        finite = np.isfinite(grid) & np.isfinite(profile)
        if finite.sum() < 2:
            continue
        stack[i] = np.interp(common, grid[finite], profile[finite],
                             left=np.nan, right=np.nan)
    return common, stack


def _columnwise_nanmean(stack: np.ndarray) -> np.ndarray:
    """NaN mean without warnings for columns unsupported by every frame."""
    if stack.ndim != 2 or stack.shape[0] == 0:
        return np.zeros(stack.shape[1] if stack.ndim == 2 else 0)
    count = np.sum(np.isfinite(stack), axis=0)
    total = np.nansum(stack, axis=0)
    return np.divide(total, count, out=np.full(stack.shape[1], np.nan), where=count > 0)


def _resolve_start_step(cfg: Dict[str, Any], sq: Dict[str, Any],
                        index, log: SimLogger) -> Optional[int]:
    """analysis_start_step: explicit int | null | 'auto' (S(q) window, else burn-in)."""
    raw = cfg.get("analysis_start_step")
    if raw is None:
        log("analysis_start_step=null: analyzing the whole file.")
        return None
    if not is_auto(raw):
        log(f"analysis_start_step={int(raw)} (explicit).")
        return int(raw)
    if sq.get("window_start_step") is not None:
        start = int(sq["window_start_step"])
        log(f"analysis_start_step=auto: using S(q) accepted-window start {start}.")
        return start
    n = index.n_frames
    cut = min(n - 1, int(np.ceil(n * float(cfg["burn_in_fraction"]))))
    start = index.records[cut].timestep
    log(f"analysis_start_step=auto: no S(q) window available; burn-in "
        f"{cfg['burn_in_fraction']:.0%} of {n} frames -> start step {start}.")
    return start


def _analyze(traj_path, cfg, sim_id, sim_meta, dirs, output_root, log: SimLogger
             ) -> Dict[str, Any]:
    sim_out: Path = dirs["root"]
    log(f"Starting interface_analysis_v2: {traj_path.name}; morphology=lamella")
    log(f"config hash {config_hash(cfg)}")

    # ---- S(q) results (read-only) ----
    sq = load_sq_results(cfg.get("sq_results_dir"), sim_id)
    for note in sq["notes"]:
        log(f"S(q) bridge: {note}")
    if sq["q_star"]:
        log(f"S(q) results: q_star={sq['q_star']:.4f}, L0={sq['L0']:.3f} "
            f"(from {sq['source']})")

    # ---- frame selection ----
    index = get_index(traj_path, cache_dir=output_root / "_traj_index",
                      use_cache=bool(cfg.get("use_frame_index", True)), log=log)
    if index.n_frames == 0:
        raise RuntimeError("no frames found in trajectory")
    report = FrameReadReport(traj_path=str(traj_path))
    start_step = _resolve_start_step(cfg, sq, index, log)
    records = select_records(index, start_step, int(cfg["frame_stride"]),
                             cfg.get("max_frames"), report,
                             last_step_window=cfg.get("last_step_window"))
    for note in report.notes:
        log(note)
    log(f"selected {report.n_selected} of {report.n_frames_in_file} frames "
        f"(stride {cfg['frame_stride']}, start_step {start_step}, "
        f"last_step_window {cfg.get('last_step_window')})")
    if not records:
        raise RuntimeError(
            f"no frames selected (start_step={start_step}); the trajectory may end "
            "before this timestep — check analysis_start_step / sq_results_dir")

    # ---- per-frame loop ----
    frame_rows: List[Dict[str, Any]] = []
    iface_rows: List[Dict[str, Any]] = []
    topo_rows: List[Dict[str, Any]] = []
    all_warnings: Dict[str, List[str]] = {}
    aligned_profiles: List[np.ndarray] = []
    aligned_grids: List[np.ndarray] = []
    naive_profiles: List[np.ndarray] = []
    naive_grids: List[np.ndarray] = []
    rhoA_s: List[np.ndarray] = []
    rhoB_s: List[np.ndarray] = []
    penetration_grids: List[np.ndarray] = []
    density_profile_results = []
    density_profile_timesteps: List[int] = []
    density_profile_metric_rows: List[Dict[str, Any]] = []
    n0_list: List[np.ndarray] = []
    last_detail: Optional[FrameAnalysis] = None
    s_grid = None
    s_bins = None
    naive_w = None

    for k, frame in enumerate(iter_selected_frames(
            traj_path, records, report, int(cfg["A_type_id"]), int(cfg["B_type_id"]),
            log=log, allow_variable_natoms=bool(cfg["allow_variable_natoms"]))):
        # Retain detail for the most recent successfully analyzed frame. Only
        # one heavy frame is live at a time, and a malformed final selected
        # record can no longer suppress maps/diagnostics for the last success.
        fa = analyze_frame(frame, cfg, sq_q_star=sq.get("q_star"),
                           keep_detail=True, log=log)
        fa.row.setdefault("morphology", "lamella")
        for surface_row in fa.interface_rows:
            surface_row.setdefault("morphology", "lamella")
        for topology_row in fa.topology_rows:
            topology_row.setdefault("morphology", "lamella")
        frame_rows.append(fa.row)
        iface_rows.extend(fa.interface_rows)
        topo_rows.extend(fa.topology_rows)
        if fa.warnings:
            all_warnings[str(fa.timestep)] = fa.warnings
        if fa.width is not None and np.all(np.isfinite(fa.width.aligned_mean_profile)):
            aligned_profiles.append(fa.width.aligned_mean_profile)
            aligned_grids.append(fa.width.s_grid.copy())
            naive_profiles.append(fa.width.naive_profile)
            naive_grids.append(fa.width.naive_profile_w.copy())
            s_grid = fa.width.s_grid
            naive_w = fa.width.naive_profile_w
        if fa.penetration is not None:
            rhoA_s.append(fa.penetration.rhoA_of_s)
            rhoB_s.append(fa.penetration.rhoB_of_s)
            penetration_grids.append(fa.penetration.s_bins.copy())
            s_bins = fa.penetration.s_bins
        if fa.density_profile is not None:
            density_profile_results.append(fa.density_profile)
            density_profile_timesteps.append(int(fa.timestep))
            dd = fa.density_profile.diagnostics
            density_profile_metric_rows.append({
                "timestep": int(fa.timestep),
                **fa.density_profile.metrics,
                "alignment_shift_bins": int(fa.density_profile.shift_bins),
                "lattice_mode": str(dd.get("lattice_mode_canonical")),
                "lamellar_repeats_stacked": dd.get("lamellar_repeats_stacked"),
                "chain_length": dd.get("chain_length"),
                "pure_a_length": dd.get("pure_a_length"),
                "topology_source": dd.get("topology_source"),
            })
        n0_list.append(np.array([fa.row["n0_x"], fa.row["n0_y"], fa.row["n0_z"]]))
        last_detail = fa

    if not frame_rows:
        raise RuntimeError(
            "all selected frames failed to parse — see frame_read_report.json")
    if report.failed_frames:
        log.warn(f"{len(report.failed_frames)} frame(s) failed to parse and were "
                 "skipped (recorded in frame_read_report.json).")

    # orientation stability across frames
    n0_arr = np.array(n0_list)
    n0_ref = n0_arr[0]
    cosines = np.abs(n0_arr @ n0_ref)
    max_drift_deg = float(np.degrees(np.arccos(np.clip(cosines.min(), 0, 1))))
    if max_drift_deg > 10.0:
        log.warn(f"lamellar normal drifts up to {max_drift_deg:.1f} deg across frames; "
                 "per-frame n0 is used, but cross-frame comparisons mix orientations.")

    # ---- aggregate ----
    summary_stats = summarize_frame_rows(frame_rows, autocorr_keys=AUTOCORR_KEYS)
    prov = provenance_block(config_hash(cfg))
    last_res = last_detail.resolution if last_detail else {}
    ori_diag = last_res.get("orientation", {})

    summary: Dict[str, Any] = {
        "sim_id": sim_id,
        **sim_meta,
        "traj_file": str(traj_path),
        "status": "ok",
        "morphology": "lamella",
        "n_frames_analyzed": len(frame_rows),
        "n_frames_failed": len(report.failed_frames),
        "timestep_first": frame_rows[0]["timestep"],
        "timestep_last": frame_rows[-1]["timestep"],
        "frame_stride": int(cfg["frame_stride"]),
        "analysis_start_step": start_step,
        "last_step_window": cfg.get("last_step_window"),
        "grid_shape": str(last_res.get("grid", {}).get("grid_shape")),
        "voxel_size": str(last_res.get("grid", {}).get("actual_voxel_size")),
        "smoothing_sigma": float(cfg["smoothing_sigma"]),
        "lamellar_normal_last": str(np.round(n0_arr[-1], 4).tolist()),
        "lattice_mode_last": str(ori_diag.get("lattice_mode")),
        "normal_window_periodic_last": last_res.get("slab", {}).get("w_periodic"),
        "normal_closure_periods_last": last_res.get("slab", {}).get("closure_periods"),
        "n0_method": (last_detail.orientation.method if last_detail and last_detail.orientation
                      else "fft_phi_dominant_mode"),
        "n0_max_drift_deg": max_drift_deg,
        "sq_q_star": sq.get("q_star"),
        "sq_L0": sq.get("L0"),
        "density_profile_enabled": bool(cfg.get("density_profile_enabled", False)),
        "density_profile_frames": len(density_profile_results),
        "note_frames_correlated": ("consecutive frames are not statistically "
                                   "independent; see *_n_eff columns"),
        **summary_stats,
        "warnings": "; ".join(log.warnings),
    }

    # ---- write outputs ----
    pd.DataFrame(frame_rows).to_csv(sim_out / "interface_per_frame.csv", index=False)
    pd.DataFrame(iface_rows).to_csv(sim_out / "interface_per_surface.csv", index=False)
    pd.DataFrame(topo_rows).to_csv(sim_out / "topology_by_threshold.csv", index=False)
    pd.DataFrame([summary]).to_csv(sim_out / "interface_summary.csv", index=False)
    write_json({"summary": summary, "provenance": prov},
               sim_out / "interface_summary.json")
    write_json({"config": {k: v for k, v in cfg.items()},
                "config_hash": config_hash(cfg), "provenance": prov},
               sim_out / "interface_parameters.json")
    write_json({"per_frame_resolution_last": last_res,
                "sq_bridge": sq,
                "n0_per_frame": [v.tolist() for v in n0_list]},
               sim_out / "interface_resolution.json")
    write_json({"summary_warnings": log.warnings, "per_frame": all_warnings},
               sim_out / "interface_warnings.json")
    write_json(report.as_dict(), sim_out / "frame_read_report.json")

    # Architecture-resolved profiles. Each frame has already been stacked
    # over equivalent lamellar repeats and phase-aligned to the A-rich peak.
    density_profile_df = None
    if density_profile_results:
        density_profile_df, density_arrays = aggregate_density_profiles(
            density_profile_results, density_profile_timesteps
        )
        density_profile_df.to_csv(sim_out / "density_profiles.csv", index=False)
        pd.DataFrame(density_profile_metric_rows).to_csv(
            sim_out / "density_profile_metrics.csv", index=False
        )
        np.savez_compressed(sim_out / "density_profiles.npz", **density_arrays)

    # profiles npz
    # Per-frame profile grids can differ by a sample or two (s_max scales with
    # each frame's slightly fluctuating L0; naive_w/s_bins length similarly
    # tracks the per-frame slab/window size), so stacking into one array
    # requires NaN-padding onto a common, centered grid rather than relying
    # on np.array() to infer a uniform shape.
    if bool(cfg.get("save_profiles", True)) and s_grid is not None:
        s_grid_common, aligned_stack = _stack_profiles_on_common_grid(
            aligned_grids, aligned_profiles)
        naive_w_common, naive_stack = (_stack_profiles_on_common_grid(naive_grids, naive_profiles)
                                       if naive_w is not None else (np.zeros(0), np.zeros((0, 0))))
        s_bins_common, rhoA_stack = (_stack_profiles_on_common_grid(penetration_grids, rhoA_s)
                                     if s_bins is not None else (np.zeros(0), np.zeros((0, 0))))
        _, rhoB_stack = (_stack_profiles_on_common_grid(penetration_grids, rhoB_s)
                        if s_bins is not None else (np.zeros(0), np.zeros((0, 0))))
        np.savez_compressed(
            sim_out / "interface_profiles.npz",
            s_grid=s_grid_common,
            aligned_profiles=aligned_stack,
            aligned_profile_mean=_columnwise_nanmean(aligned_stack),
            naive_w=naive_w_common,
            naive_profiles=naive_stack,
            s_bins=s_bins_common,
            rhoA_of_s=rhoA_stack,
            rhoB_of_s=rhoB_stack,
            aggregation_method=np.array("physical-coordinate linear interpolation; no extrapolation"),
        )

    # maps npz (last analyzed frame)
    if bool(cfg.get("save_maps", True)) and last_detail is not None and last_detail.iset is not None:
        det = last_detail
        maps: Dict[str, Any] = {
            "timestep": np.array([det.timestep]),
            "u": det.slab.u, "v": det.slab.v, "w": det.slab.w,
            "heights": det.iset.h,
            "plane_type": det.iset.plane_type,
            "plane_ideal": det.iset.plane_ideal,
            "column_class": det.iset.column_class,
            "crossing_count": det.iset.crossing_count,
            "misorientation_map_deg": det.orient_order.misorientation_map,
            "map_fB_in_A_core": det.penetration.map_fB_in_A_core,
            "map_fA_in_B_core": det.penetration.map_fA_in_B_core,
        }
        if det.spacing.local_maps:
            maps["spacing_maps"] = np.array(det.spacing.local_maps)
        if det.width is not None and det.width.width_maps:
            maps["width_maps"] = np.array(det.width.width_maps)
        if det.topology is not None and det.topology.connectivity_view is not None:
            maps["topology_labels_A_phi0"] = det.topology.connectivity_view
        np.savez_compressed(sim_out / "interface_maps.npz", **maps)

    # plots
    if bool(cfg.get("plots", True)) and last_detail is not None:
        try:
            from .diagnostics import make_all_plots
            make_all_plots(sim_id, dirs["plots"], last_detail,
                           pd.DataFrame(frame_rows), log=log)
        except Exception as exc:
            log.error(f"plotting failed: {exc}")
            log(traceback.format_exc())
        if density_profile_df is not None:
            try:
                plot_density_profile_summary(
                    sim_id, dirs["plots"], density_profile_df, log=log
                )
            except Exception as exc:
                log.error(f"density-profile plotting failed: {exc}")
                log(traceback.format_exc())

    log(f"Done: {len(frame_rows)} frames; waviness_rms="
        f"{summary.get('waviness_rms_mean', float('nan')):.3f}, "
        f"width_lam={summary.get('width_lam_corrected_mean_mean', float('nan')):.3f}, "
        f"f_bad={summary.get('bad_column_fraction_mean', float('nan')):.3f}")
    return summary
