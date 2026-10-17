"""Per-simulation pipeline for sphere, cylinder, and gyroid interfaces."""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from .config import config_hash, parse_filename_metadata, parse_sim_id
from .io_utils import SimLogger, make_sim_output_dirs, provenance_block, write_json
from .metrics import summarize_frame_rows
from .morphology_frame import MorphologyFrameAnalysis, analyze_morphology_frame
from .normal_profiles import profile_frame
from .pipeline import _columnwise_nanmean, _resolve_start_step, _stack_profiles_on_common_grid
from .sq_bridge import load_sq_results
from .trajectory import FrameReadReport, get_index, iter_selected_frames, select_records


AUTOCORR_KEYS = [
    "surface_area_per_volume", "surface_H_rms", "surface_K_negative_area_fraction",
    "normal_width_lam_mean", "overlap_O", "domain_component_count",
]


def analyze_simulation_morphology_interfaces(
    traj_path: str | Path,
    cfg: Dict[str, Any],
    output_root: str | Path,
    echo: Optional[Callable[[str], None]] = print,
) -> Dict[str, Any]:
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
        return {
            "sim_id": sim_id, **sim_meta, "traj_file": str(traj_path),
            "morphology": cfg.get("morphology"), "n_frames_analyzed": 0,
            "status": "error", "error": str(exc), "warnings": "; ".join(log.warnings),
        }
    finally:
        log.close()


def _aggregate_normal_profiles(results: List[MorphologyFrameAnalysis]):
    keys = sorted(set.intersection(*(set(r.profile.mean) for r in results if r.profile is not None)))
    out_df: Dict[str, np.ndarray] = {}
    arrays: Dict[str, np.ndarray] = {}
    common_s = None
    for key in keys:
        grids = [r.profile.s for r in results if r.profile is not None]
        profiles = [r.profile.mean[key] for r in results if r.profile is not None]
        common, stack = _stack_profiles_on_common_grid(grids, profiles)
        if common_s is None:
            common_s = common
            out_df["s"] = common
            arrays["s"] = common
        out_df[f"{key}_mean_frames"] = _columnwise_nanmean(stack)
        out_df[f"{key}_std_frames"] = (
            np.nanstd(stack, axis=0, ddof=1)
            if stack.shape[0] > 1 else np.full(stack.shape[1], np.nan)
        )
        arrays[f"{key}_per_frame"] = stack
        arrays[f"{key}_mean_frames"] = out_df[f"{key}_mean_frames"]
        arrays[f"{key}_std_frames"] = out_df[f"{key}_std_frames"]
    arrays["timesteps"] = np.asarray([r.timestep for r in results], dtype=np.int64)
    return pd.DataFrame(out_df), arrays


def _analyze(traj_path, cfg, sim_id, sim_meta, dirs, output_root,
             log: SimLogger) -> Dict[str, Any]:
    sim_out: Path = dirs["root"]
    morphology = str(cfg["morphology"]).lower()
    log(f"Starting interface_analysis_v2: {traj_path.name}; morphology={morphology}")
    log(f"config hash {config_hash(cfg)}")

    sq = load_sq_results(cfg.get("sq_results_dir"), sim_id)
    for note in sq["notes"]:
        log(f"S(q) bridge: {note}")
    if sq.get("q_star"):
        log(f"S(q) q_star={sq['q_star']:.4f}, L0={sq['L0']:.3f}")

    index = get_index(traj_path, cache_dir=Path(output_root) / "_traj_index",
                      use_cache=bool(cfg.get("use_frame_index", True)), log=log)
    if index.n_frames == 0:
        raise RuntimeError("no frames found in trajectory")
    report = FrameReadReport(traj_path=str(traj_path))
    start_step = _resolve_start_step(cfg, sq, index, log)
    records = select_records(
        index, start_step, int(cfg["frame_stride"]), cfg.get("max_frames"), report,
        last_step_window=cfg.get("last_step_window"),
    )
    for note in report.notes:
        log(note)
    if not records:
        raise RuntimeError("no frames selected; check analysis_start_step and S(q) paths")
    log(f"selected {report.n_selected} of {report.n_frames_in_file} frames")

    results: List[MorphologyFrameAnalysis] = []
    frame_rows: List[Dict[str, Any]] = []
    component_rows: List[Dict[str, Any]] = []
    topology_rows: List[Dict[str, Any]] = []
    per_frame_profile_tables: List[pd.DataFrame] = []
    all_warnings: Dict[str, List[str]] = {}
    last_detail: Optional[MorphologyFrameAnalysis] = None

    for frame in iter_selected_frames(
        traj_path, records, report, int(cfg["A_type_id"]), int(cfg["B_type_id"]),
        log=log, allow_variable_natoms=bool(cfg["allow_variable_natoms"]),
    ):
        result = analyze_morphology_frame(
            frame, cfg, sq_q_star=sq.get("q_star"), keep_detail=True, log=log,
        )
        results.append(result)
        frame_rows.append(result.row)
        component_rows.extend(result.component_rows)
        topology_rows.extend(result.topology_rows)
        per_frame_profile_tables.append(profile_frame(result.profile, result.timestep))
        if result.warnings:
            all_warnings[str(result.timestep)] = result.warnings
        last_detail = result

    if not results:
        raise RuntimeError("all selected frames failed to parse")
    if report.failed_frames:
        log.warn(f"{len(report.failed_frames)} selected frame(s) failed to parse")

    summary_stats = summarize_frame_rows(frame_rows, autocorr_keys=AUTOCORR_KEYS)
    last_res = last_detail.resolution if last_detail is not None else {}
    summary: Dict[str, Any] = {
        "sim_id": sim_id, **sim_meta, "traj_file": str(traj_path), "status": "ok",
        "morphology": morphology,
        "minority_phase_last": last_detail.domains.minority_phase,
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
        "sq_q_star": sq.get("q_star"), "sq_L0": sq.get("L0"),
        "normal_profile_frames": len(results),
        "density_profile_enabled": bool(cfg.get("density_profile_enabled", False)),
        "note_frames_correlated": (
            "frames are repeated observations, not independent simulations; use *_n_eff "
            "only as an autocorrelation diagnostic"
        ),
        **summary_stats,
        "warnings": "; ".join(log.warnings),
    }

    pd.DataFrame(frame_rows).to_csv(sim_out / "interface_per_frame.csv", index=False)
    pd.DataFrame(component_rows).to_csv(sim_out / "domain_components_per_frame.csv", index=False)
    pd.DataFrame(topology_rows).to_csv(sim_out / "topology_by_threshold.csv", index=False)
    pd.concat(per_frame_profile_tables, ignore_index=True).to_csv(
        sim_out / "normal_profiles_per_frame.csv", index=False,
    )
    profile_df, profile_arrays = _aggregate_normal_profiles(results)
    profile_df.to_csv(sim_out / "normal_profiles.csv", index=False)
    np.savez_compressed(sim_out / "normal_profiles.npz", **profile_arrays)
    pd.DataFrame([summary]).to_csv(sim_out / "interface_summary.csv", index=False)
    write_json({"summary": summary, "provenance": provenance_block(config_hash(cfg))},
               sim_out / "interface_summary.json")
    write_json({"config": dict(cfg), "config_hash": config_hash(cfg),
                "provenance": provenance_block(config_hash(cfg))},
               sim_out / "interface_parameters.json")
    write_json({"per_frame_resolution_last": last_res, "sq_bridge": sq},
               sim_out / "interface_resolution.json")
    write_json({"summary_warnings": log.warnings, "per_frame": all_warnings},
               sim_out / "interface_warnings.json")
    write_json(report.as_dict(), sim_out / "frame_read_report.json")

    if bool(cfg.get("save_maps", True)) and last_detail is not None:
        np.savez_compressed(
            sim_out / "interface_maps.npz",
            timestep=np.asarray([last_detail.timestep]),
            phi=last_detail.comp.phi,
            rho_A=last_detail.fields.rho_A,
            rho_B=last_detail.fields.rho_B,
            minority_mask=last_detail.domains.minority_mask,
            component_labels=last_detail.domains.component_labels,
        )
        surf = last_detail.surface
        np.savez_compressed(
            sim_out / "interface_surface_last.npz",
            timestep=np.asarray([last_detail.timestep]),
            vertices=surf.vertices, faces=surf.faces,
            centroids=surf.centroids, normals_to_A=surf.normals_to_A,
            face_areas=surf.face_areas, mean_curvature=surf.mean_curvature,
            gaussian_curvature=surf.gaussian_curvature,
        )

    if bool(cfg.get("plots", True)) and last_detail is not None:
        try:
            from .morphology_diagnostics import make_morphology_plots
            make_morphology_plots(
                sim_id, dirs["plots"], last_detail, pd.DataFrame(frame_rows), profile_df,
            )
        except Exception as exc:
            log.error(f"plotting failed: {exc}")
            log(traceback.format_exc())

    log(
        f"Done: {len(frame_rows)} frames; A/V="
        f"{summary.get('surface_area_per_volume_mean', float('nan')):.4f}, "
        f"width={summary.get('normal_width_lam_mean_mean', float('nan')):.4f}"
    )
    return summary
