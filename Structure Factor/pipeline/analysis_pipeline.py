"""Per-simulation orchestration: read → window detect → S(q) → write outputs.

Dual-engine design (cfg["sq_method"]):
  "cic"   — fast Cloud-In-Cell particle-mesh FFT with window deconvolution
            (production screening; per-frame exact-direct low-q validation).
  "exact" — exact full-lattice direct sum from particle coordinates
            (selective publication-quality re-runs; exact by construction
            for any structure orientation).

Both engines share the trajectory read, q-grid resolution, stationarity
diagnostics, window logic, peak detection, output schemas, and plots.
Single-pass design: the trajectory is read once; per-frame FFT diagnostics
and the engine's partial S(q) are computed together so the file is only
opened once. Notation: q = reciprocal-space wavevector magnitude (DPD units).
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from .config import config_hash, parse_filename_metadata, parse_sim_id
from .exact_direct_sq import make_exact_sq_object
from .fft_diagnostic import fft_diagnostic_frame
from .particle_mesh_sq import (
    composition_sq_to_dataframe,
    exact_direct_composition_sq,
    make_sq_object,
    partial_sq_to_dataframe,
    resolve_q_grid,
    validate_mesh_against_exact,
    MeshSqResult as SqResult,
)
from .plotting import make_all_plots, plot_log_sq_screening
from .sanity_checks import format_check_report, run_sanity_checks
from .thermo_parser import find_out_file, parse_out_file, save_job_metadata, termination_status
from .trajectory_reader import final_timestep, iter_frames
from .window_detection import detect_stable_window


def analyze_simulation(
    traj_path: str | Path,
    cfg: Dict[str, Any],
    output_root: Path,
    logger: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Run the full analysis pipeline for one trajectory file.

    Writes per-simulation output files under output_root/sim_id/ and returns
    a summary dict suitable for aggregation into the batch CSV.

    Parameters
    ----------
    traj_path:
        Path to the LAMMPS trajectory file.
    cfg:
        Validated config dict (from :func:`pipeline.config.load_config`).
    output_root:
        Root output directory; per-sim folder created under it.
    logger:
        Optional callable for progress messages (receives a string).
    """
    traj_path = Path(traj_path)
    sim_id = parse_sim_id(traj_path.name, cfg["traj_suffix"])
    sim_meta = parse_filename_metadata(sim_id)
    input_dir = traj_path.parent

    sim_out = output_root / sim_id
    sim_out.mkdir(parents=True, exist_ok=True)
    diag_plot_dir = sim_out / "diagnostic_plots"

    def log(msg: str) -> None:
        if logger:
            logger(f"[{sim_id}] {msg}")

    def fail(error_msg: str) -> Dict[str, Any]:
        """Persist an error summary so failed reruns cannot expose stale success."""
        summary = _error_summary(sim_id, sim_meta, error_msg)
        summary["traj_file"] = str(traj_path)
        summary["config_hash"] = config_hash(cfg)
        pd.DataFrame([summary]).to_csv(sim_out / "analysis_summary.csv", index=False)
        return summary

    # Engine selection: "cic" (particle-mesh screening) or "exact"
    # (full-lattice direct sum).
    sq_method = str(cfg.get("sq_method", "cic")).strip().lower()
    log(f"Starting analysis: {traj_path.name} (sq_method={sq_method})")

    # q-grid settings that may be 'auto' are resolved from the box dimensions
    # of the first frame (resolve_q_grid); until then only q_compute_max and
    # the peak-search restriction flag are known.
    q_compute_max = float(cfg["q_compute_max"])
    q_peak_search_max_restricted = cfg.get("q_peak_search_max") is not None
    resolved: Optional[Dict[str, Any]] = None  # set on first frame

    # ------------------------------------------------------------------ #
    # 0. Parse thermo / out file (optional — fails gracefully)
    # ------------------------------------------------------------------ #
    thermo_df: Optional[pd.DataFrame] = None
    job_meta: Dict[str, Any] = {}
    thermo_warnings: List[str] = []

    out_path = find_out_file(input_dir, sim_id, cfg.get("out_prefix", "out."))
    if out_path is None:
        log("No matching out/log file found — continuing with trajectory-only analysis.")
        thermo_warnings.append("No out/log file found; thermo data unavailable.")
    else:
        try:
            thermo_result = parse_out_file(out_path, sim_id=sim_id)
            thermo_df = thermo_result.thermo_df if not thermo_result.thermo_df.empty else None
            job_meta = thermo_result.job_metadata
            thermo_warnings.extend(thermo_result.warnings)
            if thermo_df is not None:
                thermo_df.to_csv(sim_out / "thermo_timeseries.csv", index=False)
            save_job_metadata(job_meta, sim_out / "job_metadata.json")
            log(f"Thermo: {len(thermo_df) if thermo_df is not None else 0} rows; "
                f"termination={termination_status(job_meta)}")
        except Exception as exc:
            log(f"Thermo parse error: {exc}")
            thermo_warnings.append(f"Thermo parse error: {exc}")

    # ------------------------------------------------------------------ #
    # 1. Single-pass trajectory read: FFT diagnostics + per-frame S(q)
    # ------------------------------------------------------------------ #
    A_id = int(cfg["A_type_id"])
    B_id = int(cfg["B_type_id"])
    analysis_start_step = cfg.get("analysis_start_step")
    if analysis_start_step is not None:
        analysis_start_step = int(analysis_start_step)
        log(
            f"analysis_start_step={analysis_start_step}: frames before this timestep "
            "are skipped at the file level (no parsing, no FFT, no Sq engine)."
        )

    # Pinned S(q) averaging window. Two mutually exclusive forms (validated in
    # config): a fixed trailing window (sq_window_steps) or explicit bounds
    # (sq_window_start_step / sq_window_end_step, intended for exact-mode
    # re-runs of a window found during cic screening). Frames outside the
    # pinned window still get the FFT diagnostic — only the expensive engine
    # call is skipped — so window detection keeps its full-trajectory view.
    sq_window_steps = cfg.get("sq_window_steps")
    sq_cutoff_step: Optional[int] = None       # lower bound (inclusive)
    sq_window_end_bound: Optional[int] = None  # upper bound (inclusive)
    sq_window_pinned = False
    traj_final_step: Optional[int] = None
    sq_window_warnings: List[str] = []
    if sq_window_steps is not None:
        sq_window_steps = int(sq_window_steps)
        try:
            traj_final_step = final_timestep(traj_path)
            sq_cutoff_step = traj_final_step - sq_window_steps
            sq_window_pinned = True
            log(
                f"sq_window_steps={sq_window_steps}: final timestep "
                f"{traj_final_step}, publishable S(q) restricted to frames with "
                f"timestep >= {sq_cutoff_step}. Earlier frames get diagnostics only."
            )
        except Exception as exc:
            msg = (
                f"Could not read final timestep ({exc}); sq_window_steps "
                "ignored — falling back to the detected stationary window."
            )
            log(f"WARNING: {msg}")
            sq_window_warnings.append(msg)
    elif cfg.get("sq_window_start_step") is not None \
            or cfg.get("sq_window_end_step") is not None:
        raw_start = cfg.get("sq_window_start_step")
        raw_end = cfg.get("sq_window_end_step")
        sq_cutoff_step = int(raw_start) if raw_start is not None else None
        sq_window_end_bound = int(raw_end) if raw_end is not None else None
        sq_window_pinned = True
        log(
            "Explicit S(q) window bounds: frames with "
            f"{sq_cutoff_step if sq_cutoff_step is not None else 'trajectory start'}"
            f" <= timestep <= "
            f"{sq_window_end_bound if sq_window_end_bound is not None else 'trajectory end'}"
            ". Frames outside get diagnostics only."
        )

    def _in_sq_window(ts: int) -> bool:
        if sq_cutoff_step is not None and ts < sq_cutoff_step:
            return False
        if sq_window_end_bound is not None and ts > sq_window_end_bound:
            return False
        return True

    per_frame_records: List[Dict[str, Any]] = []
    per_frame_sq: List[Dict[str, np.ndarray]] = []
    exact_validation_records: List[Dict[str, Any]] = []
    sq_failure_records: List[Dict[str, Any]] = []

    first_natoms: Optional[int] = None
    natoms_consistent = True

    # The Sq engine object is created once (on the first frame, after the box
    # is known and 'auto' q settings are resolved) and reused for all frames.
    _sq = None
    _sq_failed = False

    try:
        frame_iter = iter_frames(
            traj_path,
            stride=int(cfg.get("frame_stride", 1)),
            A_type_id=A_id,
            B_type_id=B_id,
            allow_variable_natoms=bool(cfg.get("allow_variable_natoms", False)),
            start_step=analysis_start_step,
        )

        frame_count = 0
        for frame in frame_iter:
            if first_natoms is None:
                first_natoms = frame.n_atoms
            elif frame.n_atoms != first_natoms:
                natoms_consistent = False

            # Resolve 'auto' q-grid settings from the first frame's box and
            # create the reusable Sq engine object. All resolved values are
            # logged here and saved to sq_resolution.json later.
            if resolved is None:
                resolved = resolve_q_grid(frame.L, cfg)
                log(
                    "Resolved q-grid from box "
                    f"L=({resolved['Lx']:.4f}, {resolved['Ly']:.4f}, {resolved['Lz']:.4f}): "
                    f"dq=({resolved['dq_x']:.5f}, {resolved['dq_y']:.5f}, {resolved['dq_z']:.5f}), "
                    f"q_compute=[{resolved['q_compute_min']:.5f}, {resolved['q_compute_max']:.3f}], "
                    f"q_bins={resolved['q_bins']} (width {resolved['q_bin_width']:.5f}), "
                    f"q_peak_search=[{resolved['q_peak_search_min']:.5f}, "
                    f"{resolved['q_peak_search_max']:.3f}], "
                    f"~{resolved['n_kvec_estimate']} reciprocal vectors"
                )
                try:
                    if sq_method == "exact":
                        _sq = make_exact_sq_object(
                            frame.L,
                            q_compute_min=resolved["q_compute_min"],
                            q_compute_max=resolved["q_compute_max"],
                            q_bins=resolved["q_bins"],
                        )
                    else:
                        _sq = make_sq_object(
                            frame.L,
                            q_compute_min=resolved["q_compute_min"],
                            q_compute_max=resolved["q_compute_max"],
                            q_bins=resolved["q_bins"],
                            grid_n=int(cfg.get("sq_grid", 128)),
                        )
                except Exception as exc:
                    _sq_failed = True
                    log(f"S(q) engine ({sq_method}) initialization failed: {exc}")

            frame_count += 1
            if frame_count % 10 == 0:
                log(f"  frame {frame_count} (timestep {frame.timestep})…")

            # FFT diagnostic (cheap per-frame q_star and phi_rms for window detection)
            if cfg.get("run_fft_diagnostic", True):
                fft_d = fft_diagnostic_frame(
                    frame,
                    A_type_id=A_id,
                    B_type_id=B_id,
                    grid_n=int(cfg.get("fft_grid", 64)),
                    phi_mode=str(cfg.get("phi_mode", "zero")),
                    phi_smooth_sigma=float(cfg.get("phi_smooth_sigma", 1.0)),
                    q_peak_search_min=resolved["q_peak_search_min"],
                    q_peak_search_max=resolved["q_peak_search_max"],
                )
            else:
                fft_d = {
                    "q_star_fft": float("nan"),
                    "peak_height_fft": float("nan"),
                    "phi_rms": float("nan"),
                    "grad_rms": float("nan"),
                }

            # Publishable Sq per frame. Frames outside a pinned window retain
            # only diagnostics, keeping the lists index-aligned.
            if not _in_sq_window(frame.timestep):
                per_frame_sq.append({"timestep": frame.timestep})
            elif _sq is not None:
                try:
                    sq_d = _sq.compute(frame, A_type_id=A_id, B_type_id=B_id)
                    per_frame_sq.append({"timestep": frame.timestep, **sq_d})
                except Exception as exc:
                    log(f"S(q) engine ({sq_method}) failed for frame {frame.timestep}: {exc}")
                    sq_failure_records.append({
                        "timestep": int(frame.timestep),
                        "error": str(exc),
                    })
                    per_frame_sq.append({"timestep": frame.timestep})
            else:
                per_frame_sq.append({"timestep": frame.timestep})

            per_frame_records.append({
                "timestep": frame.timestep,
                "n_atoms": frame.n_atoms,
                "Lx": float(frame.L[0]),
                "Ly": float(frame.L[1]),
                "Lz": float(frame.L[2]),
                **fft_d,
            })

    except Exception as exc:
        log(f"Trajectory read FAILED: {exc}")
        return fail(str(exc))

    if not per_frame_records:
        if analysis_start_step is not None:
            return fail(
                f"No frames at or after analysis_start_step={analysis_start_step}. "
                "The trajectory may end before this timestep — check the value or set it to null.",
            )
        return fail("No frames were read from trajectory.")

    frame_df = pd.DataFrame(per_frame_records)
    timesteps = frame_df["timestep"].tolist()
    q_star_fft_list = frame_df["q_star_fft"].tolist()
    peak_height_fft_list = frame_df["peak_height_fft"].tolist()
    phi_rms_list = frame_df["phi_rms"].tolist()

    log(f"Read {len(frame_df)} frames; timestep range [{timesteps[0]}, {timesteps[-1]}]")

    # ------------------------------------------------------------------ #
    # 2. Stable-window detection (uses FFT q_star, peak height, phi_rms + thermo)
    # ------------------------------------------------------------------ #
    try:
        window = detect_stable_window(
            timesteps=timesteps,
            q_star_fft=q_star_fft_list,
            phi_rms=phi_rms_list,
            peak_height_fft=peak_height_fft_list,
            burn_in_fraction=float(cfg["burn_in_fraction"]),
            block_size_steps=int(cfg["block_size_steps"]),
            min_blocks_accepted=int(cfg["min_blocks_accepted"]),
            metric_slope_tolerance=float(cfg["metric_slope_tolerance"]),
            metric_cv_tolerance=float(cfg["metric_cv_tolerance"]),
            fallback_fraction=float(cfg["fallback_fraction"]),
            thermo_df=thermo_df,
            thermo_stationarity_cols=list(cfg.get("thermo_stationarity_cols", [])),
            stationarity_metrics=list(cfg["stationarity_metrics"]),
        )
    except Exception as exc:
        log(f"Window detection failed: {exc}; using all post-burn-in frames.")
        n = len(timesteps)
        burn_cut = max(0, int(np.ceil(n * float(cfg["burn_in_fraction"]))))
        from .window_detection import WindowResult
        window = WindowResult(
            accepted_indices=list(range(burn_cut, n)),
            accepted_timesteps=timesteps[burn_cut:],
            window_start_step=timesteps[burn_cut],
            window_end_step=timesteps[-1],
            stable_found=False,
            fallback_used=True,
            warning_flags=[f"Window detection error: {exc}"],
        )

    log(
        f"Window: stable={window.stable_found}, fallback={window.fallback_used}, "
        f"n_accepted={window.n_accepted}, "
        f"steps [{window.window_start_step}, {window.window_end_step}]"
    )

    # Write window diagnostics
    if not window.block_diagnostics.empty:
        window.block_diagnostics.to_csv(sim_out / "window_diagnostics.csv", index=False)

    # ------------------------------------------------------------------ #
    # 3. Average S(q) over accepted frames
    # ------------------------------------------------------------------ #
    # With a pinned window (trailing sq_window_steps or explicit
    # sq_window_start_step/sq_window_end_step bounds), the averaging set is
    # that window rather than the detector's output. The detected window is
    # still reported everywhere; it becomes a check on the pinned one. If it
    # starts later than the pinned window, the pinned window spans frames the
    # heuristic does not call stationary — flagged, not silently accepted.
    sq_window_within_stable_window: Optional[bool] = None
    if sq_window_pinned:
        accepted_idx = [
            i for i, ts in enumerate(timesteps) if _in_sq_window(ts)
        ]
        accepted_ts = [timesteps[i] for i in accepted_idx]
        if not accepted_idx:
            msg = (
                "No frames inside the pinned Sq window "
                f"[{sq_cutoff_step}, {sq_window_end_bound}] "
                f"(final timestep {traj_final_step}); no publishable S(q) computed."
            )
            log(f"WARNING: {msg}")
            sq_window_warnings.append(msg)
        else:
            log(
                f"Sq window: steps [{accepted_ts[0]}, {accepted_ts[-1]}], "
                f"{len(accepted_idx)} frames"
            )
            if window.fallback_used:
                # No stationary window was detected, so there is nothing to
                # check the pinned window against. Leaving this True would read
                # as reassurance drawn from a fallback range that is only
                # "the last N% of frames".
                sq_window_within_stable_window = None
                msg = (
                    "No stationary window was detected (fallback in use), so the "
                    "pinned Sq window could not be validated against one."
                )
                log(f"NOTE: {msg}")
                sq_window_warnings.append(msg)
            else:
                sq_window_within_stable_window = (
                    window.window_start_step <= accepted_ts[0]
                )
                log(f"  detected stable window starts {window.window_start_step}")
                if not sq_window_within_stable_window:
                    msg = (
                        f"Pinned Sq window starts at step {accepted_ts[0]}, before "
                        f"the detected stationary window ({window.window_start_step}). "
                        "The averaged spectrum includes frames the stationarity "
                        "heuristic does not accept — inspect the diagnostic plots."
                    )
                    log(f"WARNING: {msg}")
                    sq_window_warnings.append(msg)
    else:
        accepted_idx = window.accepted_indices
        accepted_ts = window.accepted_timesteps

    # Box-dimension stability across the accepted window: the q-grid was
    # resolved from the first frame's box, so a drifting box (e.g. NPT)
    # makes the shared grid progressively wrong for later frames.
    box_max_rel_variation = 0.0
    if accepted_idx:
        acc_box = frame_df.loc[accepted_idx, ["Lx", "Ly", "Lz"]]
        for col in ("Lx", "Ly", "Lz"):
            span = float(acc_box[col].max() - acc_box[col].min())
            mean = float(acc_box[col].mean())
            if mean > 0:
                box_max_rel_variation = max(box_max_rel_variation, span / mean)

    accepted_sq = [
        per_frame_sq[i] for i in accepted_idx
        if i < len(per_frame_sq) and "S_comp" in per_frame_sq[i]
    ]
    missing_sq_timesteps = [
        int(timesteps[i]) for i in accepted_idx
        if i >= len(per_frame_sq) or "S_comp" not in per_frame_sq[i]
    ]
    analysis_complete = bool(accepted_idx) and not missing_sq_timesteps
    if missing_sq_timesteps:
        msg = (
            f"ERROR: S(q) failed or is missing for {len(missing_sq_timesteps)} "
            f"of {len(accepted_idx)} requested averaging frames. The spectrum "
            "is incomplete and cannot be numerically certified."
        )
        log(msg)
        sq_window_warnings.append(msg)

    # CIC validation is deliberately sampled across the actual accepted
    # window (beginning/middle/end), rather than from the first trajectory
    # frames encountered before the averaging window was known.
    if sq_method == "cic" and accepted_idx and _sq is not None:
        n_validate = min(
            int(cfg.get("exact_validation_frames", 3)), len(accepted_ts)
        )
        target_positions = np.linspace(
            0, len(accepted_ts) - 1, n_validate, dtype=int
        )
        validation_timesteps = sorted({
            int(accepted_ts[pos]) for pos in target_positions
        })
        target_set = set(validation_timesteps)
        try:
            for frame in iter_frames(
                traj_path, stride=1, A_type_id=A_id, B_type_id=B_id,
                allow_variable_natoms=bool(cfg.get("allow_variable_natoms", False)),
                start_step=validation_timesteps[0],
            ):
                if frame.timestep not in target_set:
                    continue
                mesh_d = _sq.compute(frame, A_type_id=A_id, B_type_id=B_id)
                exact_d = exact_direct_composition_sq(
                    frame,
                    bin_edges=_sq.bin_edges,
                    A_type_id=A_id,
                    B_type_id=B_id,
                    q_max=float(cfg.get("exact_validation_q_max", 1.0)),
                    chunk_size=int(cfg.get("exact_validation_chunk_size", 24)),
                )
                comparison = validate_mesh_against_exact(
                    mesh_d,
                    exact_d,
                    q_min=resolved["q_peak_search_min"],
                    q_max=min(
                        resolved["q_peak_search_max"],
                        float(cfg.get("exact_validation_q_max", 1.0)),
                    ),
                    peak_rel_tolerance=float(
                        cfg.get("exact_validation_peak_rel_tol", 0.10)
                    ),
                )
                exact_validation_records.append({
                    "timestep": int(frame.timestep),
                    "n_exact_vectors": exact_d["n_vectors"],
                    **comparison,
                })
                log(
                    f"  exact validation at {frame.timestep}: "
                    f"passed={comparison['passed']}, "
                    f"peak_detected={comparison['exact_peak_detected']}, "
                    f"spectrum_rmse={comparison['spectrum_rel_rmse']:.2%}"
                )
                target_set.remove(frame.timestep)
                if not target_set:
                    break
        except Exception as exc:
            log(f"Exact-direct validation failed: {exc}")
            sq_window_warnings.append(
                f"ERROR: Exact-direct validation failed: {exc}"
            )

    sq_result: Optional[SqResult] = None
    s_tot_check_err: float = 0.0
    kvec_counts: Optional[np.ndarray] = None
    kvec_count_method: Optional[str] = None
    n_kvec_qstar_bin: Optional[int] = None
    n_kvec_total: Optional[int] = None
    sq_block_info: Dict[str, Any] = {
        "n_blocks": 0,
        "spectral_stationary": None,
        "max_recent_log_distance": float("nan"),
        "low_q_intensity_relative_slope": float("nan"),
        "peak_intensity_relative_slope": float("nan"),
        "whole_window_edge_log_distance": float("nan"),
        "sem": {},
    }

    if accepted_sq:
        q_ref = accepted_sq[0]["q"]
        valid = [d for d in accepted_sq if "q" in d and len(d["q"]) == len(q_ref)]

        S_AA_acc = np.mean([d["S_AA"] for d in valid], axis=0)
        S_BB_acc = np.mean([d["S_BB"] for d in valid], axis=0)
        S_tot_acc = np.mean([d["S_tot"] for d in valid], axis=0)
        S_AB_acc = np.mean([d["S_AB"] for d in valid], axis=0)
        S_comp_acc = np.mean([d["S_comp"] for d in valid], axis=0)
        S_cc_acc = np.mean([d["S_cc"] for d in valid], axis=0)
        S_cc_normalized_acc = np.mean(
            [d["S_cc_normalized"] for d in valid], axis=0
        )
        x_A = float(np.mean([d["x_A"] for d in valid]))
        x_B = float(np.mean([d["x_B"] for d in valid]))

        # Validate S_tot ≈ S_AA + S_BB + 2*S_AB
        from .particle_mesh_sq import _check_s_tot_consistency, _find_dominant_low_q_peak
        s_tot_check_err = _check_s_tot_consistency(S_AA_acc, S_BB_acc, S_AB_acc, S_tot_acc)
        if s_tot_check_err > 1e-4:
            log(f"WARNING: S_tot consistency check failed: max rel err = {s_tot_check_err:.2e}")

        # Peak detection
        peak = _find_dominant_low_q_peak(
            q_ref, S_cc_normalized_acc,
            resolved["q_peak_search_min"], resolved["q_peak_search_max"],
            resolved["q_bins"],
            min_height_over_median=float(
                cfg.get("peak_min_height_over_median", 1.5)
            ),
            min_relative_prominence=float(
                cfg.get("peak_min_relative_prominence", 0.10)
            ),
            min_occupied_bins=int(cfg.get("peak_min_occupied_bins", 5)),
        )
        q_star = peak["q_star"]
        peak_h = peak["peak_height"]
        L0 = 2.0 * np.pi / q_star if np.isfinite(q_star) and q_star > 0 else float("nan")

        sq_block_info = _write_sq_block_analysis(
            valid, cfg, sim_out,
            q_min=resolved["q_peak_search_min"],
            q_max=resolved["q_peak_search_max"],
            peak_q=(q_star if bool(peak.get("peak_detected", False)) else None),
        )

        # Reciprocal-vector counts per q bin. Both engines expose .counts:
        # the cic engine counts the FFT lattice modes (with conjugate
        # degeneracy), the exact engine enumerates the reciprocal lattice
        # exactly. The bin holding q_star tells us how many k-vectors the
        # reported peak value is averaged over — a small count means a
        # statistically noisy peak.
        if _sq is not None:
            kvec_counts = _sq.counts.copy()
            kvec_count_method = {
                "cic": "fft_reciprocal_lattice",
                "exact": "lattice_enumeration_exact",
            }[sq_method]
        if kvec_counts is not None:
            n_kvec_total = int(kvec_counts.sum())
            if np.isfinite(q_star):
                bin_idx = int(np.argmin(np.abs(q_ref - q_star)))
                n_kvec_qstar_bin = int(kvec_counts[bin_idx])

        sq_result = SqResult(
            q=q_ref,
            S_AA=S_AA_acc,
            S_BB=S_BB_acc,
            S_AB=S_AB_acc,
            S_tot=S_tot_acc,
            S_comp=S_comp_acc,
            S_cc=S_cc_acc,
            S_cc_normalized=S_cc_normalized_acc,
            x_A=x_A,
            x_B=x_B,
            q_star=q_star,
            L0=L0,
            peak_height=peak_h,
            n_frames_averaged=len(valid),
            peak_detected=bool(peak.get("peak_detected", False)),
            peak_reason=str(peak.get("reason", "")),
            peak_background_median=float(
                peak.get("background_median", np.nan)
            ),
            peak_height_over_median=float(
                peak.get("height_over_median", np.nan)
            ),
            peak_relative_prominence=float(
                peak.get("relative_prominence", np.nan)
            ),
            q_star_at_lower_boundary=peak.get("at_lower_boundary", False),
            q_star_at_upper_boundary=peak.get("at_upper_boundary", False),
            n_peak_candidates=peak.get("n_candidates", 0),
            peak_detection_method=peak.get("method", "unknown"),
            s_tot_check_max_rel_err=s_tot_check_err,
            normalization_notes={
                "sq_method_used": sq_method,
                "engine": {
                    "cic": "CIC particle-mesh FFT with assignment-window deconvolution",
                    "exact": (
                        "exact direct sum over every reciprocal-lattice vector "
                        "from particle coordinates (orientation-exact)"
                    ),
                }[sq_method],
                "convention": "All S(q) normalized by N_total",
                "S_comp_formula": "S_comp = 2*(S_AA + S_BB) - S_tot = (1/N)|F_A - F_B|^2",
                "S_cc_formula": "S_cc = x_B^2*S_AA + x_A^2*S_BB - 2*x_A*x_B*S_AB",
                "S_cc_normalized_formula": "S_cc_normalized = S_cc/(x_A*x_B)",
                "x_A": x_A,
                "x_B": x_B,
                "peak_search_spectrum": "S_cc_normalized",
                "peak_detected": bool(peak.get("peak_detected", False)),
                "peak_reason": peak.get("reason"),
                "peak_min_height_over_median": float(
                    cfg.get("peak_min_height_over_median", 1.5)
                ),
                "peak_min_relative_prominence": float(
                    cfg.get("peak_min_relative_prominence", 0.10)
                ),
                "sq_grid": (
                    int(cfg.get("sq_grid", 128)) if sq_method == "cic" else None
                ),
                "q_compute_max": q_compute_max,
                "q_peak_search_max": resolved["q_peak_search_max"],
                "q_peak_search_max_was_null": not q_peak_search_max_restricted,
                "peak_detection_method": peak.get("method"),
                "n_peak_candidates": peak.get("n_candidates"),
            },
        )
        if sq_result.peak_detected:
            log(
                f"S(q) [{sq_method}]: significant peak q_star={q_star:.4f}, "
                f"L0={L0:.3f}, n_frames={len(valid)}, "
                f"height/median={sq_result.peak_height_over_median:.2f}"
            )
        else:
            log(
                f"S(q) [{sq_method}]: no significant finite-q peak detected; "
                f"q_star and L0 left undefined. n_frames={len(valid)}"
            )

        # Write outputs
        partial_sq_to_dataframe(sq_result).to_csv(
            sim_out / "sq_partial.csv", index=False
        )
        comp_df = composition_sq_to_dataframe(sq_result)
        sem = sq_block_info.get("sem", {})
        for key, column in (
            ("S_comp", "S_comp_contrast_sem_blocks"),
            ("S_cc", "S_cc_sem_blocks"),
            ("S_cc_normalized", "S_cc_normalized_sem_blocks"),
        ):
            if key in sem:
                comp_df[column] = sem[key]
        comp_df.to_csv(sim_out / "sq_composition.csv", index=False)

        # Save normalization notes alongside the spectrum files
        with (sim_out / "sq_normalization.json").open("w") as fh:
            json.dump(sq_result.normalization_notes, fh, indent=2, default=str)

        # Optional per-frame spectra
        if cfg.get("save_per_frame_spectra", False):
            pf_rows = []
            for d in valid:
                ts = d.get("timestep", -1)
                for bi, qv in enumerate(d["q"]):
                    pf_rows.append({
                        "timestep": ts, "q": qv,
                        "S_AA": d["S_AA"][bi], "S_BB": d["S_BB"][bi],
                        "S_AB": d["S_AB"][bi], "S_tot": d["S_tot"][bi],
                        "S_comp": d["S_comp"][bi],
                        "S_cc": d["S_cc"][bi],
                        "S_cc_normalized": d["S_cc_normalized"][bi],
                    })
            pd.DataFrame(pf_rows).to_csv(sim_out / "sq_per_frame.csv", index=False)

    else:
        log(f"No S(q) results ({sq_method}) available for accepted frames.")

    # ------------------------------------------------------------------ #
    # 3b. q-resolution metadata (full provenance of the resolved q-grid)
    # ------------------------------------------------------------------ #
    if resolved is not None:
        resolution_doc = {
            "sq_method_used": sq_method,
            "Lx": resolved["Lx"], "Ly": resolved["Ly"], "Lz": resolved["Lz"],
            "dq_x": resolved["dq_x"], "dq_y": resolved["dq_y"], "dq_z": resolved["dq_z"],
            "q_compute_min": resolved["q_compute_min"],
            "q_compute_max": resolved["q_compute_max"],
            "q_bin_width": resolved["q_bin_width"],
            "q_bins": resolved["q_bins"],
            "q_peak_search_min": resolved["q_peak_search_min"],
            "q_peak_search_max": resolved["q_peak_search_max"],
            "q_peak_search_max_was_restricted": q_peak_search_max_restricted,
            "n_kvec_estimate": resolved["n_kvec_estimate"],
            "n_kvec_sampled_exact": n_kvec_total,
            "n_kvec_in_q_star_bin": n_kvec_qstar_bin,
            "kvec_counts_available": kvec_counts is not None,
            "kvec_count_method": kvec_count_method,
            "box_dims_max_rel_variation_accepted_window": box_max_rel_variation,
            "resolution_notes": resolved["resolution_notes"],
        }
        with (sim_out / "sq_resolution.json").open("w") as fh:
            json.dump(resolution_doc, fh, indent=2, default=str)

    # Mean FFT q_star over accepted frames (for cross-check and summary).
    # This tracks whichever frames the engine actually averaged, so the FFT
    # crosscheck compares the same window. The stationary-window mean is kept
    # separately since the two sets differ when an Sq window is pinned.
    if cfg.get("run_fft_diagnostic", True):
        accepted_q_fft = float(
            frame_df.loc[accepted_idx, "q_star_fft"].dropna().mean()
        ) if accepted_idx else float("nan")
        stable_q_fft = float(
            frame_df.loc[window.accepted_indices, "q_star_fft"].dropna().mean()
        ) if window.accepted_indices else float("nan")
    else:
        accepted_q_fft = float("nan")
        stable_q_fft = float("nan")
    log(f"Diagnostic q_star (Sq-averaged frames, mean): {accepted_q_fft:.4f}")

    # ------------------------------------------------------------------ #
    # 4. Sanity checks
    # ------------------------------------------------------------------ #
    dq_min_resolved = (
        min(resolved["dq_x"], resolved["dq_y"], resolved["dq_z"]) if resolved else float("nan")
    )
    check_input = {
        "has_A_beads": True,
        "has_B_beads": True,
        # Frames actually averaged — the "too few frames" check is about the
        # statistical support of the averaged spectrum, which is the pinned
        # Sq window when one is set, not the detector's window.
        "n_accepted_frames": len(accepted_idx),
        "q_star_sq": sq_result.q_star if sq_result else float("nan"),
        "peak_detected": sq_result.peak_detected if sq_result else False,
        "q_star_at_lower_boundary": sq_result.q_star_at_lower_boundary if sq_result else False,
        "q_star_at_upper_boundary": sq_result.q_star_at_upper_boundary if sq_result else False,
        "q_star_fft_mean": accepted_q_fft,
        "n_frames_consistent": natoms_consistent,
        "s_tot_check_err": s_tot_check_err,
        "fallback_used": window.fallback_used,
        # q-resolution checks
        "n_kvec_in_q_star_bin": n_kvec_qstar_bin,
        "q_bin_width": resolved["q_bin_width"] if resolved else float("nan"),
        "dq_min": dq_min_resolved,
        "box_dims_max_rel_variation": box_max_rel_variation,
        "q_peak_search_min_resolved": resolved["q_peak_search_min"] if resolved else None,
        "q_peak_search_max_resolved": resolved["q_peak_search_max"] if resolved else None,
    }
    check_issues = run_sanity_checks(check_input, cfg)
    spectral_status = sq_block_info.get("spectral_stationary")
    stationarity_supported = bool(
        window.stable_found and spectral_status is True
    )
    if spectral_status is False:
        check_issues.append(
            "WARN: The blockwise S(q) plateau check failed. The candidate "
            "window should not be described as stationary; inspect "
            "sq_stationarity.json and sq_block_metrics.csv."
        )
    elif spectral_status is None:
        check_issues.append(
            "WARN: There were not enough valid S(q) blocks to establish a "
            "spectral plateau."
        )
    if not analysis_complete:
        check_issues.append(
            "ERROR: The S(q) average is incomplete because one or more requested "
            "frames failed. Numerical certification is disabled."
        )

    engine_type = {
        "cic": "cic_particle_mesh",
        "exact": "exact_direct_full_lattice",
    }[sq_method]

    if sq_method == "exact":
        # This engine IS the exact direct sum over the full reciprocal
        # lattice — the reference calculation the other engines are validated
        # against. Certification requires only that a spectrum was produced.
        exact_validation_passed = bool(sq_result is not None and analysis_complete)
        with (sim_out / "exact_direct_validation.json").open("w") as fh:
            json.dump(
                {
                    "engine_type": engine_type,
                    "exact_validation_passed": exact_validation_passed,
                    "all_passed": exact_validation_passed,
                    "required_frames": 0,
                    "frames_checked": 0,
                    "records": [],
                    "note": (
                        "sq_method=exact evaluates S(q) exactly at every "
                        "reciprocal-lattice vector in the compute range from "
                        "particle coordinates; it is the reference the other "
                        "engines validate against, so per-frame comparison is "
                        "not applicable. exact_validation_passed is false only "
                        "when no spectrum was produced."
                    ),
                },
                fh,
                indent=2,
                default=str,
            )
        if not exact_validation_passed:
            check_issues.append(
                "ERROR: exact engine produced no spectrum for the requested window."
            )
    else:
        # cic is a mesh approximation and must agree with the full-lattice
        # exact sum at low q on the checked frames.
        required_validation_frames = int(cfg.get("exact_validation_frames", 3))
        exact_validation_passed = (
            analysis_complete
            and len(exact_validation_records) == required_validation_frames
            and all(bool(r.get("passed")) for r in exact_validation_records)
        )
        if int(cfg.get("exact_validation_frames", 3)) > 0:
            validation_doc = {
                "engine_type": engine_type,
                "required_frames": int(cfg.get("exact_validation_frames", 3)),
                "frames_checked": len(exact_validation_records),
                "all_passed": exact_validation_passed,
                "records": exact_validation_records,
            }
            with (sim_out / "exact_direct_validation.json").open("w") as fh:
                json.dump(validation_doc, fh, indent=2, default=str)
            if len(exact_validation_records) < required_validation_frames:
                check_issues.append(
                    "ERROR: Fewer exact-direct validation frames were completed "
                    "than requested."
                )
            elif not exact_validation_passed:
                check_issues.append(
                    "ERROR: Particle-mesh S(q) failed exact-direct low-q validation."
                )
    if check_issues:
        log(format_check_report(check_issues, sim_id))

    # ------------------------------------------------------------------ #
    # 5. Write accepted_windows.json
    # ------------------------------------------------------------------ #
    accepted_windows_doc = {
        "sim_id": sim_id,
        "sq_method_used": sq_method,
        "stable_found": window.stable_found,
        "fallback_used": window.fallback_used,
        "stationarity_metrics_gating": list(cfg["stationarity_metrics"]),
        "window_start_step": window.window_start_step,
        "window_end_step": window.window_end_step,
        "n_accepted_frames": window.n_accepted,
        # Frames the S(q) engine actually averaged. Identical to the stationary
        # window unless sq_window_steps or explicit bounds pinned a window.
        "sq_window_steps": sq_window_steps,
        "sq_window_cutoff_step": sq_cutoff_step,
        "sq_window_end_bound": sq_window_end_bound,
        "sq_window_start_step": accepted_ts[0] if accepted_ts else None,
        "sq_window_end_step": accepted_ts[-1] if accepted_ts else None,
        "n_sq_window_frames": len(accepted_ts),
        "sq_window_within_stable_window": sq_window_within_stable_window,
        "traj_final_step": traj_final_step,
        "accepted_timesteps": window.accepted_timesteps,
        "sq_averaged_timesteps": accepted_ts,
        "warning_flags": (
            window.warning_flags + thermo_warnings + sq_window_warnings
            + [i for i in check_issues if i.startswith("WARN") or i.startswith("ERROR")]
        ),
        "numerical_validation_passed": exact_validation_passed,
        "publishable_sq_numerically_certified": exact_validation_passed,
        "stationarity_supported_by_all_checks": stationarity_supported,
        "analysis_complete": analysis_complete,
        "sq_failed_timesteps": missing_sq_timesteps,
        "termination_status": termination_status(job_meta) if job_meta else "unknown",
        "q_peak_search_max_used": resolved["q_peak_search_max"] if resolved else None,
        "q_peak_search_max_was_restricted": q_peak_search_max_restricted,
    }
    with (sim_out / "accepted_windows.json").open("w") as fh:
        json.dump(accepted_windows_doc, fh, indent=2, default=str)

    # Frame-level metrics
    frame_df.to_csv(sim_out / "frame_metrics.csv", index=False)

    # ------------------------------------------------------------------ #
    # 6. Diagnostic plots
    # ------------------------------------------------------------------ #
    if cfg.get("plots", True):
        try:
            make_all_plots(
                sim_id=sim_id,
                plot_dir=diag_plot_dir,
                timesteps=timesteps,
                q_star_fft=q_star_fft_list,
                phi_rms=phi_rms_list,
                window_result=window,
                sq_result=sq_result,
                cfg=cfg,
                thermo_df=thermo_df,
                q_peak_search_max=(
                    resolved["q_peak_search_max"]
                    if (resolved and q_peak_search_max_restricted) else None
                ),
                resolved=resolved,
                peak_height_fft=peak_height_fft_list,
                sq_window=(
                    (accepted_ts[0], accepted_ts[-1])
                    if (sq_window_pinned and accepted_ts) else None
                ),
                sq_method=sq_method,
                spectral_stationary=sq_block_info.get("spectral_stationary"),
            )
            if sq_result is not None:
                plot_log_sq_screening(
                    sq_result.q,
                    sq_result.S_comp,
                    sq_result.S_cc_normalized,
                    output_root / "plots" / f"{sim_id}_sq_log.png",
                    q_star=sq_result.q_star,
                    L0=sq_result.L0,
                    peak_detected=sq_result.peak_detected,
                    sim_id=sim_id,
                    sq_method=sq_method,
                    q_plot_min=resolved["q_compute_min"] if resolved else None,
                    q_plot_max=resolved["q_compute_max"] if resolved else None,
                )
        except Exception as exc:
            log(f"Plotting failed: {exc}")

    # ------------------------------------------------------------------ #
    # 7. Build and write analysis_summary.csv row
    # ------------------------------------------------------------------ #
    q_star_val = sq_result.q_star if sq_result else float("nan")
    L0_val = sq_result.L0 if sq_result else float("nan")
    summary = {
        "sim_id": sim_id,
        **sim_meta,
        "traj_file": str(traj_path),
        "sq_method_used": sq_method,
        "analysis_start_step": analysis_start_step,
        "n_total_frames": len(frame_df),
        "n_accepted_frames": window.n_accepted,
        "window_start_step": window.window_start_step,
        "window_end_step": window.window_end_step,
        "stable_window_found": window.stable_found,
        "fallback_used": window.fallback_used,
        "stationarity_metrics_gating": ",".join(cfg["stationarity_metrics"]),
        # Pinned window actually used for the S(q) average (null config →
        # same frames as the stationary window above).
        "sq_window_steps": sq_window_steps,
        "sq_window_start_step": accepted_ts[0] if accepted_ts else None,
        "sq_window_end_step": accepted_ts[-1] if accepted_ts else None,
        "sq_window_within_stable_window": sq_window_within_stable_window,
        "traj_final_step": traj_final_step,
        "q_star_sq": q_star_val,
        "L0_sq": L0_val,
        "peak_detected": sq_result.peak_detected if sq_result else False,
        "peak_reason": sq_result.peak_reason if sq_result else "",
        "peak_height_over_median": (
            sq_result.peak_height_over_median if sq_result else float("nan")
        ),
        "peak_relative_prominence": (
            sq_result.peak_relative_prominence if sq_result else float("nan")
        ),
        "peak_height_sq": sq_result.peak_height if sq_result else float("nan"),
        "n_frames_sq_averaged": sq_result.n_frames_averaged if sq_result else 0,
        "exact_validation_frames_completed": len(exact_validation_records),
        "exact_validation_passed": exact_validation_passed,
        "publishable_sq_numerically_certified": exact_validation_passed,
        "analysis_complete": analysis_complete,
        "n_sq_frames_requested": len(accepted_idx),
        "n_sq_frames_failed": len(missing_sq_timesteps),
        "q_star_at_lower_boundary": sq_result.q_star_at_lower_boundary if sq_result else False,
        "q_star_at_upper_boundary": sq_result.q_star_at_upper_boundary if sq_result else False,
        "n_peak_candidates": sq_result.n_peak_candidates if sq_result else 0,
        "peak_detection_method": sq_result.peak_detection_method if sq_result else "",
        "q_star_fft_mean": accepted_q_fft,
        "q_star_fft_mean_stable_window": stable_q_fft,
        "s_tot_check_max_rel_err": s_tot_check_err,
        # Resolved q-grid provenance (full detail in sq_resolution.json)
        "resolved_q_compute_min": resolved["q_compute_min"] if resolved else float("nan"),
        "resolved_q_bins": resolved["q_bins"] if resolved else 0,
        "resolved_q_bin_width": resolved["q_bin_width"] if resolved else float("nan"),
        "resolved_q_peak_search_min": resolved["q_peak_search_min"] if resolved else float("nan"),
        "resolved_q_peak_search_max": resolved["q_peak_search_max"] if resolved else float("nan"),
        "n_kvec_estimate": resolved["n_kvec_estimate"] if resolved else 0,
        "n_kvec_in_q_star_bin": n_kvec_qstar_bin if n_kvec_qstar_bin is not None else -1,
        "box_dims_max_rel_variation": box_max_rel_variation,
        "sq_spectral_stationary": sq_block_info.get("spectral_stationary"),
        "stationarity_supported_by_all_checks": stationarity_supported,
        "sq_stationarity_n_blocks": sq_block_info.get("n_blocks", 0),
        "sq_max_recent_log_spectrum_distance": sq_block_info.get(
            "max_recent_log_distance", float("nan")
        ),
        "sq_low_q_intensity_relative_slope": sq_block_info.get(
            "low_q_intensity_relative_slope", float("nan")
        ),
        "sq_peak_intensity_relative_slope": sq_block_info.get(
            "peak_intensity_relative_slope", float("nan")
        ),
        "sq_whole_window_edge_log_spectrum_distance": sq_block_info.get(
            "whole_window_edge_log_distance", float("nan")
        ),
        "x_A": sq_result.x_A if sq_result else float("nan"),
        "x_B": sq_result.x_B if sq_result else float("nan"),
        "termination_status": termination_status(job_meta) if job_meta else "no_out_file",
        "sanity_check_issues": "; ".join(check_issues) if check_issues else "",
        "config_hash": config_hash(cfg),
        "natoms_consistent": natoms_consistent,
        "warnings": "; ".join(window.warning_flags + thermo_warnings + sq_window_warnings),
    }

    pd.DataFrame([summary]).to_csv(sim_out / "analysis_summary.csv", index=False)
    q_str = f"{q_star_val:.4f}" if np.isfinite(q_star_val) else "NaN"
    L0_str = f"{L0_val:.3f}" if np.isfinite(L0_val) else "NaN"
    log(f"Done. q_star={q_str}, L0={L0_str}")
    return summary


def _write_sq_block_analysis(
    frames: List[Dict[str, Any]],
    cfg: Dict[str, Any],
    sim_out: Path,
    *,
    q_min: float,
    q_max: float,
    peak_q: Optional[float] = None,
) -> Dict[str, Any]:
    """Write block spectra, block uncertainties, and a full-spectrum plateau check.

    This is deliberately a stationarity diagnostic, not an equilibrium proof.
    It works when no peak exists because it compares the complete normalized
    concentration spectrum rather than tracking q_star.
    """
    if not frames:
        return {
            "n_blocks": 0, "spectral_stationary": None,
            "max_recent_log_distance": float("nan"),
            "low_q_intensity_relative_slope": float("nan"),
            "peak_intensity_relative_slope": float("nan"),
            "whole_window_edge_log_distance": float("nan"), "sem": {},
        }

    q = np.asarray(frames[0]["q"], dtype=float)
    t0 = min(int(d["timestep"]) for d in frames)
    block_steps = max(1, int(cfg.get("block_size_steps", 25000)))
    groups: Dict[int, List[Dict[str, Any]]] = {}
    for d in frames:
        block_id = (int(d["timestep"]) - t0) // block_steps
        groups.setdefault(block_id, []).append(d)

    spectrum_keys = (
        "S_AA", "S_BB", "S_AB", "S_tot",
        "S_comp", "S_cc", "S_cc_normalized",
    )
    block_spectra: List[Dict[str, Any]] = []
    long_rows: List[Dict[str, Any]] = []
    previous_normalized: Optional[np.ndarray] = None
    peak_bin_index: Optional[int] = None
    if peak_q is not None and np.isfinite(peak_q):
        finite_q = np.flatnonzero(np.isfinite(q))
        if finite_q.size:
            peak_bin_index = int(
                finite_q[np.argmin(np.abs(q[finite_q] - float(peak_q)))]
            )

    for block_id in sorted(groups):
        members = groups[block_id]
        means = {
            key: np.mean([np.asarray(d[key], dtype=float) for d in members], axis=0)
            for key in spectrum_keys
        }
        ts = [int(d["timestep"]) for d in members]
        current = means["S_cc_normalized"]
        finite = (
            np.isfinite(q) & np.isfinite(current)
            & (q >= q_min) & (q <= q_max)
        )
        low_q_intensity = (
                float(np.trapezoid(current[finite], q[finite]))
            if int(np.count_nonzero(finite)) >= 2 else float("nan")
        )
        log_distance = float("nan")
        if previous_normalized is not None:
            common = finite & np.isfinite(previous_normalized)
            if np.any(common):
                a = np.log1p(np.maximum(current[common], 0.0))
                b = np.log1p(np.maximum(previous_normalized[common], 0.0))
                log_distance = float(np.sqrt(np.mean((a - b) ** 2)))
        previous_normalized = current
        peak_bin_intensity = (
            float(current[peak_bin_index])
            if peak_bin_index is not None and np.isfinite(current[peak_bin_index])
            else float("nan")
        )
        block_spectra.append({
            "block_id": int(block_id),
            "block_start_step": min(ts),
            "block_end_step": max(ts),
            "block_mid_step": 0.5 * (min(ts) + max(ts)),
            "n_frames": len(members),
            "low_q_integrated_S_cc_normalized": low_q_intensity,
            "peak_bin_S_cc_normalized": peak_bin_intensity,
            "log_spectrum_distance_from_previous": log_distance,
            **means,
        })
        for i, qv in enumerate(q):
            row = {
                "block_id": int(block_id),
                "block_start_step": min(ts),
                "block_end_step": max(ts),
                "n_frames": len(members),
                "q": qv,
            }
            for key in spectrum_keys:
                row[key] = means[key][i]
            long_rows.append(row)

    pd.DataFrame(long_rows).to_csv(
        sim_out / "sq_block_spectra.csv", index=False
    )
    metric_rows = [{
        k: v for k, v in block.items() if k not in spectrum_keys
    } for block in block_spectra]
    pd.DataFrame(metric_rows).to_csv(
        sim_out / "sq_block_metrics.csv", index=False
    )

    n_blocks = len(block_spectra)
    min_blocks = int(cfg.get("sq_stationarity_min_blocks", 3))
    tolerance = float(cfg.get("sq_spectral_distance_tolerance", 0.10))
    recent = block_spectra[-min_blocks:] if n_blocks >= min_blocks else []
    recent_distances = [
        float(b["log_spectrum_distance_from_previous"]) for b in recent[1:]
    ]
    max_distance = (
        float(np.nanmax(recent_distances))
        if recent_distances and np.isfinite(recent_distances).any()
        else float("nan")
    )
    rel_slope = float("nan")
    if len(recent) >= 2:
        x = np.asarray([b["block_mid_step"] for b in recent], dtype=float)
        y = np.asarray([
            b["low_q_integrated_S_cc_normalized"] for b in recent
        ], dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)
        if int(np.count_nonzero(finite)) >= 2:
            slope, _ = np.polyfit(x[finite], y[finite], 1)
            y_mean = float(np.mean(y[finite]))
            x_span = float(np.max(x[finite]) - np.min(x[finite]))
            if abs(y_mean) > 1e-30:
                rel_slope = float(abs(slope) * x_span / abs(y_mean))

    # A slowly strengthening morphology can pass every short block while
    # drifting across the full candidate window. Track the intensity at the
    # peak detected from the window-averaged spectrum across *all* blocks.
    peak_rel_slope = float("nan")
    if peak_bin_index is not None and n_blocks >= min_blocks:
        x = np.asarray([b["block_mid_step"] for b in block_spectra], dtype=float)
        y = np.asarray([
            b["peak_bin_S_cc_normalized"] for b in block_spectra
        ], dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)
        if int(np.count_nonzero(finite)) >= min_blocks:
            slope, _ = np.polyfit(x[finite], y[finite], 1)
            y_mean = float(np.mean(y[finite]))
            x_span = float(np.max(x[finite]) - np.min(x[finite]))
            if abs(y_mean) > 1e-30:
                peak_rel_slope = float(abs(slope) * x_span / abs(y_mean))

    # Also compare spectra averaged over the two ends of the whole candidate
    # window. This remains meaningful when no significant peak exists.
    edge_log_distance = float("nan")
    if n_blocks >= min_blocks:
        edge_n = max(1, min(min_blocks, n_blocks // 2))
        first = np.mean(
            [b["S_cc_normalized"] for b in block_spectra[:edge_n]], axis=0
        )
        last = np.mean(
            [b["S_cc_normalized"] for b in block_spectra[-edge_n:]], axis=0
        )
        common = (
            np.isfinite(q) & np.isfinite(first) & np.isfinite(last)
            & (q >= q_min) & (q <= q_max)
        )
        if np.any(common):
            a = np.log1p(np.maximum(first[common], 0.0))
            b = np.log1p(np.maximum(last[common], 0.0))
            edge_log_distance = float(np.sqrt(np.mean((a - b) ** 2)))

    slope_tolerance = float(cfg.get("metric_slope_tolerance", 0.02))
    peak_slope_tolerance = float(
        cfg.get("sq_peak_intensity_slope_tolerance", slope_tolerance)
    )

    spectral_stationary: Optional[bool]
    if (
        n_blocks < min_blocks
        or not np.isfinite(max_distance)
        or not np.isfinite(rel_slope)
        or not np.isfinite(edge_log_distance)
        or (peak_bin_index is not None and not np.isfinite(peak_rel_slope))
    ):
        spectral_stationary = None
    else:
        spectral_stationary = bool(
            max_distance <= tolerance
            and edge_log_distance <= tolerance
            and rel_slope <= slope_tolerance
            and (
                peak_bin_index is None
                or peak_rel_slope <= peak_slope_tolerance
            )
        )

    sem: Dict[str, np.ndarray] = {}
    if n_blocks >= 2:
        for key in spectrum_keys:
            stack = np.asarray([b[key] for b in block_spectra], dtype=float)
            sem[key] = np.nanstd(stack, axis=0, ddof=1) / np.sqrt(n_blocks)

    doc = {
        "method": "blockwise_full_spectrum_stationarity_check",
        "spectrum_compared": "S_cc_normalized",
        "n_blocks": n_blocks,
        "minimum_blocks_required": min_blocks,
        "block_size_steps": block_steps,
        "spectral_stationary": spectral_stationary,
        "max_recent_log_spectrum_distance": max_distance,
        "log_spectrum_distance_tolerance": tolerance,
        "whole_window_edge_log_spectrum_distance": edge_log_distance,
        "low_q_intensity_relative_slope": rel_slope,
        "peak_q_used_for_drift_check": (
            float(q[peak_bin_index]) if peak_bin_index is not None else None
        ),
        "peak_intensity_relative_slope": peak_rel_slope,
        "peak_intensity_slope_tolerance": peak_slope_tolerance,
        "relative_slope_tolerance": slope_tolerance,
        "q_range": [float(q_min), float(q_max)],
        "note": (
            "Stationary means the measured spectrum stopped changing within "
            "these thresholds over the sampled interval. It does not distinguish "
            "thermodynamic equilibrium from a long-lived metastable state."
        ),
    }
    with (sim_out / "sq_stationarity.json").open("w") as fh:
        json.dump(doc, fh, indent=2, default=str)
    return {
        "n_blocks": n_blocks,
        "spectral_stationary": spectral_stationary,
        "max_recent_log_distance": max_distance,
        "low_q_intensity_relative_slope": rel_slope,
        "peak_intensity_relative_slope": peak_rel_slope,
        "whole_window_edge_log_distance": edge_log_distance,
        "sem": sem,
    }


def _error_summary(sim_id: str, sim_meta: Dict[str, Any], error_msg: str) -> Dict[str, Any]:
    return {
        "sim_id": sim_id,
        **sim_meta,
        "traj_file": "",
        "sq_method_used": "",
        "n_total_frames": 0,
        "n_accepted_frames": 0,
        "window_start_step": None,
        "window_end_step": None,
        "stable_window_found": False,
        "fallback_used": False,
        "q_star_sq": float("nan"),
        "L0_sq": float("nan"),
        "peak_height_sq": float("nan"),
        "n_frames_sq_averaged": 0,
        "analysis_complete": False,
        "n_sq_frames_requested": 0,
        "n_sq_frames_failed": 0,
        "exact_validation_passed": False,
        "publishable_sq_numerically_certified": False,
        "stationarity_supported_by_all_checks": False,
        "peak_detected": False,
        "peak_reason": "analysis_error",
        "q_star_at_lower_boundary": False,
        "q_star_at_upper_boundary": False,
        "n_peak_candidates": 0,
        "peak_detection_method": "",
        "q_star_fft_mean": float("nan"),
        "s_tot_check_max_rel_err": float("nan"),
        "termination_status": "analysis_error",
        "sanity_check_issues": f"ERROR: {error_msg}",
        "config_hash": "",
        "natoms_consistent": False,
        "warnings": error_msg,
    }
