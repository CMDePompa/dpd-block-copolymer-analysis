"""Automated stationary-window heuristic for DPD trajectory analysis.

This module identifies a late-time region of the trajectory where key
structural and thermodynamic metrics have stabilized. It is NOT a proof of
equilibration — it is a transparent heuristic that documents its own criteria
and flags all cases where no stable window was found.

Algorithm
---------
1. Sort frames by timestep.
2. Discard the first burn_in_fraction of frames.
3. Divide remaining frames into blocks of block_size_steps.
4. For each block compute stationarity metrics:
   Structural (trajectory-derived):
     - q_star_fft:   CV and relative linear slope
     - peak_height_fft: CV (FFT composition peak height proxy)
     - phi_rms:       CV (order-parameter magnitude)
   Thermodynamic (from out file, when available):
     - CV of each column in thermo_stationarity_cols (PotEng, TotEng, Press, Temp)
5. A block is stationary if ALL checked metrics pass their thresholds.
6. Find the longest contiguous run of stationary blocks anchored at the END
   of the post-burn trajectory (we want late-time stationarity).
7. If the run contains >= min_blocks_accepted blocks, declare it the stable window.
8. Otherwise fall back to the last fallback_fraction of frames and issue a warning.
   Fallback windows are never presented as equilibrated.

IMPORTANT: Even a "stable" window is only evidence that these specific metrics
are no longer trending. It is not thermodynamic proof of equilibration.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class WindowResult:
    """Outcome of the stationary-window search."""

    accepted_indices: List[int]          # indices into the original frame list
    accepted_timesteps: List[int]
    window_start_step: int
    window_end_step: int
    stable_found: bool
    fallback_used: bool
    warning_flags: List[str] = field(default_factory=list)
    block_diagnostics: pd.DataFrame = field(default_factory=pd.DataFrame)
    method: str = "stationary_window_heuristic"
    # Metrics that gated the decision; others in block_diagnostics are descriptive.
    stationarity_metrics: List[str] = field(default_factory=list)

    @property
    def n_accepted(self) -> int:
        return len(self.accepted_indices)


def detect_stable_window(
    timesteps: List[int],
    q_star_fft: List[float],
    phi_rms: List[float],
    *,
    peak_height_fft: Optional[List[float]] = None,
    burn_in_fraction: float = 0.2,
    block_size_steps: int = 25000,
    min_blocks_accepted: int = 4,
    metric_slope_tolerance: float = 0.02,
    metric_cv_tolerance: float = 0.05,
    fallback_fraction: float = 0.30,
    thermo_df: Optional[pd.DataFrame] = None,
    thermo_stationarity_cols: Optional[List[str]] = None,
    thermo_cv_tolerance: Optional[float] = None,
    stationarity_metrics: Optional[List[str]] = None,
) -> WindowResult:
    """Identify the latest contiguous stationary window in the trajectory.

    Parameters
    ----------
    timesteps:
        Simulation step number for each frame, in trajectory order.
    q_star_fft:
        FFT-derived composition-peak position q* per frame.
    phi_rms:
        RMS of the composition field phi per frame.
    peak_height_fft:
        FFT composition peak height per frame. Used as an additional CV metric
        when provided. If None, only q_star_fft and phi_rms are checked.
    burn_in_fraction:
        Discard this fraction of total frames from the start before analysis.
    block_size_steps:
        Width of each analysis block in simulation steps.
    min_blocks_accepted:
        Minimum number of contiguous stationary blocks to declare success.
    metric_slope_tolerance:
        A block passes the slope check if the relative linear slope of q_star
        across it is below this value: |slope × t_range / mean(q_star)| < tol.
    metric_cv_tolerance:
        A block passes the CV check if std/|mean| < tol for q_star and phi_rms.
    fallback_fraction:
        If no stable window, use the last this fraction of post-burn-in frames.
    thermo_df:
        Optional DataFrame with a 'Step' column and thermo quantities.
    thermo_stationarity_cols:
        Which thermo columns to check for stationarity.
        Defaults to PotEng, TotEng, Press, Temp.
    thermo_cv_tolerance:
        CV tolerance for thermo quantities. Defaults to 2 × metric_cv_tolerance.
    stationarity_metrics:
        Which metrics *gate* the stationary/non-stationary decision. Any subset
        of {"q_star_fft", "peak_height_fft", "phi_rms", "thermo"}; defaults to
        all four. Metrics left out are still computed and reported in
        block_diagnostics — they just do not veto a block.

        This matters because q_star_fft and peak_height_fft are derived from a
        discrete reciprocal grid of spacing 2π/L. When q* is comparable to a
        few grid spacings, a one-shell hop is a double-digit relative change,
        so their CV cannot fall below a typical tolerance no matter how
        equilibrated the system is, and they veto every block. Dropping them
        from the decision (keeping phi_rms, which is continuous) avoids reading
        grid quantization as physical drift.
    """
    if thermo_stationarity_cols is None:
        thermo_stationarity_cols = ["PotEng", "TotEng", "Press", "Temp"]
    if thermo_cv_tolerance is None:
        thermo_cv_tolerance = 2.0 * metric_cv_tolerance
    _ALL_METRICS = ("q_star_fft", "peak_height_fft", "phi_rms", "thermo")
    if stationarity_metrics is None:
        stationarity_metrics = list(_ALL_METRICS)
    unknown = set(stationarity_metrics) - set(_ALL_METRICS)
    if unknown:
        raise ValueError(
            f"Unknown stationarity_metrics {sorted(unknown)}; "
            f"valid values are {list(_ALL_METRICS)}"
        )
    if not stationarity_metrics:
        raise ValueError("stationarity_metrics must name at least one metric")
    gate = set(stationarity_metrics)

    n_frames = len(timesteps)
    if n_frames == 0:
        raise ValueError("No frames provided to window detection.")

    ts_arr = np.array(timesteps, dtype=np.int64)
    qs_arr = np.array(q_star_fft, dtype=np.float64)
    phi_arr = np.array(phi_rms, dtype=np.float64)
    ph_arr = np.array(peak_height_fft, dtype=np.float64) if peak_height_fft is not None else None

    # 1. Burn-in discard
    burn_cut = max(0, int(np.ceil(n_frames * burn_in_fraction)))
    post_burn_idx = np.arange(burn_cut, n_frames)
    if len(post_burn_idx) == 0:
        post_burn_idx = np.arange(n_frames)

    t_post = ts_arr[post_burn_idx]
    q_post = qs_arr[post_burn_idx]
    phi_post = phi_arr[post_burn_idx]
    ph_post = ph_arr[post_burn_idx] if ph_arr is not None else None

    t_start = int(t_post[0])
    t_end = int(t_post[-1])

    # 2. Divide into blocks
    block_edges = list(range(t_start, t_end, block_size_steps)) + [t_end + 1]
    blocks: List[Dict[str, Any]] = []

    for b_lo, b_hi in zip(block_edges[:-1], block_edges[1:]):
        mask = (t_post >= b_lo) & (t_post < b_hi)
        if mask.sum() < 2:
            continue
        bidx = post_burn_idx[mask]
        q_blk = q_post[mask]
        phi_blk = phi_post[mask]
        t_blk = t_post[mask]

        # --- Structural metrics ---
        q_mean = float(np.nanmean(q_blk))
        q_cv = (
            float(np.nanstd(q_blk, ddof=1) / max(abs(q_mean), 1e-12))
            if q_mean != 0 else np.nan
        )
        q_slope_rel = _relative_slope(t_blk.astype(float), q_blk)

        phi_mean = float(np.nanmean(phi_blk))
        phi_cv = float(np.nanstd(phi_blk, ddof=1) / max(abs(phi_mean), 1e-12))

        ph_cv = np.nan
        if ph_post is not None:
            ph_blk = ph_post[mask]
            ph_mean = float(np.nanmean(ph_blk))
            if ph_mean != 0 and np.isfinite(ph_mean):
                ph_cv = float(np.nanstd(ph_blk, ddof=1) / max(abs(ph_mean), 1e-12))

        # --- Thermo metrics ---
        thermo_stats: Dict[str, float] = {}
        thermo_missing_cols: List[str] = []
        if thermo_df is not None and "Step" in thermo_df.columns:
            t_blk_df = thermo_df[(thermo_df["Step"] >= b_lo) & (thermo_df["Step"] < b_hi)]
            for col in thermo_stationarity_cols:
                if col in t_blk_df.columns:
                    vals = t_blk_df[col].dropna().to_numpy(dtype=float)
                    if len(vals) >= 2:
                        col_mean = float(np.mean(vals))
                        if abs(col_mean) > 1e-12:
                            cv = float(np.std(vals, ddof=1) / abs(col_mean))
                            thermo_stats[f"cv_{col}"] = cv
                        else:
                            thermo_missing_cols.append(col)
                    else:
                        thermo_missing_cols.append(col)
                else:
                    thermo_missing_cols.append(col)
        else:
            thermo_missing_cols = list(thermo_stationarity_cols)

        # --- Stationarity decision ---
        q_ok = (
            np.isfinite(q_cv) and q_cv < metric_cv_tolerance
            and np.isfinite(q_slope_rel) and q_slope_rel < metric_slope_tolerance
        )
        phi_ok = np.isfinite(phi_cv) and phi_cv < metric_cv_tolerance
        # peak height CV uses same structural tolerance (it's a structural metric)
        ph_ok = np.isfinite(ph_cv) and ph_cv < metric_cv_tolerance
        # Missing thermo is unavailable evidence, not a passing measurement.
        requested_thermo_keys = {
            f"cv_{col}" for col in thermo_stationarity_cols
        }
        thermo_ok = (
            bool(requested_thermo_keys)
            and not thermo_missing_cols
            and requested_thermo_keys.issubset(thermo_stats)
            and all(
                np.isfinite(thermo_stats[key])
                and thermo_stats[key] < thermo_cv_tolerance
                for key in requested_thermo_keys
            )
        )
        # Only gating metrics can veto a block; the rest stay descriptive.
        checks = {
            "q_star_fft": q_ok,
            "peak_height_fft": ph_ok,
            "phi_rms": phi_ok,
            "thermo": thermo_ok,
        }
        is_stationary = all(v for k, v in checks.items() if k in gate)

        block_info = {
            "block_start_step": b_lo,
            "block_end_step": b_hi,
            "n_frames": int(mask.sum()),
            "frame_indices": bidx.tolist(),
            "q_star_fft_mean": q_mean,
            "q_star_fft_cv": q_cv,
            "q_star_fft_slope_rel": q_slope_rel,
            "phi_rms_cv": phi_cv,
            "peak_height_fft_cv": ph_cv,
            "q_star_ok": q_ok,
            "phi_ok": phi_ok,
            "peak_height_ok": ph_ok,
            "thermo_ok": thermo_ok,
            "thermo_missing_cols": ",".join(sorted(set(thermo_missing_cols))),
            "stationary": is_stationary,
            **thermo_stats,
        }
        blocks.append(block_info)

    if not blocks:
        return _fallback(
            timesteps, post_burn_idx, ts_arr, fallback_fraction,
            warning="No usable blocks after burn-in; using fallback window.",
            stationarity_metrics=sorted(gate),
        )

    # 3. Require a contiguous stationary run that reaches the latest usable
    # block. An older stationary episode cannot certify a shorter late run.
    stationary_flags = [b["stationary"] for b in blocks]
    trailing_run_len = 0
    for flag in reversed(stationary_flags):
        if not flag:
            break
        trailing_run_len += 1

    block_diag_df = pd.DataFrame([
        {k: v for k, v in b.items() if k != "frame_indices"}
        for b in blocks
    ])

    if trailing_run_len >= min_blocks_accepted:
        run_end_idx = len(blocks) - 1
        run_start_idx = run_end_idx - trailing_run_len + 1

        accepted_blocks = blocks[run_start_idx : run_end_idx + 1]
        accepted_idx: List[int] = []
        for b in accepted_blocks:
            accepted_idx.extend(b["frame_indices"])
        accepted_idx = sorted(set(accepted_idx))
        accepted_ts = [int(ts_arr[i]) for i in accepted_idx]

        return WindowResult(
            accepted_indices=accepted_idx,
            accepted_timesteps=accepted_ts,
            window_start_step=int(blocks[run_start_idx]["block_start_step"]),
            window_end_step=int(blocks[run_end_idx]["block_end_step"]),
            stable_found=True,
            fallback_used=False,
            block_diagnostics=block_diag_df,
            stationarity_metrics=sorted(gate),
        )
    else:
        # Summarize which *gating* criteria failed, so the message explains the
        # actual verdict rather than metrics that were only descriptive.
        key_by_metric = {
            "q_star_fft": "q_star_ok", "peak_height_fft": "peak_height_ok",
            "phi_rms": "phi_ok", "thermo": "thermo_ok",
        }
        parts = [
            f"{m} failed in {sum(1 for b in blocks if not b[key_by_metric[m]])}/{len(blocks)} blocks"
            for m in sorted(gate)
        ]
        failure_summary = "; ".join(parts)
        return _fallback(
            timesteps, post_burn_idx, ts_arr, fallback_fraction,
            warning=(
                f"FALLBACK WINDOW: stationary-window heuristic did not converge "
                f"(trailing run = {trailing_run_len} blocks, "
                f"need {min_blocks_accepted}). "
                f"{failure_summary}. "
                f"Using last {fallback_fraction:.0%} of post-burn-in frames. "
                "This window is NOT automatically equilibrated -- inspect the diagnostic plots."
            ),
            block_diagnostics=block_diag_df,
            stationarity_metrics=sorted(gate),
        )


def _fallback(
    timesteps: List[int],
    post_burn_idx: np.ndarray,
    ts_arr: np.ndarray,
    fallback_fraction: float,
    warning: str,
    block_diagnostics: Optional[pd.DataFrame] = None,
    stationarity_metrics: Optional[List[str]] = None,
) -> WindowResult:
    n_post = len(post_burn_idx)
    n_keep = max(1, int(np.ceil(n_post * fallback_fraction)))
    accepted_idx = post_burn_idx[-n_keep:].tolist()
    accepted_ts = [int(ts_arr[i]) for i in accepted_idx]
    warnings.warn(warning, RuntimeWarning, stacklevel=3)
    return WindowResult(
        accepted_indices=accepted_idx,
        accepted_timesteps=accepted_ts,
        window_start_step=int(ts_arr[accepted_idx[0]]),
        window_end_step=int(ts_arr[accepted_idx[-1]]),
        stable_found=False,
        fallback_used=True,
        warning_flags=[warning],
        block_diagnostics=block_diagnostics if block_diagnostics is not None else pd.DataFrame(),
        stationarity_metrics=list(stationarity_metrics or []),
    )


def _relative_slope(t: np.ndarray, y: np.ndarray) -> float:
    """Relative linear slope: |slope × t_range / mean(y)|. NaN if < 2 finite points."""
    finite = np.isfinite(y)
    if finite.sum() < 2:
        return float("nan")
    t_f, y_f = t[finite], y[finite]
    m, _ = np.polyfit(t_f, y_f, 1)
    mean_y = np.nanmean(y_f)
    if abs(mean_y) < 1e-12:
        return float("nan")
    t_range = float(t_f[-1] - t_f[0])
    return float(abs(m) * t_range / abs(mean_y))
