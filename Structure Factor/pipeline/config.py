"""Configuration loading, validation, and hashing for Sq pipeline version 4."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict

import yaml

# ---------------------------------------------------------------------------
# Defaults — every key the pipeline reads must have a value here.
# ---------------------------------------------------------------------------
_DEFAULTS: Dict[str, Any] = {
    # I/O
    "input_dir": ".",
    "output_dir": "./sq_analysis_output",
    "traj_suffix": ".prod.lammpstrj",
    "out_prefix": "out.",
    # Frame sampling
    "frame_stride": 1,
    "analysis_start_step": None,
    "burn_in_fraction": 0.2,
    # Stable-window heuristic
    "block_size_steps": 25000,
    "min_blocks_accepted": 4,
    "metric_slope_tolerance": 0.02,
    "metric_cv_tolerance": 0.05,
    "fallback_fraction": 0.30,
    "thermo_stationarity_cols": ["PotEng", "TotEng", "Press", "Temp"],
    # Which metrics gate the stationary/non-stationary decision. Others are
    # still computed and reported, but cannot veto a block. See the template
    # for why the FFT-derived metrics are often unusable as gates.
    "stationarity_metrics": ["phi_rms", "thermo"],
    # Structure-factor engine: "cic" (fast particle-mesh FFT, production
    # screening) or "exact" (full-lattice direct sum from particle
    # coordinates, orientation-exact — use for publication re-runs).
    "sq_method": "cic",
    # Fixed trailing S(q) averaging window (null → use the detected stationary
    # window). When set to N, the publishable S(q) is averaged over exactly the
    # frames with timestep >= (final timestep - N). FFT diagnostics still run
    # on every frame, so window detection remains a full-trajectory check.
    "sq_window_steps": None,
    # Explicit S(q) averaging window bounds (both null → disabled). When set,
    # the publishable S(q) is averaged over frames with
    #   sq_window_start_step <= timestep <= sq_window_end_step
    # (a null end means "to the end of the trajectory"; a null start means
    # "from the first analyzed frame"). Intended for exact-mode re-runs of the
    # exact stationary window found during CIC screening. Mutually exclusive
    # with sq_window_steps — setting both is a config error.
    "sq_window_start_step": None,
    "sq_window_end_step": None,
    "sq_grid": 128,
    "exact_validation_frames": 3,
    "exact_validation_q_max": 1.0,
    "exact_validation_peak_rel_tol": 0.10,
    "exact_validation_chunk_size": 24,
    # S(q) — computation range and binning
    # 'auto' values are resolved per simulation from the box dimensions
    # (dq = 2*pi/L per axis); see particle_mesh_sq.resolve_q_grid for the exact rules.
    # All resolved values are logged and saved to sq_resolution.json.
    # PERFORMANCE: StaticStructureFactorDirect uses ALL reciprocal-lattice vectors
    # with |q| < q_compute_max. Count scales as (q_compute_max * L / 2π)^3.
    # For L≈32, q_compute_max=4 gives ~35k vectors (~4 s/call); 8 gives ~275k vectors (~30 s/call).
    "q_compute_min": "auto",     # auto → smallest fundamental spacing min(2π/Lx, 2π/Ly, 2π/Lz)
    "q_compute_max": 4.00,       # upper limit for computing and saving the spectrum only
    "q_bin_width": "auto",       # auto → min fundamental spacing / 4; explicit value computes q_bins
    "q_bins": "auto",            # auto → ceil((q_compute_max - q_compute_min) / q_bin_width)
    # Peak-search range (separate from compute range)
    # q_peak_search_min: lower bound for peak detection (excludes near-zero q)
    # q_peak_search_max: upper bound, or null to use the full compute range
    "q_peak_search_min": "auto", # auto → the fundamental spacing (excludes q=0 artifacts)
    "q_peak_search_max": None,   # null → use q_compute_max
    # A q_star is reported only for a significant interior local maximum.
    # These transparent thresholds prevent disordered noise from acquiring a
    # forced domain spacing.
    "peak_min_height_over_median": 1.5,
    "peak_min_relative_prominence": 0.10,
    "peak_min_occupied_bins": 5,
    # Post-selection convergence check on block-averaged full spectra.
    "sq_stationarity_min_blocks": 3,
    "sq_spectral_distance_tolerance": 0.10,
    "sq_peak_intensity_slope_tolerance": 0.02,
    # Particle types
    "A_type_id": 1,
    "B_type_id": 2,
    # Cross-validation / FFT diagnostic
    "run_fft_diagnostic": True,
    "fft_grid": 64,
    "fft_crosscheck_tol": 0.15,  # relative tolerance for q_star comparison (FFT vs publishable Sq)
    "phi_mode": "zero",          # empty-voxel treatment: "zero", "nan_mean", "smooth"
    "phi_smooth_sigma": 1.0,
    # Output options
    "save_per_frame_spectra": False,
    "plots": True,
    # Consistency checks
    "allow_variable_natoms": False,
}


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load a YAML config file and fill in missing keys with defaults."""
    path = Path(path)
    with path.open("r") as fh:
        user = yaml.safe_load(fh) or {}
    cfg = dict(_DEFAULTS)
    cfg.update(user)
    if isinstance(cfg.get("sq_method"), str):
        cfg["sq_method"] = cfg["sq_method"].strip().lower()
    _validate(cfg)
    return cfg


def save_config(cfg: Dict[str, Any], path: str | Path) -> None:
    """Write config dict as YAML (human-readable, with config hash appended)."""
    path = Path(path)
    out = dict(cfg)
    out["_config_hash"] = config_hash(cfg)
    with path.open("w") as fh:
        yaml.dump(out, fh, default_flow_style=False, sort_keys=True)


def config_hash(cfg: Dict[str, Any]) -> str:
    """SHA-256 hash of the config (excluding the hash field itself) for provenance."""
    clean = {k: v for k, v in cfg.items() if k != "_config_hash"}
    blob = json.dumps(clean, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def make_template(path: str | Path) -> None:
    """Write a template YAML config with comments to *path*."""
    path = Path(path)
    template = """\
# =============================================================================
# Sq_pipeline_v4 — configuration template (dual-engine)
# =============================================================================
# Copy this file, edit the values, and pass it to run_pipeline.py --config.
# Keys not present in your file fall back to built-in defaults.
# Notation: q = reciprocal-space wavevector magnitude (DPD units).

# ---------------------------------------------------------------------------
# Structure-factor engine
# ---------------------------------------------------------------------------
# sq_method selects how the publishable S(q) is computed:
#   cic   — Cloud-In-Cell particle-mesh FFT with Fourier window deconvolution,
#           degeneracy weighting, and radial shell binning. Fast; use for
#           high-throughput campaign screening. Includes per-frame exact-direct
#           low-q validation (exact_validation_* keys below).
#   exact — exact direct sum over EVERY reciprocal-lattice vector, computed
#           from particle coordinates (no gridding, no shell subsampling).
#           Correct for any structure orientation; the recommended engine for
#           selective publication-quality re-runs of ordered trajectories.
# (A freud engine was evaluated and removed: freud's
# StaticStructureFactorDirect folds k-space into the all-positive octant and
# misweights or entirely misses Bragg intensity from oriented structures.)
# Both engines share the same q-grid resolution, window logic, peak detection,
# outputs, and plots. The engine used is recorded as sq_method_used in
# sq_normalization.json, sq_resolution.json, and analysis_summary.csv.
sq_method: cic

# ---------------------------------------------------------------------------
# I/O paths
# ---------------------------------------------------------------------------
input_dir: /path/to/simulations    # directory holding .lammpstrj and out.* files
output_dir: ./sq_analysis_output   # where to write per-sim folders and batch CSVs
traj_suffix: .prod.lammpstrj       # trailing suffix that identifies trajectory files
out_prefix: out.                   # prefix of LAMMPS out/log files

# ---------------------------------------------------------------------------
# Frame sampling
# ---------------------------------------------------------------------------
frame_stride: 1        # use every Nth frame in the trajectory (1 = all frames)
# analysis_start_step: skip every frame with timestep < this value, or null to
# read the whole file. The skip happens at the file level — skipped frames are
# not parsed and get no FFT/Sq computation — so use this when visual
# inspection already shows where order appears. Frames before the cut do not
# count toward frame_stride. NOTE: burn_in_fraction still discards the first
# fraction of the REMAINING frames; if the start step IS your burn-in, set
# burn_in_fraction: 0.0.
analysis_start_step: null
burn_in_fraction: 0.2  # discard this fraction of (post-start-step) frames before analysis

# ---------------------------------------------------------------------------
# Stable-window heuristic
# (Automated stationary-window detector — NOT a proof of equilibration.)
# Metrics: FFT q_star CV + slope, FFT peak height CV, phi_rms CV, thermo CVs.
# ---------------------------------------------------------------------------
block_size_steps: 25000        # width of each analysis block in simulation steps
min_blocks_accepted: 4         # minimum number of contiguous stationary blocks required
metric_slope_tolerance: 0.02   # max relative linear slope of q_star across a block
metric_cv_tolerance: 0.05      # max CV (std/mean) of q_star and phi_rms within a block
fallback_fraction: 0.30        # if no stable window found, use last this fraction of traj
# Thermo columns to include in block stationarity checks (must exist in the out file)
thermo_stationarity_cols:
  - PotEng
  - TotEng
  - Press
  - Temp

# stationarity_metrics: which metrics may VETO a block. Everything listed in
# the heuristic is always computed and written to window_diagnostics.csv; this
# key only controls which ones decide stationary vs not.
#
# Why you may want to drop the FFT-derived ones: q_star_fft and
# peak_height_fft come from a discrete reciprocal grid of spacing 2*pi/L. If q*
# is only a few grid spacings from the origin, moving by a single shell is a
# double-digit relative change, so their CV has a floor set by geometry rather
# than by physics and no equilibrated system can pass a ~0.05 tolerance. Left
# in, they veto every block and force a fallback window. phi_rms is continuous
# and does not have this problem.
#
# Neighbouring-bin movement of q* is NOT evidence of non-stationarity, and
# replicate-to-replicate differences in q* are not either — in weakly ordered
# systems that scatter can be physical.
#
# Valid entries: q_star_fft, peak_height_fft, phi_rms, thermo
stationarity_metrics:
  - phi_rms
  - thermo

# ---------------------------------------------------------------------------
# S(q) averaging window
# ---------------------------------------------------------------------------
# Three mutually exclusive ways to choose which frames the publishable S(q)
# is averaged over (frames outside the window still get the cheap FFT
# diagnostics, so the stationary-window heuristic always sees the whole
# trajectory and becomes a *check* on a pinned window):
#
# 1. All three keys null → average over the DETECTED stationary window
#    (the automatic cic-screening behaviour).
# 2. sq_window_steps: N → fixed trailing window, frames with
#    timestep >= (final timestep - N). The final timestep is read from the
#    end of each file.
# 3. sq_window_start_step / sq_window_end_step → EXPLICIT bounds,
#    frames with start <= timestep <= end (null end = to the end of the
#    trajectory; null start = from the first analyzed frame). Intended for
#    sq_method: exact re-runs of the exact stationary window discovered
#    during cic screening — copy sq_window_start_step/sq_window_end_step
#    from the cic run's accepted_windows.json.
#
# Setting sq_window_steps together with explicit bounds is a config error.
# If a pinned window starts before the detected stationary window, a warning
# is raised (sq_window_within_stable_window: false in accepted_windows.json).
sq_window_steps: null
sq_window_start_step: null
sq_window_end_step: null

# ---------------------------------------------------------------------------
# S(q) — computation range and binning
# ---------------------------------------------------------------------------
# 'auto' values are resolved per simulation from the box read out of the
# trajectory. The fundamental reciprocal spacings are dq_x = 2*pi/Lx,
# dq_y = 2*pi/Ly, dq_z = 2*pi/Lz; reciprocal vectors exist only at integer
# multiples of these. Resolved values are logged and written per simulation
# to sq_resolution.json — nothing is chosen silently.
#
#   q_compute_min: auto  -> min(dq_x, dq_y, dq_z), the smallest |q| the box supports
#   q_bin_width:   auto  -> min fundamental spacing / 4 (resolves individual q-shells)
#   q_bins:        auto  -> ceil((q_compute_max - q_compute_min) / q_bin_width)
#   Explicit q_bins overrides auto bin width; explicit q_bin_width computes
#   q_bins. Setting BOTH explicitly is a config error.
#
# PERFORMANCE NOTE: both engines cover every reciprocal-lattice vector with
# |q| < q_compute_max, and that count scales as (q_compute_max*L/2*pi)^3.
# For L~32: q_compute_max=4 gives ~35k vectors; 8 gives ~275k (~8x slower).
# q_compute_max is the upper limit for spectrum computation and output ONLY.
# It is not a scientific cutoff for the morphology peak.
q_compute_min: auto     # minimum q, or 'auto' for the box fundamental spacing
q_compute_max: 4.00     # maximum q for computation (always explicit; ≤4 for speed)
q_bin_width: auto       # q bin width, or 'auto' for fundamental spacing / 4
q_bins: auto            # number of q bins, or 'auto' to derive from q_bin_width

# ---------------------------------------------------------------------------
# Peak-search range (distinct from computation range)
# ---------------------------------------------------------------------------
# q_peak_search_min/max define where to look for the composition-contrast peak.
# q_peak_search_min: auto puts it at the fundamental reciprocal spacing so
# q = 0 and near-zero artifacts are excluded.
# q_peak_search_max: null means use the full q_compute_max range.
# When q_peak_search_max is set, the restriction is recorded in metadata and plots.
q_peak_search_min: auto   # or an explicit q value
q_peak_search_max: null   # upper limit for peak detection, or null for full range

# A q_star is NOT forced. A candidate must be an interior local maximum and
# pass both tests below. If no candidate passes, peak_detected=false and q_star
# and L0 are left blank/NaN. Inspect the log-S(q) plot before changing these.
peak_min_height_over_median: 1.5
peak_min_relative_prominence: 0.10
peak_min_occupied_bins: 5

# The accepted S(q) frames are also divided into timestep blocks. The pipeline
# compares the complete normalized concentration spectra between consecutive
# blocks using an RMS log-distance. This is a post-analysis stationarity check,
# not a proof of thermodynamic equilibrium.
sq_stationarity_min_blocks: 3
sq_spectral_distance_tolerance: 0.10
# Maximum fitted change in S_cc_normalized(q*) across the entire candidate
# window. This catches slow coarsening that looks quiet inside short blocks.
sq_peak_intensity_slope_tolerance: 0.02

# ---------------------------------------------------------------------------
# cic engine settings (ignored when sq_method: exact)
# ---------------------------------------------------------------------------
# sq_grid: CIC mesh resolution per dimension. q_compute_max must stay well
# below the mesh Nyquist limit pi*sq_grid/L or initialization fails; 128 is
# the production value. Spot-check 64/128/256 give the same low-q peak.
sq_grid: 128
# Per-frame validation of the mesh spectrum against the exact direct sum at
# low q. A peak-bin mismatch or an intensity error above the tolerance is an
# analysis ERROR, not a warning. The exact engine skips this (it IS the
# reference calculation).
exact_validation_frames: 3        # frames checked per simulation
exact_validation_q_max: 1.0       # upper q of the validation range
exact_validation_peak_rel_tol: 0.10   # max relative peak-intensity error
exact_validation_chunk_size: 24   # k-vectors per chunk (memory/speed knob)

# ---------------------------------------------------------------------------
# Bead type IDs (must match LAMMPS atom types in the trajectory)
# ---------------------------------------------------------------------------
A_type_id: 1
B_type_id: 2

# ---------------------------------------------------------------------------
# FFT composition-field diagnostic (cross-validation with publishable q_star)
# ---------------------------------------------------------------------------
run_fft_diagnostic: true   # run voxel-FFT diagnostic on all frames
fft_grid: 64               # grid resolution per dimension
fft_crosscheck_tol: 0.15   # flag if |q_star_sq - q_star_fft| / q_star_sq > this
phi_mode: zero             # empty-voxel treatment: "zero" | "nan_mean" | "smooth"
phi_smooth_sigma: 1.0      # periodic Gaussian width in voxels for phi_mode=smooth

# ---------------------------------------------------------------------------
# Output options
# ---------------------------------------------------------------------------
save_per_frame_spectra: false  # save S(q) CSV for every accepted frame
plots: true                    # generate diagnostic plots

# ---------------------------------------------------------------------------
# Consistency checks
# ---------------------------------------------------------------------------
allow_variable_natoms: false   # set true only if particle count is expected to change
"""
    path.write_text(template, encoding="utf-8")


# ---------------------------------------------------------------------------
# Internal validation
# ---------------------------------------------------------------------------
def _is_auto(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() == "auto"


def _validate(cfg: Dict[str, Any]) -> None:
    errs = []

    # q_compute_max must always be an explicit positive number (it sets the
    # computational budget; there is no sensible box-derived default for it).
    try:
        q_max = float(cfg["q_compute_max"])
        if q_max <= 0:
            errs.append("q_compute_max must be positive")
    except (TypeError, ValueError):
        errs.append("q_compute_max must be a number (not 'auto')")
        q_max = None

    # Keys that accept a positive number or the string 'auto'
    for key in ("q_compute_min", "q_peak_search_min", "q_bin_width", "q_bins"):
        v = cfg.get(key, "auto")
        if _is_auto(v):
            continue
        try:
            if float(v) <= 0:
                errs.append(f"{key} must be positive or 'auto'")
        except (TypeError, ValueError):
            errs.append(f"{key} must be a number or 'auto' (got {v!r})")

    # q_bins and q_bin_width are alternative ways to set the same thing:
    # both explicit at once is ambiguous, and we never choose silently.
    if not _is_auto(cfg.get("q_bins", "auto")) and not _is_auto(cfg.get("q_bin_width", "auto")):
        errs.append(
            "q_bins and q_bin_width cannot both be explicit — set one and "
            "leave the other 'auto' (q_bins wins over auto bin width; "
            "q_bin_width is used to compute q_bins)"
        )

    # Numeric cross-checks only where both sides are explicit; auto values
    # are resolved per simulation against the box and re-checked there.
    if q_max is not None:
        if not _is_auto(cfg.get("q_compute_min", "auto")) and float(cfg["q_compute_min"]) >= q_max:
            errs.append("q_compute_min must be less than q_compute_max")
        q_peak_hi = cfg.get("q_peak_search_max")
        if q_peak_hi is not None:
            if float(q_peak_hi) > q_max:
                errs.append("q_peak_search_max cannot exceed q_compute_max")
            if not _is_auto(cfg.get("q_peak_search_min", "auto")) \
                    and float(cfg["q_peak_search_min"]) >= float(q_peak_hi):
                errs.append("q_peak_search_min must be less than q_peak_search_max")

    start_step = cfg.get("analysis_start_step")
    if start_step is not None:
        try:
            if int(start_step) < 0:
                errs.append("analysis_start_step must be a non-negative integer or null")
        except (TypeError, ValueError):
            errs.append(f"analysis_start_step must be an integer or null (got {start_step!r})")

    valid_metrics = {"q_star_fft", "peak_height_fft", "phi_rms", "thermo"}
    sm = cfg.get("stationarity_metrics")
    if not isinstance(sm, (list, tuple)) or not sm:
        errs.append("stationarity_metrics must be a non-empty list")
    else:
        bad = sorted(set(sm) - valid_metrics)
        if bad:
            errs.append(
                f"stationarity_metrics contains unknown entries {bad}; "
                f"valid values are {sorted(valid_metrics)}"
            )
        if not bool(cfg.get("run_fft_diagnostic", True)):
            unavailable = sorted(set(sm) & {
                "q_star_fft", "peak_height_fft", "phi_rms"
            })
            if unavailable:
                errs.append(
                    "run_fft_diagnostic=false is incompatible with structural "
                    f"stationarity metrics {unavailable}; use thermo only or "
                    "enable the FFT diagnostic"
                )

    if cfg.get("sq_method") not in ("cic", "exact"):
        errs.append(
            "sq_method must be \"cic\" or \"exact\" "
            f"(got {cfg.get('sq_method')!r}). The freud engine was removed: "
            "freud's StaticStructureFactorDirect folds k-space into the "
            "all-positive octant and misweights or misses Bragg intensity "
            "from oriented structures — use \"exact\" for publication re-runs."
        )

    window_value = cfg.get("sq_window_steps")
    if window_value is not None:
        try:
            if int(window_value) <= 0:
                errs.append("sq_window_steps must be a positive integer or null")
        except (TypeError, ValueError):
            errs.append(
                f"sq_window_steps must be an integer or null (got {window_value!r})"
            )

    # The version-2 key was renamed; fail loudly rather than ignoring it.
    if "freud_window_steps" in cfg:
        errs.append(
            "freud_window_steps is no longer supported — rename it to "
            "sq_window_steps"
        )

    # Explicit S(q) window bounds: non-negative integers or null; ordered when
    # both set; ambiguous when combined with the trailing-window key.
    for bound_key in ("sq_window_start_step", "sq_window_end_step"):
        bound_value = cfg.get(bound_key)
        if bound_value is not None:
            try:
                if int(bound_value) < 0:
                    errs.append(f"{bound_key} must be a non-negative integer or null")
            except (TypeError, ValueError):
                errs.append(
                    f"{bound_key} must be an integer or null (got {bound_value!r})"
                )
    start_bound = cfg.get("sq_window_start_step")
    end_bound = cfg.get("sq_window_end_step")
    try:
        if start_bound is not None and end_bound is not None \
                and int(start_bound) >= int(end_bound):
            errs.append("sq_window_start_step must be less than sq_window_end_step")
    except (TypeError, ValueError):
        pass  # already reported above
    explicit_bounds_set = start_bound is not None or end_bound is not None
    trailing_set = cfg.get("sq_window_steps") is not None
    if explicit_bounds_set and trailing_set:
        errs.append(
            "sq_window_start_step/sq_window_end_step and sq_window_steps are "
            "two ways to pin the S(q) averaging window — set one, null the "
            "other (explicit bounds are intended for exact-mode re-runs of a "
            "window found during cic screening)"
        )
    if int(cfg.get("sq_grid", 128)) < 16:
        errs.append("sq_grid must be at least 16")
    if str(cfg.get("phi_mode", "zero")).strip().lower() not in {
        "zero", "nan_mean", "smooth"
    }:
        errs.append("phi_mode must be zero, nan_mean, or smooth")
    if float(cfg.get("phi_smooth_sigma", 1.0)) <= 0:
        errs.append("phi_smooth_sigma must be positive")
    if int(cfg.get("exact_validation_frames", 3)) < 1:
        errs.append("exact_validation_frames must be at least 1")
    if float(cfg.get("exact_validation_q_max", 1.0)) <= 0:
        errs.append("exact_validation_q_max must be positive")
    tol = float(cfg.get("exact_validation_peak_rel_tol", 0.10))
    if not (0 < tol < 1):
        errs.append("exact_validation_peak_rel_tol must be in (0, 1)")
    if float(cfg.get("peak_min_height_over_median", 1.5)) <= 1:
        errs.append("peak_min_height_over_median must be greater than 1")
    prom = float(cfg.get("peak_min_relative_prominence", 0.10))
    if not (0 < prom < 1):
        errs.append("peak_min_relative_prominence must be in (0, 1)")
    if int(cfg.get("peak_min_occupied_bins", 5)) < 3:
        errs.append("peak_min_occupied_bins must be at least 3")
    if int(cfg.get("sq_stationarity_min_blocks", 3)) < 2:
        errs.append("sq_stationarity_min_blocks must be at least 2")
    if float(cfg.get("sq_spectral_distance_tolerance", 0.10)) <= 0:
        errs.append("sq_spectral_distance_tolerance must be positive")
    if float(cfg.get("sq_peak_intensity_slope_tolerance", 0.02)) <= 0:
        errs.append("sq_peak_intensity_slope_tolerance must be positive")

    if not (0.0 <= cfg["burn_in_fraction"] < 1.0):
        errs.append("burn_in_fraction must be in [0, 1)")
    if not (0.0 < cfg["fallback_fraction"] <= 1.0):
        errs.append("fallback_fraction must be in (0, 1]")
    if cfg["min_blocks_accepted"] < 1:
        errs.append("min_blocks_accepted must be >= 1")
    if errs:
        raise ValueError("Config errors:\n  " + "\n  ".join(errs))


def parse_sim_id(filename: str, traj_suffix: str) -> str:
    """Strip traj_suffix from a filename to get the simulation ID."""
    name = Path(filename).name
    if name.endswith(traj_suffix):
        return name[: -len(traj_suffix)]
    return Path(filename).stem


RE_PA = re.compile(r"pA(?P<pA>\d+)")
RE_A12 = re.compile(r"a12[_-](?P<a12>\d+(?:[p.]\d+)?)")


def parse_filename_metadata(sim_id: str) -> Dict[str, Any]:
    """Extract pA, a12, replicate from the simulation ID string."""
    rep = None
    base = sim_id
    a12_match = RE_A12.search(sim_id)
    if a12_match:
        tail = sim_id[a12_match.end():]
        rep_match = re.fullmatch(r"_(\d+)", tail)
        if rep_match:
            rep = int(rep_match.group(1))
            base = sim_id[:a12_match.end()]

    pA_raw = None
    m = RE_PA.search(base)
    if m:
        v = int(m.group("pA"))
        pA_raw = v / 10.0 if v < 10 else v / 100.0

    a12 = None
    m = RE_A12.search(sim_id)
    if m:
        a12 = float(m.group("a12").replace("p", "."))

    return {"sim_id": sim_id, "base": base, "replicate": rep, "pA": pA_raw, "a12": a12}
