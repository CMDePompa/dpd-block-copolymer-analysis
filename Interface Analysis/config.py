"""Configuration loading, validation, hashing, and template for interface_analysis_v2.

Follows the conventions of ``Sq_pipeline_v4/pipeline/config.py``:
a flat dict with a complete ``_DEFAULTS`` table, YAML load + validation,
a stable ``config_hash`` for provenance, and a commented template writer.

Every 'auto' key is resolved per simulation (from the box, the S(q) outputs,
or the data) and the resolved value + rule is written to
``interface_resolution.json`` — nothing is chosen silently.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# ---------------------------------------------------------------------------
# Defaults — every key the pipeline reads must have a value here.
# Units are DPD reduced units (r_c = 1) unless stated otherwise.
# ---------------------------------------------------------------------------
_DEFAULTS: Dict[str, Any] = {
    # ---- Analysis route ----
    # This is a user declaration, not a classifier. V2 never changes it.
    "morphology": "lamella",       # lamella | sphere | cylinder | gyroid
    "minority_phase": "auto",      # auto (global bead fraction) | A | B

    # ---- I/O ----
    "input_dir": ".",
    "output_dir": "./interface_output",
    "traj_suffix": ".prod.lammpstrj",
    "sims": None,                    # optional list of sim_ids (or filenames) to restrict the batch
    "sq_results_dir": None,          # per-sim S(q) outputs (q_star, L0, accepted windows); null = don't use
    "sq_pipeline_dir": None,         # Structure Factor reader location; null = auto-locate

    # ---- Bead types ----
    "A_type_id": 1,
    "B_type_id": 2,
    "allow_variable_natoms": False,

    # ---- Frame selection ----
    # analysis_start_step: "auto" -> use the S(q) accepted window start when
    # available, else fall back to burn_in_fraction; an integer uses that
    # timestep; null reads the whole file.
    "analysis_start_step": "auto",
    "burn_in_fraction": 0.3,         # fallback burn-in when no S(q) window is available
    "last_step_window": None,        # timestep span from trajectory end (e.g. 100000)
    "frame_stride": 1,               # analyze every Nth selected frame
    "max_frames": None,              # cap analyzed frames (keeps the LAST max_frames — most equilibrated)
    "use_frame_index": True,         # build/cache a byte-offset index for fast frame seeking

    # ---- Density grid ----
    "target_voxel_size": 0.5,        # r_c; grid n_i = round(L_i / target); actual voxel recorded
    "grid_shape": None,              # explicit [nx, ny, nz] override (null = derive from voxel size)
    "smoothing_sigma": 0.75,         # r_c (PHYSICAL units); converted per axis to voxels
    "deposition": "cic",             # "cic" (cloud-in-cell) | "ngp" (nearest grid point)

    # ---- Composition field ----
    "density_cutoff_fraction": 0.05, # voxels with rho_A+rho_B < frac * mean are masked invalid
    "interface_level": 0.0,          # phi level set used as the A/B dividing surface

    # ---- General 3-D interfaces (sphere/cylinder/gyroid) ----
    "surface_normal_samples": 2048,  # area-systematic surface locations per frame
    "normal_profile_points": 65,     # samples along each local surface normal (odd)
    "normal_profile_halfwidth": "auto",  # auto -> fraction of S(q) period below
    "normal_profile_halfwidth_fraction": 0.25,
    "cylinder_axis": "auto",        # auto from normal tensor | explicit [x,y,z]

    # ---- Lamellar normal ----
    "n0_mode": "auto",               # "auto" (FFT + S(q) crosscheck) | "fft" | "fixed"
    "n0_fixed": None,                # [x, y, z] used when n0_mode == "fixed"
    "secondary_peak_warn_ratio": 0.5,  # warn if 2nd FFT peak intensity / 1st > this (polydomain)
    "qstar_crosscheck_tol": 0.15,    # warn if |q*_fft| vs S(q) q_star relative difference > this

    # ---- Oblique resampling ----
    "interp_order": 3,               # spline order for map_coordinates (mode='grid-wrap')
    "lateral_voxel_size": "auto",    # sampling step along u, v ('auto' -> target_voxel_size)
    "normal_sample_size": "auto",    # sampling step along w ('auto' -> target_voxel_size / 2)
    "n_periods": "auto",             # auto -> shortest verified box-periodic normal closure
    "lateral_extent": "auto",        # physical extent of the (u, v) window ('auto' -> median box edge)

    # ---- Interface detection ----
    "phi_hysteresis": 0.10,          # min |phi| excursion on both sides for a crossing to count
    "assignment_tolerance": 0.35,    # fraction of L0/2: max |offset from ideal plane| for unambiguous label
    "min_column_amplitude": 0.20,    # columns with max|phi| below this are flagged low-amplitude
    "close_pair_fraction": 0.30,     # crossings closer than this fraction of L0/2 form a "close pair"
    "max_height_infill_fraction": 0.20,  # above this, slope/area metrics are invalid

    # ---- Intrinsic width ----
    "width_fit_halfwidth_fraction": 0.25,  # tanh-fit window half-width as fraction of L0
    "width_min_r2": 0.90,            # fits below this R^2 are counted as failed
    "obliquity_correction": True,    # multiply local widths by cos(theta_local) from height-field slopes
    "width_fit_max_columns": 1024,   # per interface per frame; columns subsampled beyond this

    # ---- Penetration / mixing ----
    "mixing_thresholds": [0.25, 0.5, 0.75],  # |phi| < threshold => "mixed"; primary is 0.5
    "penetration_halfwindow": "auto",        # signed-distance integration half-window ('auto' -> L0/2)

    # ---- Chain-region density profiles (initial A segment and remaining A/B blocks) ----
    "density_profile_enabled": False,
    "density_profile_chain_length": None,
    "density_profile_pure_a_length": None,
    "density_profile_bins": 100,
    "density_profile_alignment_sigma_bins": 1.0,
    "density_profile_interface_window_fraction": 0.05,
    # If no explicit bead_index-like dump column exists, opt in only when
    # atom IDs are known to increase along each bonded molecule.
    "density_profile_allow_inferred_bead_order": False,

    # ---- Topology ----
    "topology_thresholds": [-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6],
    "topology_threshold_mode": "normalized_plateaus",  # normalize phase offset/amplitude first
    "topology_connectivity": 6,      # 6 (face) or 26 (vertex) voxel connectivity for labeling

    # ---- Outputs ----
    "save_maps": True,               # interface_maps.npz (height/spacing/width/defect maps)
    "save_profiles": True,           # interface_profiles.npz (aligned profiles, spectra)
    "plots": True,
    "diagnostic_plot_stride": 0,     # 0 = plot only the last analyzed frame; N = every Nth frame too
    "autocorrelation_estimate": True,  # estimate integrated autocorrelation time / N_eff for key series

    # ---- Calibration mode (run with --calibrate) ----
    "calibration_frames": 2,         # frames (from the end of the selection) per parameter combo
    "calibration_voxel_sizes": [0.4, 0.5, 0.667],
    "calibration_sigmas": [0.5, 0.75, 1.0],
    "calibration_cutoffs": [0.02, 0.05, 0.10],
    "calibration_hysteresis": [0.05, 0.10, 0.20],
}

_VALID_DEPOSITION = ("cic", "ngp")
_VALID_N0_MODE = ("auto", "fft", "fixed")
_VALID_MORPHOLOGY = ("lamella", "sphere", "cylinder", "gyroid")
_VALID_MINORITY_PHASE = ("auto", "A", "B")


def load_config(path: Optional[str | Path]) -> Dict[str, Any]:
    """Load a YAML config file and fill in missing keys with defaults.

    ``path=None`` returns pure defaults (still validated).
    Unknown keys raise, so typos never silently fall back to defaults.
    """
    cfg = dict(_DEFAULTS)
    if path is not None:
        path = Path(path)
        with path.open("r") as fh:
            user = yaml.safe_load(fh) or {}
        if not isinstance(user, dict):
            raise ValueError(f"Config file {path} did not parse to a mapping.")
        unknown = sorted(set(user) - set(_DEFAULTS) - {"_config_hash"})
        if unknown:
            raise ValueError(
                f"Unknown config key(s): {', '.join(unknown)}. "
                "Check spelling against the template (run with --make-template)."
            )
        user.pop("_config_hash", None)
        cfg.update(user)
    _validate(cfg)
    return cfg


def save_config(cfg: Dict[str, Any], path: str | Path) -> None:
    """Write config dict as YAML with its hash appended (provenance)."""
    path = Path(path)
    out = dict(cfg)
    out["_config_hash"] = config_hash(cfg)
    with path.open("w") as fh:
        yaml.dump(out, fh, default_flow_style=False, sort_keys=True)


def config_hash(cfg: Dict[str, Any]) -> str:
    """SHA-256 hash of the config (excluding the hash field itself)."""
    clean = {k: v for k, v in cfg.items() if k != "_config_hash"}
    blob = json.dumps(clean, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def is_auto(value: Any) -> bool:
    """True if a config value is the literal string 'auto' (case-insensitive)."""
    return isinstance(value, str) and value.strip().lower() == "auto"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _positive_number(cfg: Dict[str, Any], key: str, errs: List[str], allow_auto: bool = False) -> None:
    v = cfg.get(key)
    if allow_auto and is_auto(v):
        return
    try:
        if float(v) <= 0:
            errs.append(f"{key} must be positive" + (" or 'auto'" if allow_auto else ""))
    except (TypeError, ValueError):
        errs.append(f"{key} must be a number{' or ''auto''' if allow_auto else ''} (got {v!r})")


def _validate(cfg: Dict[str, Any]) -> None:
    errs: List[str] = []

    for key in ("target_voxel_size", "smoothing_sigma", "phi_hysteresis",
                "width_fit_halfwidth_fraction", "qstar_crosscheck_tol"):
        _positive_number(cfg, key, errs)
    for key in ("lateral_voxel_size", "normal_sample_size", "lateral_extent",
                "penetration_halfwindow", "n_periods", "normal_profile_halfwidth"):
        _positive_number(cfg, key, errs, allow_auto=True)

    morphology = str(cfg.get("morphology", "")).strip().lower()
    if morphology not in _VALID_MORPHOLOGY:
        errs.append(f"morphology must be one of {_VALID_MORPHOLOGY} (got {cfg.get('morphology')!r})")
    else:
        cfg["morphology"] = morphology
    minority = str(cfg.get("minority_phase", "")).strip()
    if minority.lower() == "auto":
        cfg["minority_phase"] = "auto"
    elif minority.upper() in ("A", "B"):
        cfg["minority_phase"] = minority.upper()
    else:
        errs.append(
            f"minority_phase must be one of {_VALID_MINORITY_PHASE} "
            f"(got {cfg.get('minority_phase')!r})"
        )

    try:
        if not -1.0 < float(cfg["interface_level"]) < 1.0:
            errs.append("interface_level must be in (-1, 1)")
    except (TypeError, ValueError):
        errs.append("interface_level must be a number in (-1, 1)")
    try:
        if int(cfg["surface_normal_samples"]) < 32:
            errs.append("surface_normal_samples must be an integer >= 32")
    except (TypeError, ValueError):
        errs.append("surface_normal_samples must be an integer >= 32")
    try:
        if int(cfg["normal_profile_points"]) < 9:
            errs.append("normal_profile_points must be an integer >= 9")
    except (TypeError, ValueError):
        errs.append("normal_profile_points must be an integer >= 9")
    try:
        frac = float(cfg["normal_profile_halfwidth_fraction"])
        if not 0.0 < frac <= 0.5:
            errs.append("normal_profile_halfwidth_fraction must be in (0, 0.5]")
    except (TypeError, ValueError):
        errs.append("normal_profile_halfwidth_fraction must be a number in (0, 0.5]")

    cylinder_axis = cfg.get("cylinder_axis")
    if not is_auto(cylinder_axis):
        ok = isinstance(cylinder_axis, (list, tuple)) and len(cylinder_axis) == 3
        if ok:
            try:
                ok = any(float(x) != 0.0 for x in cylinder_axis)
            except (TypeError, ValueError):
                ok = False
        if not ok:
            errs.append("cylinder_axis must be 'auto' or a nonzero [x, y, z] vector")

    if cfg["deposition"] not in _VALID_DEPOSITION:
        errs.append(f"deposition must be one of {_VALID_DEPOSITION} (got {cfg['deposition']!r})")
    if cfg["n0_mode"] not in _VALID_N0_MODE:
        errs.append(f"n0_mode must be one of {_VALID_N0_MODE} (got {cfg['n0_mode']!r})")
    if cfg["n0_mode"] == "fixed":
        v = cfg.get("n0_fixed")
        ok = isinstance(v, (list, tuple)) and len(v) == 3
        if ok:
            try:
                ok = any(float(x) != 0.0 for x in v)
            except (TypeError, ValueError):
                ok = False
        if not ok:
            errs.append("n0_mode='fixed' requires n0_fixed: [x, y, z] with a nonzero component")

    if cfg["grid_shape"] is not None:
        gs = cfg["grid_shape"]
        try:
            ok = isinstance(gs, (list, tuple)) and len(gs) == 3 and all(int(n) >= 8 for n in gs)
        except (TypeError, ValueError):
            ok = False
        if not ok:
            errs.append("grid_shape must be null or a list of three integers >= 8")

    start = cfg.get("analysis_start_step")
    if start is not None and not is_auto(start):
        try:
            if int(start) < 0:
                errs.append("analysis_start_step must be >= 0, 'auto', or null")
        except (TypeError, ValueError):
            errs.append(f"analysis_start_step must be an integer, 'auto', or null (got {start!r})")

    if not (0.0 <= float(cfg["burn_in_fraction"]) < 1.0):
        errs.append("burn_in_fraction must be in [0, 1)")
    if cfg["last_step_window"] is not None:
        try:
            if int(cfg["last_step_window"]) < 1:
                errs.append("last_step_window must be >= 1 or null")
        except (TypeError, ValueError):
            errs.append("last_step_window must be an integer >= 1 or null")
    if int(cfg["frame_stride"]) < 1:
        errs.append("frame_stride must be >= 1")
    if cfg["max_frames"] is not None and int(cfg["max_frames"]) < 1:
        errs.append("max_frames must be >= 1 or null")
    if int(cfg["interp_order"]) not in (0, 1, 2, 3, 4, 5):
        errs.append("interp_order must be an integer in [0, 5]")
    if not (0.0 < float(cfg["assignment_tolerance"]) <= 0.5):
        errs.append("assignment_tolerance must be in (0, 0.5] (fraction of L0/2)")
    if not (0.0 <= float(cfg["min_column_amplitude"]) < 1.0):
        errs.append("min_column_amplitude must be in [0, 1)")
    if not (0.0 <= float(cfg["max_height_infill_fraction"]) < 1.0):
        errs.append("max_height_infill_fraction must be in [0, 1)")
    if not (0.0 < float(cfg["density_cutoff_fraction"]) < 1.0):
        errs.append("density_cutoff_fraction must be in (0, 1)")
    if not (0.0 < float(cfg["width_min_r2"]) <= 1.0):
        errs.append("width_min_r2 must be in (0, 1]")
    if int(cfg["topology_connectivity"]) not in (6, 26):
        errs.append("topology_connectivity must be 6 or 26")
    if cfg["topology_threshold_mode"] not in ("absolute_phi", "normalized_plateaus"):
        errs.append("topology_threshold_mode must be 'absolute_phi' or 'normalized_plateaus'")

    if bool(cfg["density_profile_enabled"]):
        try:
            chain_length = int(cfg["density_profile_chain_length"])
            pure_length = int(cfg["density_profile_pure_a_length"])
            if chain_length <= 1:
                errs.append("density_profile_chain_length must be > 1 when profiles are enabled")
            if not 0 < pure_length < chain_length:
                errs.append(
                    "density_profile_pure_a_length must be between 1 and chain_length - 1"
                )
        except (TypeError, ValueError):
            errs.append(
                "density_profile_chain_length and density_profile_pure_a_length must be "
                "integers when density_profile_enabled is true"
            )
    try:
        if int(cfg["density_profile_bins"]) < 16:
            errs.append("density_profile_bins must be >= 16")
    except (TypeError, ValueError):
        errs.append("density_profile_bins must be an integer >= 16")
    try:
        if float(cfg["density_profile_alignment_sigma_bins"]) < 0:
            errs.append("density_profile_alignment_sigma_bins must be >= 0")
    except (TypeError, ValueError):
        errs.append("density_profile_alignment_sigma_bins must be a number >= 0")
    try:
        window = float(cfg["density_profile_interface_window_fraction"])
        if not 0 < window < 0.25:
            errs.append("density_profile_interface_window_fraction must be in (0, 0.25)")
    except (TypeError, ValueError):
        errs.append("density_profile_interface_window_fraction must be a number in (0, 0.25)")

    mt = cfg["mixing_thresholds"]
    if not (isinstance(mt, (list, tuple)) and mt and all(0.0 < float(t) < 1.0 for t in mt)):
        errs.append("mixing_thresholds must be a non-empty list of values in (0, 1)")
    tt = cfg["topology_thresholds"]
    if not (isinstance(tt, (list, tuple)) and tt and all(-1.0 < float(t) < 1.0 for t in tt)):
        errs.append("topology_thresholds must be a non-empty list of values in (-1, 1)")

    if errs:
        raise ValueError("Config errors:\n  " + "\n  ".join(errs))


def validate_config(cfg: Dict[str, Any]) -> None:
    """Validate an already-loaded config after command-line overrides."""
    _validate(cfg)


# ---------------------------------------------------------------------------
# Simulation-ID helpers (same conventions as the S(q) pipeline)
# ---------------------------------------------------------------------------
def parse_sim_id(filename: str, traj_suffix: str) -> str:
    """Strip traj_suffix from a filename to get the simulation ID."""
    name = Path(filename).name
    if name.endswith(traj_suffix):
        return name[: -len(traj_suffix)]
    return Path(filename).stem


RE_PA = re.compile(r"pA(?P<pA>\d+)")
RE_A12 = re.compile(r"a12[_-](?P<a12>[\d.]+)")
RE_REP = re.compile(r"^(?P<base>.+)_(?P<rep>\d+)$")


def parse_filename_metadata(sim_id: str) -> Dict[str, Any]:
    """Extract pA, a12, replicate from the simulation ID string."""
    rep = None
    base = sim_id
    m = RE_REP.match(sim_id)
    if m:
        base = m.group("base")
        rep = int(m.group("rep"))

    pA_raw = None
    m = RE_PA.search(base)
    if m:
        v = int(m.group("pA"))
        pA_raw = v / 10.0 if v < 10 else v / 100.0

    a12 = None
    m = RE_A12.search(base)
    if m:
        a12 = float(m.group("a12"))

    return {"sim_id": sim_id, "base": base, "replicate": rep, "pA": pA_raw, "a12": a12}


# ---------------------------------------------------------------------------
# Template
# ---------------------------------------------------------------------------
def make_template(path: str | Path) -> None:
    """Write a fully commented YAML config template to *path*."""
    Path(path).write_text(_TEMPLATE, encoding="utf-8")


_TEMPLATE = """\
# =============================================================================
# interface_analysis V2 — configuration template
# =============================================================================
# Copy this file, edit, and pass to run_interface.py --config.
# Missing keys fall back to built-in defaults; unknown keys are an error.
# Units: DPD reduced units (r_c = 1) unless noted. 'auto' values are resolved
# per simulation and recorded in interface_resolution.json.

# ---------------------------------------------------------------------------
# REQUIRED SCIENTIFIC CHOICE
# ---------------------------------------------------------------------------
# Declare the morphology already established by visual inspection and S(q).
# V2 does not classify or silently change this choice.
morphology: lamella                 # lamella | sphere | cylinder | gyroid
# For spheres/cylinders, auto uses the less abundant global bead type.
# Set A or B explicitly if the intended domain phase differs.
minority_phase: auto                # auto | A | B

# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
input_dir: ./traj files            # directory holding *.prod.lammpstrj files
output_dir: ./interface_output     # per-sim folders + combined tables written here
traj_suffix: .prod.lammpstrj
sims: null                         # null = all trajectories; or a list of sim_ids to restrict
# Per-sim S(q) pipeline outputs (analysis_summary.csv, accepted_windows.json).
# When set, q_star/L0 are cross-checked and the accepted window provides the
# default analysis_start_step. Never written to — read-only.
sq_results_dir: ../Structure Factor/output
sq_pipeline_dir: null              # sibling Structure Factor directory; null = auto-locate

# ---------------------------------------------------------------------------
# Bead types
# ---------------------------------------------------------------------------
A_type_id: 1
B_type_id: 2
allow_variable_natoms: false

# ---------------------------------------------------------------------------
# Frame selection
# ---------------------------------------------------------------------------
# analysis_start_step: 'auto' uses the S(q) accepted-window start when
# available (falls back to burn_in_fraction), an integer skips all earlier
# frames at the file level, null reads the whole file.
analysis_start_step: auto
burn_in_fraction: 0.3
last_step_window: null             # e.g. 100000 keeps the final 100,000 timesteps
frame_stride: 1                    # analyze every Nth selected frame
max_frames: null                   # cap; keeps the LAST max_frames (most equilibrated)
use_frame_index: true              # cache byte offsets for fast seeking in multi-GB files

# ---------------------------------------------------------------------------
# Density grid  (rho_A, rho_B via periodic cloud-in-cell + Gaussian smoothing)
# ---------------------------------------------------------------------------
# Defaults are physically motivated for DPD melts (bead diameter ~ r_c = 1,
# rho = 3): voxel 0.5 r_c resolves the interface region; sigma = 0.75 r_c
# averages ~50 beads per smoothing volume (phi noise ~0.1) while staying well
# below the lamellar period (L0 ~ 10-14 r_c). Run --calibrate to verify on
# your systems; do NOT tune by visual appeal alone.
target_voxel_size: 0.5             # r_c
grid_shape: null                   # explicit [nx, ny, nz] override
smoothing_sigma: 0.75              # r_c (physical units, converted per axis to voxels)
deposition: cic                    # cic | ngp

# ---------------------------------------------------------------------------
# Composition field  phi = (rho_A - rho_B) / (rho_A + rho_B)
# ---------------------------------------------------------------------------
density_cutoff_fraction: 0.05      # mask voxels with rho_A+rho_B < frac * mean (should be ~0 in a melt)
interface_level: 0.0               # phi level set defining the A/B dividing surface

# ---------------------------------------------------------------------------
# Local-normal analysis for sphere, cylinder, and gyroid interfaces
# ---------------------------------------------------------------------------
surface_normal_samples: 2048       # area-systematic surface samples per frame
normal_profile_points: 65          # points along each local normal (made odd if needed)
normal_profile_halfwidth: auto      # physical r_c; auto uses the fraction below
normal_profile_halfwidth_fraction: 0.25 # fraction of S(q) period L0
cylinder_axis: auto                 # auto from surface normals | explicit [x, y, z]

# ---------------------------------------------------------------------------
# Lamellar-only normal n0 (ignored for sphere/cylinder/gyroid)
# ---------------------------------------------------------------------------
n0_mode: auto                      # auto (FFT + S(q) crosscheck) | fft | fixed
n0_fixed: null                     # [x, y, z], used only when n0_mode: fixed
secondary_peak_warn_ratio: 0.5     # polydomain warning threshold (2nd/1st FFT peak intensity)
qstar_crosscheck_tol: 0.15         # relative |q*| disagreement (FFT vs S(q)) that triggers a warning

# ---------------------------------------------------------------------------
# Lamellar-only oblique resampling (ignored for sphere/cylinder/gyroid)
# ---------------------------------------------------------------------------
interp_order: 3                    # map_coordinates spline order (mode='grid-wrap'); 1 = trilinear
lateral_voxel_size: auto           # u, v sampling step (auto -> target_voxel_size)
normal_sample_size: auto           # w sampling step (auto -> target_voxel_size / 2)
n_periods: auto                    # auto -> shortest verified box-periodic normal closure
lateral_extent: auto               # (u, v) window size (auto -> median box edge length)

# ---------------------------------------------------------------------------
# Lamellar-only interface detection (ignored for sphere/cylinder/gyroid)
# ---------------------------------------------------------------------------
phi_hysteresis: 0.10               # noise immunity: required |phi| excursion around a crossing
assignment_tolerance: 0.35         # fraction of L0/2; larger offsets from the ideal plane = ambiguous
min_column_amplitude: 0.20         # columns whose max|phi| is below this are flagged low-amplitude
close_pair_fraction: 0.30          # crossing pairs closer than this fraction of L0/2 are flagged
max_height_infill_fraction: 0.20  # invalidate slope/area metrics above this missing fraction

# ---------------------------------------------------------------------------
# Lamellar intrinsic width settings. Nonlamellar morphologies use local-normal
# tanh fits with normal_profile_* settings above.
# ---------------------------------------------------------------------------
width_fit_halfwidth_fraction: 0.25 # fit window half-width as fraction of L0
width_min_r2: 0.90
obliquity_correction: true         # correct local widths by cos(theta) from height-field slopes
width_fit_max_columns: 1024        # per interface per frame (columns subsampled beyond this)

# ---------------------------------------------------------------------------
# Penetration / mixing
# ---------------------------------------------------------------------------
mixing_thresholds: [0.25, 0.5, 0.75]   # f_mix reported at each |phi| threshold
penetration_halfwindow: auto           # signed-distance integration half-window (auto -> L0/2)

# ---------------------------------------------------------------------------
# Chain-region density profiles for fixed A/B blocks
# ---------------------------------------------------------------------------
# Uses the detected reciprocal-lattice mode, so axis-aligned and oblique
# lamellae share the same exactly periodic coordinate. These are global
# projected profiles; use the local width analysis for intrinsic width.
density_profile_enabled: false
density_profile_chain_length: null     # required integer when enabled (32 for this campaign)
density_profile_pure_a_length: null    # pure A block length (16 for this campaign)
density_profile_bins: 100
density_profile_alignment_sigma_bins: 1.0
density_profile_interface_window_fraction: 0.05
# Enable only when atom IDs increase along chain order. An explicit dump
# column named bead_index/bead/monomer_index/monomer/chain_index is preferred.
density_profile_allow_inferred_bead_order: false

# ---------------------------------------------------------------------------
# Topology (thresholded composition sets, periodic-aware)
# ---------------------------------------------------------------------------
topology_thresholds: [-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6]
topology_threshold_mode: normalized_plateaus # comparable thresholds despite shifted phase plateaus
topology_connectivity: 6           # 6 (face) | 26 (vertex)

# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------
save_maps: true
save_profiles: true
plots: true
diagnostic_plot_stride: 0          # 0 = plot last analyzed frame only; N = every Nth frame too
autocorrelation_estimate: true

# ---------------------------------------------------------------------------
# Lamellar calibration mode (run_interface.py --calibrate)
# ---------------------------------------------------------------------------
calibration_frames: 2
calibration_voxel_sizes: [0.4, 0.5, 0.667]
calibration_sigmas: [0.5, 0.75, 1.0]
calibration_cutoffs: [0.02, 0.05, 0.10]
calibration_hysteresis: [0.05, 0.10, 0.20]
"""
