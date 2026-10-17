"""Sq_pipeline_v4 — dual-engine partial structure factor pipeline for DPD block copolymers.

Engines (cfg["sq_method"]):
  "cic"   — Cloud-In-Cell particle-mesh FFT with window deconvolution (fast screening)
  "exact" — exact full-lattice direct sum over the box reciprocal lattice,
            computed from particle coordinates (publication re-runs)

Notation: q = reciprocal-space wavevector magnitude (DPD units).
All user-facing outputs (files, columns, plots) use q, S(q), q_star, L0 = 2π/q_star.
"""

from .config import load_config, save_config, config_hash
from .trajectory_reader import iter_frames, FrameData
from .thermo_parser import parse_out_file, ThermoResult
from .window_detection import detect_stable_window, WindowResult
from .particle_mesh_sq import (
    compute_particle_mesh_sq_single_frame,
    exact_direct_composition_sq,
    make_sq_object,
    resolve_q_grid,
    MeshSqResult,
)
from .exact_direct_sq import ExactDirectSq, make_exact_sq_object
from .fft_diagnostic import fft_diagnostic_frame
from .analysis_pipeline import analyze_simulation
from .batch_runner import run_batch
from .sanity_checks import run_sanity_checks
