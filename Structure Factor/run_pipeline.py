#!/usr/bin/env python3
"""CLI entry point for structure-factor pipeline version 4 (dual-engine).

Usage (HPC / command line):
    python run_pipeline.py --config config.yaml
    python run_pipeline.py --config config.yaml --sq-method exact --traj-file X.prod.lammpstrj
    python run_pipeline.py --config config.yaml --input-dir /path/to/sims --output-dir /path/to/out
    python run_pipeline.py --config config.yaml --test-mode   # process first 2 frames (sanity check)

Engines: sq_method "cic" (fast particle-mesh screening, default) or "exact"
(full-lattice direct sum from particle coordinates, for publication re-runs).
The config YAML is always required for a full run. CLI flags override config values.
Notation: q = reciprocal-space wavevector magnitude (DPD units).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Validated particle-mesh S(q) pipeline for DPD block copolymer simulations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--config", "-c", default=None, help="Path to YAML config file.")
    ap.add_argument("--input-dir", default=None, help="Override config input_dir.")
    ap.add_argument("--output-dir", default=None, help="Override config output_dir.")
    ap.add_argument("--traj-file", default=None,
                    help="Analyze one trajectory file instead of the whole input directory.")
    ap.add_argument("--traj-suffix", default=None, help="Override traj_suffix.")
    ap.add_argument("--sq-method", default=None, choices=["cic", "exact"],
                    help="Override config sq_method: cic (fast particle-mesh "
                         "screening) or exact (full-lattice direct sum, for "
                         "publication re-runs).")
    ap.add_argument("--aggregate-only", action="store_true",
                    help="Aggregate existing per-simulation summaries in output_dir and exit.")
    ap.add_argument("--test-mode", action="store_true",
                    help="Quick sanity check: analyze first 2 frames and print key numbers.")
    ap.add_argument("--make-template", default=None, metavar="PATH",
                    help="Write a template config YAML to PATH and exit.")
    ap.add_argument("--version", action="store_true", help="Print dependency versions and exit.")
    return ap


def print_versions() -> None:
    """Print versions of key dependencies for reproducibility."""
    import platform
    print(f"Python:    {sys.version}")
    print(f"Platform:  {platform.platform()}")
    try:
        import numpy as np
        print(f"NumPy:     {np.__version__}")
    except ImportError:
        print("NumPy:     NOT INSTALLED")
    try:
        import pandas as pd
        print(f"Pandas:    {pd.__version__}")
    except ImportError:
        print("Pandas:    NOT INSTALLED")
    try:
        import matplotlib
        print(f"Matplotlib:{matplotlib.__version__}")
    except ImportError:
        print("Matplotlib:NOT INSTALLED  ← required for plots")
    try:
        import yaml
        print(f"PyYAML:    {yaml.__version__}")
    except ImportError:
        print("PyYAML:    NOT INSTALLED  ← required for config")
    try:
        import scipy
        print(f"SciPy:     {scipy.__version__}  (peak detection + phi_mode=smooth)")
    except ImportError:
        print("SciPy:     not installed  (optional; enables local-maxima peak detection)")


def run_test_mode(cfg: dict) -> None:
    """Read two frames and exercise the selected S(q) engine.

    cic: requires agreement with the exact-direct low-q reference.
    exact: computes the reference spectrum and checks internal consistency
    (finite peak, nonzero cross-partial).
    """
    from pipeline.config import parse_sim_id
    from pipeline.trajectory_reader import iter_frames
    from pipeline.fft_diagnostic import fft_diagnostic_frame

    sq_method = str(cfg.get("sq_method", "cic")).strip().lower()

    input_dir = Path(cfg["input_dir"])
    suffix = cfg.get("traj_suffix", ".prod.lammpstrj")
    trajs = sorted([p for p in input_dir.iterdir() if p.is_file() and p.name.endswith(suffix)])
    if not trajs:
        print(f"ERROR: No trajectories found in {input_dir} with suffix '{suffix}'")
        sys.exit(1)
    preferred = cfg.get("test_traj_file")
    if preferred:
        selected = input_dir / str(preferred)
        if not selected.is_file():
            print(f"ERROR: test_traj_file does not exist: {selected}")
            sys.exit(1)
        trajs = [selected]

    print(f"\n=== TEST MODE ({sq_method}): checking first trajectory ({trajs[0].name}) ===")

    start_step = cfg.get("analysis_start_step")
    if start_step is not None:
        start_step = int(start_step)
        print(f"analysis_start_step={start_step}: skipping earlier frames without parsing.")
    window_steps = cfg.get("sq_window_steps")
    if window_steps is not None:
        from pipeline.trajectory_reader import final_timestep
        cutoff = final_timestep(trajs[0]) - int(window_steps)
        start_step = cutoff if start_step is None else max(start_step, cutoff)
        print(
            f"sq_window_steps={window_steps}: testing frames at or after "
            f"timestep {start_step}."
        )
    elif cfg.get("sq_window_start_step") is not None:
        explicit_start = int(cfg["sq_window_start_step"])
        start_step = explicit_start if start_step is None else max(start_step, explicit_start)
        print(
            f"sq_window_start_step={explicit_start}: testing frames at or after "
            f"timestep {start_step}."
        )

    frames_read = []
    for frame in iter_frames(trajs[0], stride=1, A_type_id=cfg["A_type_id"],
                             B_type_id=cfg["B_type_id"], start_step=start_step):
        frames_read.append(frame)
        if len(frames_read) >= 2:
            break

    if not frames_read:
        if start_step is not None:
            print(f"ERROR: No frames at or after analysis_start_step={start_step}. "
                  "The trajectory may end before this timestep.")
        else:
            print("ERROR: No frames read.")
        sys.exit(1)

    print(f"Read {len(frames_read)} frame(s).")

    # Resolve 'auto' q-grid settings from the first frame's box
    from pipeline.particle_mesh_sq import resolve_q_grid
    resolved = resolve_q_grid(frames_read[0].L, cfg)
    print(
        f"Resolved q-grid: dq=({resolved['dq_x']:.5f}, {resolved['dq_y']:.5f}, "
        f"{resolved['dq_z']:.5f}), q_compute=[{resolved['q_compute_min']:.5f}, "
        f"{resolved['q_compute_max']:.3f}], q_bins={resolved['q_bins']} "
        f"(width {resolved['q_bin_width']:.5f}), "
        f"q_peak_search=[{resolved['q_peak_search_min']:.5f}, {resolved['q_peak_search_max']:.3f}], "
        f"~{resolved['n_kvec_estimate']} reciprocal vectors"
    )

    # Create the reusable engine object.
    sf = None
    try:
        if sq_method == "exact":
            from pipeline.exact_direct_sq import make_exact_sq_object
            sf = make_exact_sq_object(
                frames_read[0].L,
                q_compute_min=resolved["q_compute_min"],
                q_compute_max=resolved["q_compute_max"],
                q_bins=resolved["q_bins"],
            )
        else:
            from pipeline.particle_mesh_sq import make_sq_object
            sf = make_sq_object(
                frames_read[0].L,
                q_compute_min=resolved["q_compute_min"],
                q_compute_max=resolved["q_compute_max"],
                q_bins=resolved["q_bins"],
                grid_n=int(cfg.get("sq_grid", 128)),
            )
    except Exception as exc:
        print(f"  S(q) engine ({sq_method}) initialization failed: {exc}")

    for fr in frames_read:
        print(f"  timestep={fr.timestep}, N={fr.n_atoms}, L={fr.L.round(3)}")
        n_A = (fr.types == cfg["A_type_id"]).sum()
        n_B = (fr.types == cfg["B_type_id"]).sum()
        print(f"  A beads: {n_A}, B beads: {n_B}, fraction A: {n_A/(n_A+n_B):.3f}")

        d = fft_diagnostic_frame(
            fr, A_type_id=cfg["A_type_id"], B_type_id=cfg["B_type_id"],
            grid_n=cfg.get("fft_grid", 64),
            phi_mode=str(cfg.get("phi_mode", "zero")),
            phi_smooth_sigma=float(cfg.get("phi_smooth_sigma", 1.0)),
            q_peak_search_min=resolved["q_peak_search_min"],
            q_peak_search_max=resolved["q_peak_search_max"],
        )
        print(f"  FFT q_star={d['q_star_fft']:.4f}, phi_rms={d['phi_rms']:.4f}, "
              f"peak_height={d['peak_height_fft']:.4f}")

        if sf is not None:
            try:
                import numpy as np
                from pipeline.particle_mesh_sq import (
                    exact_direct_composition_sq,
                    validate_mesh_against_exact,
                    _find_dominant_low_q_peak,
                )
                sq_d = sf.compute(
                    fr, A_type_id=cfg["A_type_id"], B_type_id=cfg["B_type_id"],
                )
                peak = _find_dominant_low_q_peak(
                    sq_d["q"], sq_d["S_comp"],
                    resolved["q_peak_search_min"], resolved["q_peak_search_max"],
                    resolved["q_bins"],
                )
                q_star = peak["q_star"]
                L0 = 6.28318 / q_star if q_star > 0 else float("nan")
                print(
                    f"  [{sq_method}] q_star={q_star:.4f}, L0={L0:.3f}, "
                    f"peak_height={peak['peak_height']:.4f}, "
                    f"method={peak['method']}, n_candidates={peak['n_candidates']}"
                )
                if sq_method == "exact":
                    # This engine IS the exact reference — verify internal
                    # consistency. A significant peak is not required because
                    # a valid disordered spectrum should leave q_star undefined.
                    finite_cc = sq_d["S_cc_normalized"][
                        np.isfinite(sq_d["S_cc_normalized"])
                    ]
                    if not len(finite_cc):
                        raise RuntimeError("no finite normalized concentration spectrum")
                    finite_ab = sq_d["S_AB"][np.isfinite(sq_d["S_AB"])]
                    if not len(finite_ab) or float(np.max(np.abs(finite_ab))) < 1e-6:
                        raise RuntimeError("S_AB is ~0 at every q — computation error")
                    print("  TEST PASSED — exact full-lattice S(q) computed.")
                else:
                    exact = exact_direct_composition_sq(
                        fr, sf.bin_edges,
                        A_type_id=cfg["A_type_id"],
                        B_type_id=cfg["B_type_id"],
                        q_max=float(cfg.get("exact_validation_q_max", 1.0)),
                    )
                    validation = validate_mesh_against_exact(
                        sq_d, exact,
                        q_min=resolved["q_peak_search_min"],
                        q_max=min(
                            resolved["q_peak_search_max"],
                            float(cfg.get("exact_validation_q_max", 1.0)),
                        ),
                        peak_rel_tolerance=float(
                            cfg.get("exact_validation_peak_rel_tol", 0.10)
                        ),
                    )
                    print(
                        f"  Exact q_star={validation['q_star_exact']:.4f}, "
                        f"{sq_method} peak error={validation['peak_rel_error']:.2%}"
                    )
                    if not validation["passed"]:
                        raise RuntimeError(
                            f"{sq_method}/exact validation failed: "
                            f"{validation['reason']}"
                        )
                    print(f"  TEST PASSED — {sq_method} agrees with the "
                          "exact-direct low-q reference.")
            except Exception as exc:
                print(f"  S(q) test FAILED: {exc}")
                raise
        else:
            raise RuntimeError(f"S(q) engine ({sq_method}) object is unavailable.")


def main(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)

    if args.version:
        print_versions()
        return

    if args.make_template:
        from pipeline.config import make_template
        make_template(args.make_template)
        print(f"Template config written to: {args.make_template}")
        return

    if args.config is None:
        ap.print_help()
        print("\nERROR: --config is required for a batch run. Use --make-template to generate a template.")
        sys.exit(1)

    from pipeline.config import load_config
    cfg = load_config(args.config)

    # CLI overrides
    if args.input_dir:
        cfg["input_dir"] = args.input_dir
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.traj_suffix:
        cfg["traj_suffix"] = args.traj_suffix
    if args.sq_method:
        cfg["sq_method"] = args.sq_method

    if args.aggregate_only:
        from pipeline.batch_runner import aggregate_existing_outputs
        summary_path = aggregate_existing_outputs(cfg["output_dir"], logger=print)
        print(f"\nAggregation complete. Summary: {summary_path}")
        return

    if args.test_mode:
        run_test_mode(cfg)
        return

    if args.traj_file:
        import datetime
        import pandas as pd

        from pipeline.analysis_pipeline import analyze_simulation
        from pipeline.config import save_config

        traj_path = Path(args.traj_file).expanduser().resolve()
        if not traj_path.is_file():
            print(f"ERROR: trajectory file not found: {traj_path}")
            sys.exit(1)
        if not traj_path.name.endswith(cfg.get("traj_suffix", ".prod.lammpstrj")):
            print(
                f"ERROR: trajectory file does not end with "
                f"{cfg.get('traj_suffix', '.prod.lammpstrj')}: {traj_path.name}"
            )
            sys.exit(1)

        output_root = Path(cfg["output_dir"]).expanduser().resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        run_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_config(cfg, output_root / f"run_config_single_{run_ts}.yaml")

        summary = analyze_simulation(traj_path, cfg, output_root, logger=print)
        sim_id = summary.get("sim_id", traj_path.stem)
        single_summary_path = output_root / f"analysis_summary_{sim_id}.csv"
        pd.DataFrame([summary]).to_csv(single_summary_path, index=False)
        if not bool(summary.get("analysis_complete", False)):
            print(f"\nSingle-trajectory FAILED. Error summary: {single_summary_path}")
            sys.exit(2)
        print(f"\nSingle-trajectory complete. Summary: {single_summary_path}")
        return

    from pipeline.batch_runner import run_batch
    summary_path = run_batch(cfg, logger=print)
    print(f"\nBatch complete. Summary: {summary_path}")


if __name__ == "__main__":
    main()
