"""Command-line interface for the interface-analysis pipeline.

Usage:
    python run_interface.py --config interface_config_v2.yaml
    python run_interface.py --config cfg.yaml --morphology gyroid
    python run_interface.py --config cfg.yaml --sims N32_n3000_pA3_a12_32_5
    python run_interface.py --config cfg.yaml --test-mode      # 1 frame, quick check
    python run_interface.py --config cfg.yaml --calibrate      # lamella only
    python run_interface.py --make-template interface_config_v2.yaml
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=("Morphology-aware interface analysis for lamella, sphere, "
                     "cylinder, and gyroid DPD melts."),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--config", "-c", default=None, help="Path to YAML config file.")
    ap.add_argument("--input-dir", default=None, help="Override config input_dir.")
    ap.add_argument("--output-dir", default=None, help="Override config output_dir.")
    ap.add_argument("--traj-suffix", default=None, help="Override traj_suffix.")
    ap.add_argument("--morphology", choices=("lamella", "sphere", "cylinder", "gyroid"),
                    default=None, help="Override the config's declared target morphology.")
    ap.add_argument("--sims", default=None,
                    help="Comma-separated sim_ids to analyze (subset of discovered files).")
    ap.add_argument("--frame-stride", type=int, default=None, help="Override frame_stride.")
    ap.add_argument("--max-frames", type=int, default=None, help="Override max_frames.")
    ap.add_argument("--last-step-window", type=int, default=None,
                    help="Analyze only the final timestep span (e.g. 100000).")
    ap.add_argument("--test-mode", action="store_true",
                    help="Analyze only the first selected frame per sim and print key numbers.")
    ap.add_argument("--calibrate", action="store_true",
                    help="Run the parameter-sensitivity calibration instead of the full analysis.")
    ap.add_argument("--make-template", default=None, metavar="PATH",
                    help="Write a documented template config YAML to PATH and exit.")
    ap.add_argument("--version", action="store_true",
                    help="Print package/dependency versions and exit.")
    return ap


def main(argv: Optional[List[str]] = None) -> None:
    ap = build_parser()
    args = ap.parse_args(argv)

    if args.version:
        from .io_utils import package_versions
        from . import __version__
        print(f"interface_analysis_v2 {__version__}")
        for k, v in package_versions().items():
            print(f"  {k:<12} {v}")
        return

    if args.make_template:
        from .config import make_template
        make_template(args.make_template)
        print(f"Template config written to: {args.make_template}")
        return

    from .config import load_config
    cfg = load_config(args.config)  # None -> defaults (validated)

    if args.input_dir:
        cfg["input_dir"] = args.input_dir
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.traj_suffix:
        cfg["traj_suffix"] = args.traj_suffix
    if args.morphology:
        cfg["morphology"] = args.morphology
    if args.sims:
        cfg["sims"] = [s.strip() for s in args.sims.split(",") if s.strip()]
    if args.frame_stride is not None:
        cfg["frame_stride"] = int(args.frame_stride)
    if args.max_frames is not None:
        cfg["max_frames"] = int(args.max_frames)
    if args.last_step_window is not None:
        cfg["last_step_window"] = int(args.last_step_window)
    if args.test_mode:
        cfg["max_frames"] = 1
        cfg["plots"] = True
        print("TEST MODE: analyzing one frame per simulation.")

    from .config import validate_config
    validate_config(cfg)

    if args.calibrate:
        if cfg["morphology"] != "lamella":
            ap.error("--calibrate currently applies only to morphology: lamella")
        from pathlib import Path
        from .batch import discover_trajectories
        from .calibration import calibrate_simulation
        trajs = discover_trajectories(cfg)
        if not trajs:
            print("ERROR: no trajectories found."); sys.exit(1)
        out = Path(cfg["output_dir"]).expanduser()
        out.mkdir(parents=True, exist_ok=True)
        for traj in trajs:
            rep = calibrate_simulation(traj, cfg, out)
            print(f"{traj.name}: {rep['verdict']}")
        return

    from .batch import run_batch
    combined = run_batch(cfg, echo=print)
    print(f"\nBatch complete. Combined table: {combined}")


if __name__ == "__main__":
    main()
