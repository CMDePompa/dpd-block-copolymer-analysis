"""Batch orchestration: discover trajectories, analyze each, build comparisons.

Mirrors the S(q) pipeline's batch layout: per-sim folders under output_dir,
a combined summary CSV, and the run's config saved with a timestamp. Adds a
cross-system comparison table + figure over the headline interface metrics.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from .config import parse_sim_id, save_config
from .io_utils import LogFn, write_json
from .morphology_pipeline import analyze_simulation_morphology_interfaces
from .pipeline import analyze_simulation_interfaces as analyze_lamellar_interfaces

# headline metrics for the comparison table/figure: (per-frame key, label)
_COMPARE_LAMELLA = [
    ("waviness_rms", "RMS waviness [$r_c$]"),
    ("width_lam_corrected_mean", r"width $\lambda$ [$r_c$]"),
    ("width_10_90", "10-90% width [$r_c$]"),
    ("spacing_mean", "spacing [$r_c$]"),
    ("spacing_cv", "spacing CV"),
    ("excess_area_mean", "excess area"),
    ("S_n", "$S_n$"),
    ("bad_column_fraction", "bad-column fraction"),
    ("f_mix_voxel_0.5", "mixed volume (|phi|<0.5)"),
    ("P_A_to_B", r"$P_{A\to B}$"),
    ("P_B_to_A", r"$P_{B\to A}$"),
    ("overlap_O", "overlap O"),
    ("topo_A0_connection_score", "topology connection score (A)"),
    ("slope_rms", "RMS slope"),
]

_COMPARE_GENERAL = [
    ("surface_area_per_volume", r"interface area / volume [$r_c^{-1}$]"),
    ("surface_H_abs_mean", r"mean $|H|$ [$r_c^{-1}$]"),
    ("surface_H_rms", r"RMS $H$ [$r_c^{-1}$]"),
    ("surface_K_negative_area_fraction", "negative-K area fraction"),
    ("normal_width_lam_mean", r"intrinsic width $\lambda$ [$r_c$]"),
    ("normal_width_10_90", r"intrinsic 10-90% width [$r_c$]"),
    ("f_mix_voxel_0.5", "mixed volume (|phi|<0.5)"),
    ("overlap_O", "density overlap O"),
    ("domain_component_count", "minority-domain count"),
    ("domain_wrapping_component_count", "wrapping-domain count"),
    ("domain_interface_level_chi", "minority-domain Euler chi"),
    ("surface_normal_anisotropy", "surface-normal anisotropy"),
]


def discover_trajectories(cfg: Dict[str, Any]) -> List[Path]:
    input_dir = Path(cfg["input_dir"]).expanduser()
    suffix = cfg["traj_suffix"]
    trajs = sorted(p for p in input_dir.iterdir()
                   if p.is_file() and p.name.endswith(suffix))
    sims = cfg.get("sims")
    if sims:
        wanted = {s if not s.endswith(suffix) else s[: -len(suffix)] for s in sims}
        trajs = [p for p in trajs if parse_sim_id(p.name, suffix) in wanted]
    return trajs


def run_batch(cfg: Dict[str, Any], echo: Optional[LogFn] = print) -> Path:
    """Analyze every discovered trajectory; write combined outputs."""
    output_root = Path(cfg["output_dir"]).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)

    run_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_config(cfg, output_root / f"interface_v2_run_config_{run_ts}.yaml")

    trajs = discover_trajectories(cfg)
    if not trajs:
        raise FileNotFoundError(
            f"No files ending in '{cfg['traj_suffix']}' found in {cfg['input_dir']}"
            + (f" matching sims={cfg['sims']}" if cfg.get("sims") else "")
            + ". Check input_dir/traj_suffix/sims in your config.")
    if echo:
        echo(f"Found {len(trajs)} trajectory file(s).")

    summaries: List[Dict[str, Any]] = []
    for i, traj in enumerate(trajs, 1):
        if echo:
            echo(f"\n--- [{i}/{len(trajs)}] {traj.name} ---")
        morphology = str(cfg.get("morphology", "lamella")).lower()
        if morphology == "lamella":
            analyze = analyze_lamellar_interfaces
        else:
            analyze = analyze_simulation_morphology_interfaces
        summaries.append(analyze(traj, cfg, output_root, echo=echo))

    df = pd.DataFrame(summaries)
    combined = output_root / "interface_comparison.csv"
    df.to_csv(combined, index=False)
    if echo:
        echo(f"\nWrote combined comparison table: {combined}")

    try:
        _comparison_outputs(df, output_root, echo)
    except Exception as exc:
        if echo:
            echo(f"WARN: comparison figure failed: {exc}")
    return combined


def _comparison_outputs(df: pd.DataFrame, output_root: Path, echo: Optional[LogFn]) -> None:
    """Compact cross-system comparison table + figure (mean +/- std over frames)."""
    ok = df[df.get("status", "ok") == "ok"] if "status" in df else df
    if len(ok) == 0:
        return
    morphologies = sorted({str(x) for x in ok.get("morphology", pd.Series(["lamella"]))})
    metrics = _COMPARE_LAMELLA if morphologies == ["lamella"] else _COMPARE_GENERAL
    rows = []
    for _, r in ok.iterrows():
        row = {"sim_id": r["sim_id"], "pA": r.get("pA"), "a12": r.get("a12"),
               "morphology": r.get("morphology", "lamella"),
               "n_frames": r.get("n_frames_analyzed")}
        for key, label in metrics:
            row[key] = r.get(f"{key}_mean")
            row[f"{key}_std"] = r.get(f"{key}_std")
        rows.append(row)
    comp = pd.DataFrame(rows)
    comp.to_csv(output_root / "interface_comparison_headline.csv", index=False)
    write_json({"metrics": [k for k, _ in metrics],
                "note": "values are means over analyzed frames; std over frames "
                        "(frames may be correlated — see *_n_eff in per-sim summaries)"},
               output_root / "interface_comparison_headline.meta.json")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(metrics)
    ncol = 4
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 2.9 * nrow),
                             constrained_layout=True, squeeze=False)
    x = np.arange(len(comp))
    short = [s.replace("_pure_diblock", "\npure_diblock").replace("N32_n3000_", "")
             for s in comp["sim_id"]]
    for i, (key, label) in enumerate(metrics):
        ax = axes[i // ncol][i % ncol]
        y = comp[key].to_numpy(dtype=float)
        yerr = comp[f"{key}_std"].to_numpy(dtype=float)
        ax.errorbar(x, y, yerr=yerr, fmt="o", capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels(short, fontsize=7, rotation=20)
        ax.set_title(label, fontsize=9)
    for i in range(n, nrow * ncol):
        axes[i // ncol][i % ncol].axis("off")
    fig.suptitle(
        "Interface metrics across systems (mean +/- std over frames)\n"
        f"configured morphology: {', '.join(morphologies)}"
    )
    fig.savefig(output_root / "interface_comparison_summary.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    if echo:
        echo("Wrote interface_comparison_headline.csv and interface_comparison_summary.png")
