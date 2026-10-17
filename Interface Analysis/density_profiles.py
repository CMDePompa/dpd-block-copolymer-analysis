"""Chain-region-resolved one-dimensional profiles for lamellae.

The first configured chain segment contains A beads. The remaining segment
may contain fixed A and B blocks. Profiles separate A in the initial segment
from A and B in the remaining segment, using the reciprocal-lattice mode.  A
mode ``m`` defines the exactly periodic phase

    phase = (m_x f_x + m_y f_y + m_z f_z) mod 1,

where ``f`` is the particle's fractional box coordinate.  Consequently an
oblique lamella is not projected onto an arbitrary Cartesian axis, and every
equivalent lamellar repeat in the box is stacked onto the same [0, 1) period.

The result is a *global projected morphology profile*.  Waviness can broaden
its interfaces, so it must not replace the locally aligned intrinsic-width
calculation in :mod:`interface_analysis_v2.width`.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from .io_utils import LogFn
from .trajectory import FrameData

BASE_CATEGORIES = ("A_head", "A_tail", "B_tail")
DERIVED_CATEGORIES = ("A_total", "B_total", "total")
ALL_CATEGORIES = BASE_CATEGORIES + DERIVED_CATEGORIES
PROFILE_QUANTITIES = tuple(
    [f"count_{name}" for name in ALL_CATEGORIES]
    + [f"rho_{name}" for name in ALL_CATEGORIES]
    + [f"phi_{name}" for name in ALL_CATEGORIES if name != "total"]
)


@dataclass
class DensityProfileResult:
    """One phase-aligned, architecture-resolved particle profile."""

    phase: np.ndarray
    profiles: Dict[str, np.ndarray]
    shift_bins: int
    metrics: Dict[str, float]
    diagnostics: Dict[str, Any]
    warnings: List[str]


def _canonical_mode(mode: Sequence[int]) -> Tuple[int, int, int]:
    """Return a deterministic sign for a Friedel-equivalent lattice mode."""
    out = tuple(int(v) for v in mode)
    if len(out) != 3 or not any(out):
        raise ValueError(f"density profile requires a nonzero 3-D lattice mode; got {mode!r}")
    for value in out:
        if value:
            return tuple(-v for v in out) if value < 0 else out
    return out  # unreachable, retained for static type checkers


def _validate_molecule_sizes(mol_ids: np.ndarray, chain_length: int, context: str) -> None:
    molecules, counts = np.unique(mol_ids, return_counts=True)
    bad = np.flatnonzero(counts != chain_length)
    if bad.size:
        examples = [(int(molecules[i]), int(counts[i])) for i in bad[:8]]
        raise ValueError(
            f"{context}: every molecule must contain density_profile_chain_length="
            f"{chain_length} atoms; example (molecule, count) values: {examples}."
        )


def _infer_bead_indices_from_ids(
    atom_ids: np.ndarray, mol_ids: np.ndarray, chain_length: int, context: str
) -> np.ndarray:
    """Assign 1..N within each molecule after sorting by atom ID."""
    if np.unique(atom_ids).size != atom_ids.size:
        raise ValueError(f"{context}: atom IDs are not unique; bead order cannot be inferred.")
    order = np.lexsort((atom_ids, mol_ids))
    sorted_mol = mol_ids[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_mol)) + 1]
    counts = np.diff(np.r_[starts, len(order)])
    if np.any(counts != chain_length):
        # This duplicates the clearer validation above but protects direct use.
        raise ValueError(f"{context}: molecule sizes do not match chain length {chain_length}.")
    within = np.arange(len(order)) - np.repeat(starts, counts) + 1
    bead_indices = np.empty(len(order), dtype=np.int32)
    bead_indices[order] = within.astype(np.int32)
    return bead_indices


def _resolve_bead_indices(
    frame: FrameData,
    chain_length: int,
    allow_inferred_order: bool,
) -> Tuple[np.ndarray, str, List[str]]:
    context = f"timestep {frame.timestep}"
    if frame.mol_ids is None:
        raise ValueError(
            f"{context}: density-profile decomposition requires a 'mol' dump column."
        )
    mol_ids = np.asarray(frame.mol_ids, dtype=np.int64)
    _validate_molecule_sizes(mol_ids, chain_length, context)

    if frame.bead_indices is not None:
        bead = np.asarray(frame.bead_indices, dtype=np.int64)
        order = np.lexsort((bead, mol_ids))
        expected = np.tile(np.arange(1, chain_length + 1), len(bead) // chain_length)
        if not np.array_equal(bead[order], expected):
            raise ValueError(
                f"{context}: explicit bead-index values must be exactly 1..{chain_length} "
                "within every molecule."
            )
        return bead.astype(np.int32), "dump_bead_index_column", []

    if not allow_inferred_order:
        raise ValueError(
            f"{context}: no recognized bead-index dump column was found. Add a 'bead_index' "
            "column, or set density_profile_allow_inferred_bead_order: true only when atom "
            "IDs are known to increase along each chain."
        )
    if frame.atom_ids is None:
        raise ValueError(
            f"{context}: atom-ID order was requested for topology inference, but the dump "
            "has no 'id' column."
        )
    bead = _infer_bead_indices_from_ids(
        np.asarray(frame.atom_ids, dtype=np.int64), mol_ids, chain_length, context
    )
    warning = (
        "density-profile bead indices were inferred by sorting atom IDs within each "
        "molecule; this is valid only if IDs increase along bonded chain order."
    )
    return bead, "atom_id_order_within_molecule", [warning]


def _category_masks(
    frame: FrameData,
    bead_indices: np.ndarray,
    pure_a_length: int,
    a_type: int,
    b_type: int,
) -> Dict[str, np.ndarray]:
    known = (frame.types == a_type) | (frame.types == b_type)
    if not np.all(known):
        unknown = np.unique(frame.types[~known]).tolist()
        raise ValueError(
            f"timestep {frame.timestep}: density profiles expect only A type {a_type} "
            f"and B type {b_type}; found extra types {unknown}."
        )
    pure = bead_indices <= pure_a_length
    invalid_pure = pure & (frame.types != a_type)
    if np.any(invalid_pure):
        ids = (frame.atom_ids[invalid_pure][:8].tolist()
               if frame.atom_ids is not None else np.flatnonzero(invalid_pure)[:8].tolist())
        raise ValueError(
            f"timestep {frame.timestep}: pure-block positions 1..{pure_a_length} contain "
            f"non-A beads (examples {ids}); check chain lengths and bead ordering."
        )
    tail = ~pure
    masks = {
        "A_head": pure & (frame.types == a_type),
        "A_tail": tail & (frame.types == a_type),
        "B_tail": tail & (frame.types == b_type),
    }
    membership = sum(mask.astype(np.int8) for mask in masks.values())
    if not np.all(membership == 1):
        raise RuntimeError("density-profile categories do not partition all atoms exactly once")
    return masks


def _binned_profiles(
    frame: FrameData,
    masks: Mapping[str, np.ndarray],
    lattice_mode: Sequence[int],
    n_bins: int,
) -> Tuple[Dict[str, np.ndarray], Tuple[int, int, int]]:
    mode = _canonical_mode(lattice_mode)
    fractional = np.mod((frame.coords - frame.box_lo[None, :]) / frame.L[None, :], 1.0)
    phase = np.mod(fractional @ np.asarray(mode, dtype=float), 1.0)
    indices = np.minimum(np.floor(phase * n_bins).astype(np.int64), n_bins - 1)

    counts: Dict[str, np.ndarray] = {}
    for name in BASE_CATEGORIES:
        counts[name] = np.bincount(indices[masks[name]], minlength=n_bins).astype(float)
    counts["A_total"] = counts["A_head"] + counts["A_tail"]
    counts["B_total"] = counts["B_tail"].copy()
    counts["total"] = counts["A_total"] + counts["B_total"]
    if int(round(counts["total"].sum())) != frame.n_atoms:
        raise RuntimeError("density-profile bin counts do not sum to the frame atom count")

    bin_volume = float(np.prod(frame.L)) / n_bins
    profiles: Dict[str, np.ndarray] = {}
    for name in ALL_CATEGORIES:
        profiles[f"count_{name}"] = counts[name]
        profiles[f"rho_{name}"] = counts[name] / bin_volume
    denominator = counts["total"]
    for name in ALL_CATEGORIES:
        if name == "total":
            continue
        profiles[f"phi_{name}"] = np.divide(
            counts[name], denominator,
            out=np.full(n_bins, np.nan, dtype=float), where=denominator > 0,
        )
    return profiles, mode


def _circular_fill(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        raise ValueError("density profile contains no finite bins")
    if finite.all():
        return values.copy()
    n = len(values)
    good = np.flatnonzero(finite)
    return np.interp(np.arange(n), np.r_[good - n, good, good + n],
                     np.tile(values[good], 3))


def _align(profiles: Mapping[str, np.ndarray], sigma_bins: float) -> Tuple[Dict[str, np.ndarray], int]:
    guide = _circular_fill(np.asarray(profiles["phi_A_total"], dtype=float))
    smooth = gaussian_filter1d(guide, sigma=sigma_bins, mode="wrap") if sigma_bins > 0 else guide
    shift = len(guide) // 2 - int(np.argmax(smooth))
    return {name: np.roll(np.asarray(values), shift) for name, values in profiles.items()}, shift


def _periodic_crossings(values: np.ndarray, level: float) -> np.ndarray:
    y = np.asarray(values, dtype=float) - level
    positions: List[float] = []
    n = len(y)
    for i in range(n):
        j = (i + 1) % n
        if y[i] == 0:
            positions.append(float(i))
        elif y[i] * y[j] < 0:
            frac = abs(y[i]) / (abs(y[i]) + abs(y[j]))
            positions.append(float(i + frac))
    return np.asarray(positions, dtype=float)


def _profile_metrics(
    profiles: Mapping[str, np.ndarray], sigma_bins: float, interface_window_fraction: float
) -> Dict[str, float]:
    phi_a = _circular_fill(np.asarray(profiles["phi_A_total"], dtype=float))
    smooth_a = gaussian_filter1d(phi_a, sigma=sigma_bins, mode="wrap") if sigma_bins > 0 else phi_a
    lo, hi = float(np.min(smooth_a)), float(np.max(smooth_a))
    crossings = _periodic_crossings(smooth_a, 0.5 * (lo + hi))
    enrichment = float("nan")
    if crossings.size:
        tail_a = _circular_fill(np.asarray(profiles["phi_A_tail"], dtype=float))
        if sigma_bins > 0:
            tail_a = gaussian_filter1d(tail_a, sigma=sigma_bins, mode="wrap")
        centers = np.arange(len(tail_a), dtype=float) + 0.5
        distances = np.full(len(tail_a), np.inf)
        for crossing in crossings:
            d = np.abs((centers - crossing + len(tail_a) / 2) % len(tail_a)
                       - len(tail_a) / 2)
            distances = np.minimum(distances, d)
        halfwindow = max(1.0, interface_window_fraction * len(tail_a))
        near = distances <= halfwindow
        core = distances >= 3.0 * halfwindow
        if near.any() and core.any():
            enrichment = float(np.mean(tail_a[near]) - np.mean(tail_a[core]))
    return {
        "density_profile_domain_contrast": hi - lo,
        "density_profile_A_tail_interface_enrichment": enrichment,
        "density_profile_midpoint_crossings": float(len(crossings)),
    }


def compute_density_profile(
    frame: FrameData,
    lattice_mode: Sequence[int],
    cfg: Mapping[str, Any],
) -> DensityProfileResult:
    """Compute one architecture-resolved profile using the detected FFT mode."""
    chain_length = int(cfg["density_profile_chain_length"])
    pure_a_length = int(cfg["density_profile_pure_a_length"])
    n_bins = int(cfg["density_profile_bins"])
    sigma_bins = float(cfg["density_profile_alignment_sigma_bins"])
    bead, topology_source, warnings = _resolve_bead_indices(
        frame, chain_length,
        bool(cfg.get("density_profile_allow_inferred_bead_order", False)),
    )
    masks = _category_masks(
        frame, bead, pure_a_length,
        int(cfg["A_type_id"]), int(cfg["B_type_id"]),
    )
    profiles, mode = _binned_profiles(frame, masks, lattice_mode, n_bins)
    aligned, shift = _align(profiles, sigma_bins)
    metrics = _profile_metrics(
        aligned, sigma_bins, float(cfg["density_profile_interface_window_fraction"])
    )
    divisor = gcd(gcd(abs(mode[0]), abs(mode[1])), abs(mode[2]))
    diagnostics = {
        "method": "reciprocal_lattice_phase_stacking",
        "lattice_mode_canonical": list(mode),
        "lamellar_repeats_stacked": int(divisor),
        "n_bins": n_bins,
        "chain_length": chain_length,
        "pure_a_length": pure_a_length,
        "topology_source": topology_source,
        "alignment_shift_bins": int(shift),
        "coordinate_note": (
            "phase=(m dot fractional_coordinate) mod 1; A-rich maximum shifted to phase 0.5"
        ),
        "interpretation_note": (
            "global projected profile; waviness broadens transitions, so use the local width "
            "analysis for intrinsic interface width"
        ),
    }
    phase = (np.arange(n_bins, dtype=float) + 0.5) / n_bins
    return DensityProfileResult(
        phase=phase, profiles=aligned, shift_bins=shift, metrics=metrics,
        diagnostics=diagnostics, warnings=warnings,
    )


def aggregate_density_profiles(
    results: Sequence[DensityProfileResult], timesteps: Sequence[int]
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """Return a mean/std CSV table and full per-frame arrays for NPZ output."""
    if not results:
        raise ValueError("cannot aggregate zero density profiles")
    n_bins = len(results[0].phase)
    if any(len(r.phase) != n_bins for r in results):
        raise ValueError("density-profile bin count changed between frames")
    phase = results[0].phase.copy()
    table: Dict[str, np.ndarray] = {
        "z_scaled": phase,
        "position_over_L0": phase - 0.5,
    }
    arrays: Dict[str, np.ndarray] = {
        "z_scaled": phase,
        "position_over_L0": phase - 0.5,
        "timesteps": np.asarray(timesteps, dtype=np.int64),
    }
    for name in PROFILE_QUANTITIES:
        stack = np.stack([r.profiles[name] for r in results])
        finite = np.isfinite(stack)
        count = finite.sum(axis=0)
        total = np.nansum(stack, axis=0)
        mean = np.divide(total, count, out=np.full(n_bins, np.nan), where=count > 0)
        squared = np.nansum((stack - mean[None, :]) ** 2, axis=0)
        std = np.sqrt(np.divide(squared, count, out=np.full(n_bins, np.nan),
                                where=count > 0))
        arrays[name] = stack
        arrays[f"{name}_mean"] = mean
        arrays[f"{name}_std"] = std
        table[f"{name}_mean"] = arrays[f"{name}_mean"]
        table[f"{name}_std"] = arrays[f"{name}_std"]
    table["n_frames"] = np.full(n_bins, len(results), dtype=int)
    return pd.DataFrame(table), arrays


def plot_density_profile_summary(
    sim_id: str,
    plot_dir: Path,
    profile_df: pd.DataFrame,
    log: Optional[LogFn] = None,
) -> None:
    """Plot time-averaged category fractions and number densities."""
    plot_dir = Path(plot_dir)
    x = profile_df["position_over_L0"].to_numpy(float)

    def draw(path: Path, keys: Sequence[Tuple[str, str, str, str]], ylabel: str,
             title: str, ylim: Optional[Tuple[float, float]] = None) -> None:
        fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
        for key, label, color, style in keys:
            y = profile_df[f"{key}_mean"].to_numpy(float)
            sd = profile_df[f"{key}_std"].to_numpy(float)
            ax.plot(x, y, style, color=color, lw=2.0, label=label)
            ax.fill_between(x, y - sd, y + sd, color=color, alpha=0.15, linewidth=0)
        ax.set(xlabel=r"position within stacked lamellar period, $s/L_0$",
               ylabel=ylabel, xlim=(-0.5, 0.5))
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.axvline(0.0, color="0.65", lw=0.7, ls=":")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, fontsize=9, ncol=2)
        ax.set_title(title, fontsize=10)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        if log:
            log(f"plot: {path.name}")

    draw(
        plot_dir / f"{sim_id}_block_composition_profile.png",
        (("phi_A_head", r"$A_{head}$", "#1f77b4", "-"),
         ("phi_A_tail", r"$A_{tail}$", "#ff7f0e", "-"),
         ("phi_B_tail", r"$B_{tail}$", "#2ca02c", "-"),
         ("phi_A_total", r"$A_{total}$", "#7f3c8d", "--")),
        "local bead fraction",
        f"{sim_id}: architecture-resolved lamellar composition",
        (0.0, 1.0),
    )
    draw(
        plot_dir / f"{sim_id}_block_density_profile.png",
        (("rho_A_head", r"$\rho_{A,head}$", "#1f77b4", "-"),
         ("rho_A_tail", r"$\rho_{A,tail}$", "#ff7f0e", "-"),
         ("rho_B_tail", r"$\rho_{B,tail}$", "#2ca02c", "-"),
         ("rho_total", r"$\rho_{total}$", "#222222", "--")),
        "number density",
        f"{sim_id}: architecture-resolved lamellar number density",
    )
