"""Composition field phi = (rho_A - rho_B) / (rho_A + rho_B) with masking.

phi = +1 in pure A, -1 in pure B, and 0 where local A and B densities
are equal. The nominal A/B interface is the level set phi = 0. Tanh fits
include an offset to accommodate unequal bulk composition plateaus.

Voxels whose total density falls below ``density_cutoff_fraction * mean`` are
masked invalid (phi set to 0 there for downstream FFT/interpolation stability)
and counted; in a dense DPD melt this fraction should be ~0 and a warning is
raised otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.ndimage import map_coordinates

from .density_field import DensityFields, GridSpec
from .io_utils import LogFn


@dataclass
class CompositionField:
    """phi on the analysis grid plus its construction record."""

    phi: np.ndarray             # float64, invalid voxels filled with 0.0
    invalid_mask: np.ndarray    # bool, True where denominator below cutoff
    grid: GridSpec
    stats: Dict[str, Any]
    warnings: List[str]


def composition_field(
    fields: DensityFields,
    density_cutoff_fraction: float,
    log: Optional[LogFn] = None,
) -> CompositionField:
    """Build phi from smoothed A/B densities with a denominator cutoff."""
    rho_A, rho_B = fields.rho_A, fields.rho_B
    denom = rho_A + rho_B
    mean_denom = float(denom.mean())
    cutoff = float(density_cutoff_fraction) * mean_denom
    invalid = denom < cutoff

    phi = np.zeros_like(denom)
    valid = ~invalid
    phi[valid] = (rho_A[valid] - rho_B[valid]) / denom[valid]

    frac_invalid = float(invalid.mean())
    warnings: List[str] = []
    if frac_invalid > 0.01:
        warnings.append(
            f"{100 * frac_invalid:.2f}% of voxels fall below the density cutoff "
            f"({cutoff:.4f} = {density_cutoff_fraction} * mean): the grid is too "
            "fine or the smoothing too narrow for this density."
        )

    phi_valid = phi[valid] if valid.any() else np.array([0.0])
    stats: Dict[str, Any] = {
        "phi_min": float(phi_valid.min()),
        "phi_max": float(phi_valid.max()),
        "phi_mean": float(phi_valid.mean()),
        "phi_rms": float(np.sqrt(np.mean(phi_valid ** 2))),
        "fraction_invalid_voxels": frac_invalid,
        "density_cutoff_used": cutoff,
        "density_cutoff_fraction": float(density_cutoff_fraction),
        "denominator_mean": mean_denom,
        "denominator_min": float(denom.min()),
        "denominator_max": float(denom.max()),
        "invalid_fill_value": 0.0,
    }
    if log:
        for w in warnings:
            log(f"WARN: {w}")
    return CompositionField(phi=phi, invalid_mask=invalid, grid=fields.grid,
                            stats=stats, warnings=warnings)


def sample_field_at_points(
    field: np.ndarray,
    grid: GridSpec,
    points: np.ndarray,
    order: int = 1,
) -> np.ndarray:
    """Interpolate a grid field at arbitrary physical points (periodic).

    Uses the voxel-center convention and ``mode='grid-wrap'`` (the EXACT
    periodic mode of scipy.ndimage — plain 'wrap' is not truly periodic).
    """
    idx_coords = grid.points_to_index_coords(np.asarray(points, dtype=float))
    return map_coordinates(field, idx_coords, order=order, mode="grid-wrap")
