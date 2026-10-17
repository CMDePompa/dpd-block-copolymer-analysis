"""Periodic 3D number-density fields from bead positions.

Method: cloud-in-cell (CIC) deposition onto a regular periodic grid followed
by periodic Gaussian smoothing (``scipy.ndimage.gaussian_filter(mode="wrap")``).

Why CIC + scipy rather than ``freud.density.GaussianDensity``:
  * deposition + filtering is exactly mass conserving (verified in tests);
  * the smoothing width is decoupled from deposition, so sensitivity scans
    vary one thing at a time;
  * every step is transparent numpy/scipy — no library API pitfalls;
  * freud's GaussianDensity is retained as an independent cross-check in the
    test suite (when freud is installed), not in the production path.

Conventions:
  * grid values live at voxel CENTERS: c_i = lo + (i + 0.5) * h;
  * the deposited array is COUNTS (sum equals particle number); number
    density is counts / voxel_volume;
  * Gaussian smoothing preserves the sum (periodic wrap), so densities remain
    normalized after smoothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter

from .io_utils import LogFn
from .trajectory import FrameData

_GAUSS_TRUNCATE = 4.0  # gaussian_filter kernel extent in sigmas


@dataclass
class GridSpec:
    """Geometry of a periodic analysis grid."""

    box_lo: np.ndarray          # (3,)
    box_hi: np.ndarray          # (3,)
    shape: Tuple[int, int, int]
    voxel: np.ndarray           # (3,) physical voxel edge lengths

    @property
    def L(self) -> np.ndarray:
        return self.box_hi - self.box_lo

    @property
    def voxel_volume(self) -> float:
        return float(np.prod(self.voxel))

    def axis_centers(self, axis: int) -> np.ndarray:
        n = self.shape[axis]
        return self.box_lo[axis] + (np.arange(n) + 0.5) * self.voxel[axis]

    def points_to_index_coords(self, points: np.ndarray) -> np.ndarray:
        """Map physical points to fractional INDEX coordinates (center convention).

        Suitable for ``scipy.ndimage.map_coordinates(..., mode='grid-wrap')``.
        Returns shape (3, N).
        """
        u = (points - self.box_lo[None, :]) / self.voxel[None, :] - 0.5
        return u.T.copy()


def resolve_grid(
    box_lo: np.ndarray,
    box_hi: np.ndarray,
    target_voxel_size: float,
    grid_shape_override: Optional[List[int]] = None,
) -> Tuple[GridSpec, Dict[str, Any]]:
    """Choose grid dimensions from the box and target voxel size.

    n_i = max(8, round(L_i / target_voxel_size)); the ACTUAL voxel size per
    axis (L_i / n_i) is recorded — it can differ slightly from the target.
    """
    L = np.asarray(box_hi, dtype=float) - np.asarray(box_lo, dtype=float)
    notes: Dict[str, Any] = {"target_voxel_size": float(target_voxel_size)}
    if grid_shape_override is not None:
        shape = tuple(int(n) for n in grid_shape_override)
        notes["rule"] = "explicit grid_shape from config"
    else:
        shape = tuple(max(8, int(round(Li / float(target_voxel_size)))) for Li in L)
        notes["rule"] = "n_i = max(8, round(L_i / target_voxel_size))"
    voxel = L / np.asarray(shape, dtype=float)
    grid = GridSpec(box_lo=np.asarray(box_lo, float).copy(),
                    box_hi=np.asarray(box_hi, float).copy(),
                    shape=shape, voxel=voxel)
    notes["grid_shape"] = list(shape)
    notes["actual_voxel_size"] = voxel.tolist()
    notes["voxel_volume"] = grid.voxel_volume
    return grid, notes


# ---------------------------------------------------------------------------
# Deposition
# ---------------------------------------------------------------------------
def deposit_counts(coords: np.ndarray, grid: GridSpec, method: str = "cic") -> np.ndarray:
    """Deposit particles onto the periodic grid; returns float64 COUNTS.

    'cic': trilinear cloud-in-cell (each particle shares weight over the 8
    nearest voxel centers, periodic wrap). 'ngp': nearest grid point.
    The result satisfies counts.sum() == len(coords) exactly (to float64
    accumulation accuracy).
    """
    n = np.asarray(grid.shape, dtype=np.int64)
    counts = np.zeros(grid.shape, dtype=np.float64)
    if len(coords) == 0:
        return counts
    u = (coords - grid.box_lo[None, :]) / grid.voxel[None, :]  # in [0, n) for folded coords

    if method == "ngp":
        idx = np.floor(u).astype(np.int64) % n[None, :]
        np.add.at(counts, (idx[:, 0], idx[:, 1], idx[:, 2]), 1.0)
        return counts

    if method != "cic":
        raise ValueError(f"Unknown deposition method: {method!r}")

    # CIC with voxel-center convention: weights split between centers i0, i0+1
    uc = u - 0.5
    i0 = np.floor(uc).astype(np.int64)
    frac = uc - i0                       # in [0, 1)
    w_lo = 1.0 - frac
    w_hi = frac
    idx_lo = i0 % n[None, :]
    idx_hi = (i0 + 1) % n[None, :]
    ix = (idx_lo[:, 0], idx_hi[:, 0])
    iy = (idx_lo[:, 1], idx_hi[:, 1])
    iz = (idx_lo[:, 2], idx_hi[:, 2])
    wx = (w_lo[:, 0], w_hi[:, 0])
    wy = (w_lo[:, 1], w_hi[:, 1])
    wz = (w_lo[:, 2], w_hi[:, 2])
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                np.add.at(counts, (ix[a], iy[b], iz[c]), wx[a] * wy[b] * wz[c])
    return counts


def smooth_field(field: np.ndarray, sigma_physical: float, grid: GridSpec) -> np.ndarray:
    """Periodic Gaussian smoothing with sigma given in PHYSICAL units."""
    sigma_vox = [float(sigma_physical) / float(h) for h in grid.voxel]
    return gaussian_filter(field, sigma=sigma_vox, mode="wrap", truncate=_GAUSS_TRUNCATE)


# ---------------------------------------------------------------------------
# Full A/B field construction with diagnostics
# ---------------------------------------------------------------------------
@dataclass
class DensityFields:
    """Smoothed A and B number-density fields plus construction diagnostics."""

    rho_A: np.ndarray
    rho_B: np.ndarray
    grid: GridSpec
    diagnostics: Dict[str, Any]
    warnings: List[str]


def build_density_fields(
    frame: FrameData,
    A_type_id: int,
    B_type_id: int,
    target_voxel_size: float,
    smoothing_sigma: float,
    grid_shape_override: Optional[List[int]] = None,
    deposition: str = "cic",
    log: Optional[LogFn] = None,
) -> DensityFields:
    """Construct smoothed rho_A(r), rho_B(r) for one frame.

    Emits resolution-adequacy warnings (returned AND logged):
      * smoothing narrower than the grid can support;
      * voxels too coarse for the requested smoothing (aliasing);
      * too few beads per smoothing volume (noisy phi);
    These implement the "diagnose inadequate grid/smoothing" requirement —
    parameters are never judged by visual appearance.
    """
    grid, grid_notes = resolve_grid(frame.box_lo, frame.box_hi,
                                    target_voxel_size, grid_shape_override)
    pos_A = frame.positions_of_type(A_type_id)
    pos_B = frame.positions_of_type(B_type_id)
    n_A, n_B = len(pos_A), len(pos_B)

    counts_A = deposit_counts(pos_A, grid, method=deposition)
    counts_B = deposit_counts(pos_B, grid, method=deposition)
    mass_err_A = abs(counts_A.sum() - n_A) / max(n_A, 1)
    mass_err_B = abs(counts_B.sum() - n_B) / max(n_B, 1)

    smooth_A = smooth_field(counts_A, smoothing_sigma, grid)
    smooth_B = smooth_field(counts_B, smoothing_sigma, grid)
    rho_A = smooth_A / grid.voxel_volume
    rho_B = smooth_B / grid.voxel_volume

    rho_tot = rho_A + rho_B
    mean_tot = float(rho_tot.mean())
    warnings: List[str] = []

    h_max = float(np.max(grid.voxel))
    sigma = float(smoothing_sigma)
    if sigma < 0.75 * h_max:
        warnings.append(
            f"smoothing_sigma={sigma:.3f} < 0.75*voxel ({h_max:.3f}): deposition "
            "graininess is not smoothed out; increase smoothing_sigma or refine the grid."
        )
    if h_max > 2.0 * sigma:
        warnings.append(
            f"voxel size {h_max:.3f} > 2*smoothing_sigma ({sigma:.3f}): the Gaussian "
            "kernel is under-sampled (aliasing risk); refine the grid."
        )
    # Effective independent-sample volume of a 3D Gaussian kernel per axis is
    # (integral of G)^2 / integral of G^2 = 2*sigma*sqrt(pi).
    v_eff = (2.0 * sigma * np.sqrt(np.pi)) ** 3
    n_eff = mean_tot * v_eff
    if n_eff < 20:
        warnings.append(
            f"only ~{n_eff:.1f} beads per smoothing volume (need >~20): phi noise "
            f"~{1.0 / max(np.sqrt(n_eff), 1e-9):.2f} will produce spurious zero "
            "crossings; increase smoothing_sigma."
        )
    min_tot = float(rho_tot.min())
    if min_tot < 0.2 * mean_tot:
        warnings.append(
            f"min total density {min_tot:.3f} < 20% of mean ({mean_tot:.3f}): "
            "voids or severe density holes present; phi is unreliable there "
            "(masked by density_cutoff_fraction)."
        )

    diagnostics: Dict[str, Any] = {
        **grid_notes,
        "deposition": deposition,
        "smoothing_sigma_physical": sigma,
        "smoothing_sigma_voxels": [sigma / float(h) for h in grid.voxel],
        "gaussian_truncate_sigmas": _GAUSS_TRUNCATE,
        "n_A_beads": n_A,
        "n_B_beads": n_B,
        "fraction_A": n_A / max(n_A + n_B, 1),
        "mass_conservation_rel_err_A": mass_err_A,
        "mass_conservation_rel_err_B": mass_err_B,
        "mean_total_density": mean_tot,
        "min_total_density": min_tot,
        "max_total_density": float(rho_tot.max()),
        "beads_per_smoothing_volume": float(n_eff),
        "normalization": "counts/voxel_volume; periodic Gaussian smoothing preserves totals",
    }

    if log:
        for w in warnings:
            log(f"WARN: {w}")
    return DensityFields(rho_A=rho_A, rho_B=rho_B, grid=grid,
                         diagnostics=diagnostics, warnings=warnings)
