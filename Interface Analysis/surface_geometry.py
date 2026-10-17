"""Interfacial area, excess area, local normals, and orientational order.

Two independent area estimates:
  * height-field excess area per interface, A_j = sum sqrt(1 + h_u^2 + h_v^2)
    du dv, with excess eps_A = A_j / A_flat - 1 (A_flat = lateral window
    area). This is single-valued-surface geometry and is the primary metric.
  * a whole-sample marching-cubes area of the phi = 0 isosurface as an
    INDEPENDENT cross-check. Marching cubes is not periodic; we pad the grid
    by one voxel with wrap so boundary-crossing surfaces close, and label the
    estimate accordingly.

Local normals use the phase-field gradient on the periodic 3-D grid,
    n(r) = grad(phi) / |grad(phi)|,
evaluated in the interfacial band (|phi| < band) and weighted by |grad(phi)|
so bulk voxels (grad ~ 0, normal undefined) do not contribute. The
orientational order parameter is

    S_n = < (3 cos^2 theta - 1) / 2 >,  cos theta = |n . n0|,

with mean/spread of theta and the fraction of interfacial area above chosen
misorientation thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .composition import CompositionField
from .height_fields import HeightMap, gradient_valid_mask


# ---------------------------------------------------------------------------
# Excess area from height fields
# ---------------------------------------------------------------------------
@dataclass
class AreaResult:
    per_interface_area: List[float]
    per_interface_excess: List[float]
    total_height_field_area: float       # sampled oblique windows, not always whole-box area
    flat_area: float
    mean_excess_area: float
    marching_cubes_area: Optional[float]
    marching_cubes_note: str
    area_per_volume: Optional[float]
    warnings: List[str] = field(default_factory=list)


def height_field_area(hmap: HeightMap, max_filled_fraction: float = 0.20) -> tuple:
    """Return (area, excess) of one interface from its height-field slopes."""
    if hmap.filled_fraction > max_filled_fraction:
        return float("nan"), float("nan")
    du, dv = hmap.du, hmap.dv
    hu, hv = np.gradient(hmap.h_filled, du, dv, edge_order=1)
    dA = np.sqrt(1.0 + hu ** 2 + hv ** 2)
    # Count only cells whose derivative stencil is entirely measured.
    mask = gradient_valid_mask(hmap.valid)
    if mask.sum() < 4:
        return float("nan"), float("nan")
    area = float(np.sum(dA[mask]) * du * dv)
    flat = float(mask.sum() * du * dv)
    excess = area / flat - 1.0 if flat > 0 else float("nan")
    return area, excess


def compute_area(
    hmaps: List[HeightMap],
    comp: Optional[CompositionField] = None,
    box_volume: Optional[float] = None,
    marching_cubes: bool = True,
    max_filled_fraction: float = 0.20,
) -> AreaResult:
    """Excess area per interface + optional marching-cubes cross-check."""
    per_area: List[float] = []
    per_excess: List[float] = []
    flat_total = 0.0
    for hm in hmaps:
        a, e = height_field_area(hm, max_filled_fraction=max_filled_fraction)
        per_area.append(a)
        per_excess.append(e)
        if np.isfinite(a):
            flat_total += float(gradient_valid_mask(hm.valid).sum() * hm.du * hm.dv)
    finite_area = [a for a in per_area if np.isfinite(a)]
    finite_excess = [e for e in per_excess if np.isfinite(e)]
    total_hf_area = float(np.sum(finite_area)) if finite_area else float("nan")
    mean_excess = float(np.mean(finite_excess)) if finite_excess else float("nan")

    mc_area: Optional[float] = None
    mc_note = "not computed"
    warnings: List[str] = []
    if marching_cubes and comp is not None:
        mc_area, mc_note = _marching_cubes_area(comp)

    area_per_volume = None
    if mc_area is not None and box_volume:
        area_per_volume = mc_area / box_volume

    return AreaResult(
        per_interface_area=per_area, per_interface_excess=per_excess,
        total_height_field_area=total_hf_area, flat_area=flat_total,
        mean_excess_area=mean_excess, marching_cubes_area=mc_area,
        marching_cubes_note=mc_note, area_per_volume=area_per_volume, warnings=warnings,
    )


def _marching_cubes_area(comp: CompositionField):
    """Total area of the phi=0 isosurface via marching cubes (periodic-padded)."""
    if np.any(comp.invalid_mask):
        return None, ("marching-cubes area skipped: invalid-density voxels are filled "
                      "with phi=0 and would create artificial isosurfaces")
    try:
        from skimage.measure import marching_cubes, mesh_surface_area
    except ImportError:
        return None, "scikit-image not installed; marching-cubes crosscheck skipped"
    phi = comp.phi
    voxel = comp.grid.voxel
    # Pad by one voxel with wrap so boundary-crossing surfaces close periodically.
    padded = np.pad(phi, 1, mode="wrap")
    try:
        verts, faces, _, _ = marching_cubes(padded, level=0.0, spacing=tuple(voxel))
        area = float(mesh_surface_area(verts, faces))
    except (ValueError, RuntimeError) as exc:
        return None, f"marching_cubes failed: {exc}"
    # The wrap pad duplicates one voxel layer on each side; the extracted area
    # therefore slightly over-counts near the seams. Report as an independent
    # order-of-magnitude crosscheck, not an exact value.
    return area, ("phi=0 isosurface, wrap-padded 1 voxel; approximate near periodic "
                  "seams (independent crosscheck, not exact)")


# ---------------------------------------------------------------------------
# Local normals and orientational order
# ---------------------------------------------------------------------------
@dataclass
class OrientationOrderResult:
    S_n: float
    mean_angle_deg: float
    std_angle_deg: float
    angle_hist_centers: np.ndarray
    angle_hist: np.ndarray
    fraction_above: Dict[str, float]     # fraction of interfacial area with theta > threshold
    misorientation_map: np.ndarray       # 2-D projected mean-angle map (deg)
    n_band_voxels: int
    band: float
    diagnostics: Dict[str, Any]


def _periodic_gradient(field: np.ndarray, voxel: np.ndarray):
    """Exact periodic central-difference gradient (mode='wrap')."""
    g = []
    for ax in range(3):
        fwd = np.roll(field, -1, axis=ax)
        bwd = np.roll(field, 1, axis=ax)
        g.append((fwd - bwd) / (2.0 * voxel[ax]))
    return g


def compute_orientational_order(
    comp: CompositionField,
    n0: np.ndarray,
    band: float = 0.5,
    misorientation_thresholds_deg: Optional[List[float]] = None,
) -> OrientationOrderResult:
    """Local-normal orientational order parameter S_n from grad(phi)."""
    if misorientation_thresholds_deg is None:
        misorientation_thresholds_deg = [10.0, 20.0, 30.0, 45.0]
    phi = comp.phi
    voxel = comp.grid.voxel
    n0 = np.asarray(n0, float)
    n0 = n0 / np.linalg.norm(n0)

    gx, gy, gz = _periodic_gradient(phi, voxel)
    gmag = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)

    interface_band = (np.abs(phi) < band) & (gmag > 1e-8) & (~comp.invalid_mask)
    n_band = int(interface_band.sum())
    if n_band < 8:
        return _empty_orientation_order(band)

    nx = gx[interface_band] / gmag[interface_band]
    ny = gy[interface_band] / gmag[interface_band]
    nz = gz[interface_band] / gmag[interface_band]
    weights = gmag[interface_band]

    cos_theta = np.abs(nx * n0[0] + ny * n0[1] + nz * n0[2])
    cos_theta = np.clip(cos_theta, 0.0, 1.0)
    theta_deg = np.degrees(np.arccos(cos_theta))

    wsum = float(weights.sum())
    S_n = float(np.sum(weights * (3.0 * cos_theta ** 2 - 1.0) / 2.0) / wsum)
    mean_angle = float(np.sum(weights * theta_deg) / wsum)
    var_angle = float(np.sum(weights * (theta_deg - mean_angle) ** 2) / wsum)
    std_angle = float(np.sqrt(max(var_angle, 0.0)))

    hist, edges = np.histogram(theta_deg, bins=45, range=(0, 90), weights=weights, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    fraction_above = {
        f"{t:g}deg": float(np.sum(weights[theta_deg > t]) / wsum)
        for t in misorientation_thresholds_deg
    }

    # Spatial misorientation map: gradient-weighted mean angle projected along
    # the ORIGINAL axis most parallel to n0 (a rough but useful 2-D view).
    proj_axis = int(np.argmax(np.abs(n0)))
    ang_field = np.full(phi.shape, np.nan)
    ang_field[interface_band] = theta_deg
    w_field = np.zeros(phi.shape)
    w_field[interface_band] = weights
    num = np.nansum(np.where(np.isnan(ang_field), 0.0, ang_field) * w_field, axis=proj_axis)
    den = np.sum(w_field, axis=proj_axis)
    miso_map = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)

    diagnostics = {
        "band": band,
        "n_band_voxels": n_band,
        "projection_axis": proj_axis,
        "weighting": "|grad(phi)| (interface-localized)",
        "note": "S_n=1 perfectly aligned lamellae; 0 isotropic; grad from periodic differences",
    }
    return OrientationOrderResult(
        S_n=S_n, mean_angle_deg=mean_angle, std_angle_deg=std_angle,
        angle_hist_centers=centers, angle_hist=hist, fraction_above=fraction_above,
        misorientation_map=miso_map, n_band_voxels=n_band, band=band, diagnostics=diagnostics,
    )


def _empty_orientation_order(band) -> OrientationOrderResult:
    nan = float("nan")
    return OrientationOrderResult(
        S_n=nan, mean_angle_deg=nan, std_angle_deg=nan,
        angle_hist_centers=np.zeros(0), angle_hist=np.zeros(0),
        fraction_above={}, misorientation_map=np.zeros((0, 0)),
        n_band_voxels=0, band=band,
        diagnostics={"note": "too few interfacial-band voxels for orientation order"},
    )
