"""Periodic instantaneous dividing surface and curvature geometry.

The A/B interface is the isosurface ``phi = interface_level`` of the same
coarse-grained composition field used by the lamellar pipeline.  A single
positive-side periodic ghost layer supplies the cells that cross each box
boundary; this covers one fundamental periodic cell without duplicating the
whole surface.

Curvatures are evaluated from derivatives of the smooth level-set field and
sampled at triangle centroids.  With the normal pointing toward increasing
phi (the A-rich side),

    H = (|g|^2 tr(D2 phi) - g^T D2(phi) g) / (2 |g|^3)
    K = g^T cof(D2 phi) g / |g|^4

where g = grad(phi).  These definitions recover |H|=1/R, K=1/R^2 for a
sphere and |H|=1/(2R), K=0 for a cylinder.  The sign of H depends on which
phase is inside; K is orientation independent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
from scipy.ndimage import map_coordinates

from .composition import CompositionField


@dataclass
class SurfaceGeometry:
    vertices: np.ndarray
    faces: np.ndarray
    centroids: np.ndarray
    normals_to_A: np.ndarray
    face_areas: np.ndarray
    mean_curvature: np.ndarray
    gaussian_curvature: np.ndarray
    area: float
    area_per_volume: float
    H_mean: float
    H_abs_mean: float
    H_rms: float
    K_mean: float
    K_negative_area_fraction: float
    K_integral: float
    gauss_bonnet_solid_chi_estimate: float
    normal_tensor: np.ndarray
    normal_tensor_eigenvalues: np.ndarray
    normal_tensor_eigenvectors: np.ndarray
    normal_anisotropy: float
    diagnostics: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)


def _periodic_derivative(a: np.ndarray, axis: int, spacing: float) -> np.ndarray:
    return (np.roll(a, -1, axis=axis) - np.roll(a, 1, axis=axis)) / (2.0 * spacing)


def level_set_curvatures(phi: np.ndarray, voxel: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (H, K, |grad phi|) on a periodic orthorhombic grid."""
    h = np.asarray(voxel, dtype=float)
    g = [_periodic_derivative(phi, i, h[i]) for i in range(3)]
    hess = np.empty((3, 3) + phi.shape, dtype=float)
    for i in range(3):
        for j in range(i, 3):
            dij = _periodic_derivative(g[i], j, h[j])
            if i != j:
                dij = 0.5 * (dij + _periodic_derivative(g[j], i, h[i]))
            hess[i, j] = dij
            hess[j, i] = dij

    g2 = sum(gi * gi for gi in g)
    gmag = np.sqrt(g2)
    tr = hess[0, 0] + hess[1, 1] + hess[2, 2]
    ghg = np.zeros_like(phi)
    for i in range(3):
        for j in range(3):
            ghg += g[i] * hess[i, j] * g[j]
    eps = max(float(np.nanmax(gmag)) * 1e-10, 1e-14)
    valid = gmag > eps
    H = np.full_like(phi, np.nan, dtype=float)
    H[valid] = (g2[valid] * tr[valid] - ghg[valid]) / (2.0 * gmag[valid] ** 3)

    a, b, c = hess[0, 0], hess[0, 1], hess[0, 2]
    d, e = hess[1, 1], hess[1, 2]
    f = hess[2, 2]
    cof00 = d * f - e * e
    cof01 = c * e - b * f
    cof02 = b * e - c * d
    cof11 = a * f - c * c
    cof12 = b * c - a * e
    cof22 = a * d - b * b
    gcg = (g[0] * (cof00 * g[0] + cof01 * g[1] + cof02 * g[2])
           + g[1] * (cof01 * g[0] + cof11 * g[1] + cof12 * g[2])
           + g[2] * (cof02 * g[0] + cof12 * g[1] + cof22 * g[2]))
    K = np.full_like(phi, np.nan, dtype=float)
    K[valid] = gcg[valid] / (g2[valid] ** 2)
    return H, K, gmag


def _sample_periodic(field: np.ndarray, comp: CompositionField, points: np.ndarray) -> np.ndarray:
    grid = comp.grid
    wrapped = grid.box_lo + np.mod(points - grid.box_lo, grid.L)
    coords = grid.points_to_index_coords(wrapped)
    return map_coordinates(field, coords, order=1, mode="grid-wrap")


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    good = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(good):
        return float("nan")
    return float(np.sum(values[good] * weights[good]) / np.sum(weights[good]))


def extract_surface_geometry(comp: CompositionField, level: float = 0.0) -> SurfaceGeometry:
    """Extract one periodic phi-level surface and its area/curvature metrics."""
    try:
        from skimage.measure import marching_cubes
    except ImportError as exc:
        raise RuntimeError(
            "scikit-image is required for morphology-aware surface analysis"
        ) from exc

    valid_phi = comp.phi[~comp.invalid_mask]
    if valid_phi.size == 0 or not (float(np.min(valid_phi)) < level < float(np.max(valid_phi))):
        raise ValueError(
            f"interface level {level:g} is not bracketed by the valid composition field "
            f"[{float(np.min(valid_phi)) if valid_phi.size else float('nan'):.3g}, "
            f"{float(np.max(valid_phi)) if valid_phi.size else float('nan'):.3g}]"
        )

    # One positive-side ghost sample creates exactly the final periodic cell
    # along each direction; unlike symmetric padding, it does not duplicate
    # cells from the fundamental box.
    padded = np.pad(comp.phi, ((0, 1), (0, 1), (0, 1)), mode="wrap")
    verts0, faces, _, _ = marching_cubes(
        padded, level=float(level), spacing=tuple(float(v) for v in comp.grid.voxel),
        allow_degenerate=False,
    )
    tri = verts0[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    twice_area = np.linalg.norm(cross, axis=1)
    keep = twice_area > 1e-14
    if not np.all(keep):
        faces = faces[keep]
        tri = tri[keep]
        cross = cross[keep]
        twice_area = twice_area[keep]
    areas = 0.5 * twice_area

    origin = comp.grid.box_lo + 0.5 * comp.grid.voxel
    vertices = verts0 + origin
    centroids = np.mean(vertices[faces], axis=1)

    gx = _periodic_derivative(comp.phi, 0, float(comp.grid.voxel[0]))
    gy = _periodic_derivative(comp.phi, 1, float(comp.grid.voxel[1]))
    gz = _periodic_derivative(comp.phi, 2, float(comp.grid.voxel[2]))
    grad_face = np.column_stack([
        _sample_periodic(gx, comp, centroids),
        _sample_periodic(gy, comp, centroids),
        _sample_periodic(gz, comp, centroids),
    ])
    grad_norm = np.linalg.norm(grad_face, axis=1)
    normals = np.divide(
        grad_face, grad_norm[:, None], out=np.zeros_like(grad_face),
        where=grad_norm[:, None] > 1e-14,
    )

    H_grid, K_grid, gmag = level_set_curvatures(comp.phi, comp.grid.voxel)
    H = _sample_periodic(H_grid, comp, centroids)
    K = _sample_periodic(K_grid, comp, centroids)

    area = float(np.sum(areas))
    volume = float(np.prod(comp.grid.L))
    H_mean = _weighted_mean(H, areas)
    H_abs = _weighted_mean(np.abs(H), areas)
    H_rms = float(np.sqrt(_weighted_mean(H * H, areas)))
    K_mean = _weighted_mean(K, areas)
    goodK = np.isfinite(K)
    neg_frac = (float(np.sum(areas[goodK & (K < 0)]) / np.sum(areas[goodK]))
                if np.any(goodK) else float("nan"))
    K_integral = float(np.sum(K[goodK] * areas[goodK])) if np.any(goodK) else float("nan")

    good_n = np.linalg.norm(normals, axis=1) > 0
    tensor = np.zeros((3, 3), dtype=float)
    if np.any(good_n):
        w = areas[good_n]
        n = normals[good_n]
        tensor = np.einsum("f,fi,fj->ij", w, n, n) / np.sum(w)
    evals, evecs = np.linalg.eigh(tensor)
    anisotropy = float(np.sqrt(1.5 * np.sum((evals - 1.0 / 3.0) ** 2)))

    warnings: List[str] = []
    invalid_fraction = float(comp.invalid_mask.mean())
    if invalid_fraction > 0:
        warnings.append(
            f"{100 * invalid_fraction:.2f}% invalid-density voxels were filled at phi=0; "
            "surface area and curvature may contain artificial patches."
        )
    n_weak = int(np.sum(grad_norm < max(float(np.nanmax(gmag)) * 1e-3, 1e-12)))
    if n_weak:
        warnings.append(
            f"{n_weak} surface faces have very weak composition gradients; their normals "
            "and curvatures are poorly conditioned."
        )

    return SurfaceGeometry(
        vertices=vertices, faces=faces.astype(np.int32), centroids=centroids,
        normals_to_A=normals, face_areas=areas, mean_curvature=H,
        gaussian_curvature=K, area=area, area_per_volume=area / volume,
        H_mean=H_mean, H_abs_mean=H_abs, H_rms=H_rms, K_mean=K_mean,
        K_negative_area_fraction=neg_frac, K_integral=K_integral,
        gauss_bonnet_solid_chi_estimate=(K_integral / (4.0 * np.pi)
                                         if np.isfinite(K_integral) else float("nan")),
        normal_tensor=tensor, normal_tensor_eigenvalues=evals,
        normal_tensor_eigenvectors=evecs, normal_anisotropy=anisotropy,
        diagnostics={
            "method": "periodic-positive-ghost marching cubes + implicit-level-set curvature",
            "interface_level": float(level),
            "n_vertices": int(len(vertices)),
            "n_faces": int(len(faces)),
            "normal_direction": "toward increasing phi (A-rich side)",
            "curvature_note": (
                "H and K are area-weighted samples of derivatives of the smoothed phi field; "
                "compare systems only at fixed grid and smoothing settings"
            ),
            "gauss_bonnet_note": "integral(K)dA/(4*pi), comparable to solid-domain chi",
        },
        warnings=warnings,
    )
