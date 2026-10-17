"""Morphology-specific domain geometry from periodic thresholded fields.

The user supplies the morphology; these routines quantify it and emit
consistency warnings but never relabel a simulation.  Connected components
and wrapping use the same periodic union-find implementation as the topology
module.  Sphere and cylinder radii are reported both from volume/area ratios
and from the largest periodic Euclidean distance to the interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt

from .composition import CompositionField
from .surface_mesh import SurfaceGeometry
from .topology import euler_characteristic_periodic, periodic_label


@dataclass
class DomainGeometryResult:
    morphology: str
    minority_phase: str
    minority_mask: np.ndarray
    component_labels: np.ndarray
    component_rows: List[Dict[str, Any]]
    metrics: Dict[str, float]
    diagnostics: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)


def periodic_inside_distance(mask: np.ndarray, voxel: np.ndarray) -> np.ndarray:
    """Distance from inside voxels to the opposite phase with periodic BCs."""
    tiled = np.tile(np.asarray(mask, dtype=bool), (3, 3, 3))
    dist = distance_transform_edt(tiled, sampling=tuple(float(v) for v in voxel))
    shape = np.asarray(mask.shape, dtype=int)
    sl = tuple(slice(int(n), int(2 * n)) for n in shape)
    return dist[sl]


def _circular_centroid_and_shape(labels: np.ndarray, component: int,
                                 grid) -> Tuple[np.ndarray, np.ndarray, float]:
    ijk = np.argwhere(labels == component)
    if len(ijk) == 0:
        return np.full(3, np.nan), np.full(3, np.nan), float("nan")
    frac = (ijk + 0.5) / np.asarray(grid.shape, dtype=float)
    centroid_frac = np.empty(3)
    for d in range(3):
        z = np.mean(np.exp(2j * np.pi * frac[:, d]))
        centroid_frac[d] = (np.angle(z) / (2 * np.pi)) % 1.0
    delta = (frac - centroid_frac + 0.5) % 1.0 - 0.5
    xyz = delta * grid.L
    cov = xyz.T @ xyz / max(len(xyz), 1)
    evals = np.linalg.eigvalsh(cov)
    total = float(np.sum(evals))
    asphericity = (1.5 * float(np.sum((evals - total / 3.0) ** 2)) / (total * total)
                   if total > 0 else float("nan"))
    center = grid.box_lo + centroid_frac * grid.L
    return center, evals, asphericity


def _nearest_neighbor_distances(centers: np.ndarray, box_length: np.ndarray) -> np.ndarray:
    if len(centers) < 2:
        return np.full(len(centers), np.nan)
    out = np.full(len(centers), np.inf)
    for i in range(len(centers)):
        delta = centers - centers[i]
        delta -= box_length * np.round(delta / box_length)
        dist = np.linalg.norm(delta, axis=1)
        dist[i] = np.inf
        out[i] = float(np.min(dist))
    return out


def _resolve_minority_phase(comp: CompositionField, configured: object,
                            level: float) -> tuple[str, np.ndarray, float, float]:
    A = (~comp.invalid_mask) & (comp.phi > level)
    B = (~comp.invalid_mask) & (comp.phi < level)
    fA, fB = float(np.mean(A)), float(np.mean(B))
    text = str(configured).strip().upper()
    if text == "AUTO":
        text = "A" if fA <= fB else "B"
    if text not in ("A", "B"):
        raise ValueError("minority_phase must be 'A', 'B', or 'auto'")
    return text, (A if text == "A" else B), fA, fB


def analyze_domain_geometry(comp: CompositionField, surface: SurfaceGeometry,
                            cfg: Mapping[str, Any]) -> DomainGeometryResult:
    morphology = str(cfg["morphology"]).strip().lower()
    level = float(cfg.get("interface_level", 0.0))
    phase, mask, fA, fB = _resolve_minority_phase(
        comp, cfg.get("minority_phase", "auto"), level,
    )
    pc = periodic_label(mask, connectivity=int(cfg.get("topology_connectivity", 6)),
                        compute_chi=False)
    inside_distance = periodic_inside_distance(mask, comp.grid.voxel)
    voxel_volume = comp.grid.voxel_volume
    total_volume = float(np.sum(mask) * voxel_volume)
    box_volume = float(np.prod(comp.grid.L))

    component_rows: List[Dict[str, Any]] = []
    centers: List[np.ndarray] = []
    center_row_indices: List[int] = []
    for i in range(1, pc.n_components + 1):
        voxels = int(pc.volumes[i - 1])
        volume = voxels * voxel_volume
        wraps = pc.wraps[i - 1]
        center, inertia, asphericity = _circular_centroid_and_shape(pc.labels, i, comp.grid)
        if np.any(wraps):
            # A single circular centroid cannot faithfully unwrap a winding domain.
            inertia = np.full(3, np.nan)
            asphericity = float("nan")
        max_core = float(np.max(inside_distance[pc.labels == i]))
        radius_eq = float((3.0 * volume / (4.0 * np.pi)) ** (1.0 / 3.0))
        row = {
            "component": i,
            "voxel_count": voxels,
            "volume": volume,
            "volume_fraction_box": volume / box_volume,
            "wrap_x": bool(wraps[0]), "wrap_y": bool(wraps[1]),
            "wrap_z": bool(wraps[2]), "wrap_rank_cartesian": int(np.sum(wraps)),
            "centroid_x": float(center[0]), "centroid_y": float(center[1]),
            "centroid_z": float(center[2]),
            "inertia_eigenvalue_1": float(inertia[0]),
            "inertia_eigenvalue_2": float(inertia[1]),
            "inertia_eigenvalue_3": float(inertia[2]),
            "asphericity": asphericity,
            "equivalent_sphere_radius": radius_eq,
            "maximum_core_distance": max_core,
        }
        component_rows.append(row)
        if not np.any(wraps):
            centers.append(center)
            center_row_indices.append(len(component_rows) - 1)

    if centers:
        nn = _nearest_neighbor_distances(np.asarray(centers), comp.grid.L)
        for row_index, value in zip(center_row_indices, nn):
            component_rows[row_index]["nearest_neighbor_distance"] = float(value)
    for row in component_rows:
        row.setdefault("nearest_neighbor_distance", float("nan"))

    evals = surface.normal_tensor_eigenvalues
    evecs = surface.normal_tensor_eigenvectors
    auto_axis = evecs[:, 0]
    axis_cfg = cfg.get("cylinder_axis", "auto")
    if isinstance(axis_cfg, str) and axis_cfg.lower() == "auto":
        cylinder_axis = auto_axis
        axis_source = "smallest-eigenvalue eigenvector of area-weighted interface-normal tensor"
    else:
        cylinder_axis = np.asarray(axis_cfg, dtype=float)
        cylinder_axis /= np.linalg.norm(cylinder_axis)
        axis_source = "explicit cylinder_axis"
    for value in cylinder_axis:
        if abs(value) > 1e-12:
            if value < 0:
                cylinder_axis = -cylinder_axis
            break
    cylinder_alignment = float(np.clip(1.0 - 3.0 * evals[0], -0.5, 1.0))

    in_core = inside_distance[mask]
    domain_chi = float(euler_characteristic_periodic(mask))
    metrics: Dict[str, float] = {
        "minority_volume_fraction": total_volume / box_volume,
        "domain_component_count": float(pc.n_components),
        "domain_wrapping_component_count": float(np.sum(pc.wraps.any(axis=1))),
        "domain_fully_wrapping_component_count": float(np.sum(pc.wraps.all(axis=1))),
        "domain_interface_level_chi": domain_chi,
        "domain_core_distance_mean": float(np.mean(in_core)) if in_core.size else float("nan"),
        "domain_core_distance_p95": float(np.percentile(in_core, 95)) if in_core.size else float("nan"),
        "domain_core_distance_max": float(np.max(in_core)) if in_core.size else float("nan"),
        "sphere_radius_volume_area": (3.0 * total_volume / surface.area
                                       if surface.area > 0 else float("nan")),
        "cylinder_radius_volume_area": (2.0 * total_volume / surface.area
                                         if surface.area > 0 else float("nan")),
        "cylinder_axis_x": float(cylinder_axis[0]),
        "cylinder_axis_y": float(cylinder_axis[1]),
        "cylinder_axis_z": float(cylinder_axis[2]),
        "cylinder_axis_alignment": cylinder_alignment,
        "surface_normal_tensor_eig_min": float(evals[0]),
        "surface_normal_tensor_eig_mid": float(evals[1]),
        "surface_normal_tensor_eig_max": float(evals[2]),
    }
    sphere_rows = [r for r in component_rows if r["wrap_rank_cartesian"] == 0]
    if sphere_rows:
        radii = np.asarray([r["equivalent_sphere_radius"] for r in sphere_rows])
        asph = np.asarray([r["asphericity"] for r in sphere_rows])
        nn = np.asarray([r["nearest_neighbor_distance"] for r in sphere_rows])
        metrics.update({
            "sphere_count_nonwrapping": float(len(sphere_rows)),
            "sphere_radius_equivalent_mean": float(np.mean(radii)),
            "sphere_radius_equivalent_std": (float(np.std(radii, ddof=1))
                                              if len(radii) > 1 else 0.0),
            "sphere_asphericity_mean": float(np.nanmean(asph)),
            "sphere_nearest_neighbor_mean": (float(np.nanmean(nn))
                                               if np.any(np.isfinite(nn)) else float("nan")),
        })

    warnings: List[str] = []
    n_wrap = int(np.sum(pc.wraps.any(axis=1)))
    n_full = int(np.sum(pc.wraps.all(axis=1)))
    if morphology == "sphere" and n_wrap:
        warnings.append(
            f"target morphology is sphere but {n_wrap} minority component(s) wrap the box; "
            "inspect for coalescence, a transition, or threshold sensitivity."
        )
    if morphology == "cylinder":
        if n_wrap == 0:
            warnings.append(
                "target morphology is cylinder but no minority component wraps a Cartesian "
                "box direction; finite/tilted cylinders or broken domains may be present."
            )
        if cylinder_alignment < 0.5:
            warnings.append(
                f"cylinder normal-tensor alignment is low ({cylinder_alignment:.2f}); "
                "axes may be disordered or the selected morphology may not describe the frame."
            )
    if morphology == "gyroid" and n_full == 0:
        warnings.append(
            "target morphology is gyroid but no minority component wraps all three Cartesian "
            "directions; inspect threshold sensitivity and network continuity."
        )

    return DomainGeometryResult(
        morphology=morphology, minority_phase=phase, minority_mask=mask,
        component_labels=pc.labels, component_rows=component_rows, metrics=metrics,
        diagnostics={
            "interface_level": level,
            "volume_fraction_A_at_level": fA,
            "volume_fraction_B_at_level": fB,
            "minority_phase_rule": str(cfg.get("minority_phase", "auto")),
            "periodic_connectivity": int(cfg.get("topology_connectivity", 6)),
            "inside_distance_method": "3x3x3 periodic tiled Euclidean distance transform",
            "cylinder_axis_source": axis_source,
            "wrap_rank_note": (
                "wrap flags are Cartesian winding indicators; an oblique periodic tube may "
                "wind through more than one Cartesian direction"
            ),
        }, warnings=warnings,
    )
