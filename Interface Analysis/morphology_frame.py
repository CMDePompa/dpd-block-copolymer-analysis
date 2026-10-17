"""Per-frame analysis for sphere, cylinder, and gyroid morphologies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from .composition import CompositionField, composition_field
from .density_field import DensityFields, build_density_fields
from .domain_geometry import DomainGeometryResult, analyze_domain_geometry
from .general_mixing import GeneralMixingResult, compute_general_mixing
from .length_scale import LengthScaleResult, resolve_characteristic_length
from .normal_profiles import NormalProfileResult, compute_normal_profiles
from .surface_mesh import SurfaceGeometry, extract_surface_geometry
from .topology import TopologyResult, compute_topology
from .trajectory import FrameData


@dataclass
class MorphologyFrameAnalysis:
    timestep: int
    row: Dict[str, Any]
    component_rows: List[Dict[str, Any]]
    topology_rows: List[Dict[str, Any]]
    warnings: List[str]
    resolution: Dict[str, Any]
    fields: Optional[DensityFields] = None
    comp: Optional[CompositionField] = None
    surface: Optional[SurfaceGeometry] = None
    profile: Optional[NormalProfileResult] = None
    domains: Optional[DomainGeometryResult] = None
    mixing: Optional[GeneralMixingResult] = None
    topology: Optional[TopologyResult] = None
    length_scale: Optional[LengthScaleResult] = None


def analyze_morphology_frame(frame: FrameData, cfg: Mapping[str, Any],
                             sq_q_star: Optional[float] = None,
                             keep_detail: bool = False,
                             log=None) -> MorphologyFrameAnalysis:
    morphology = str(cfg["morphology"]).lower()
    if morphology not in ("sphere", "cylinder", "gyroid"):
        raise ValueError(f"nonlamellar frame analysis does not support {morphology!r}")
    warnings: List[str] = []

    fields = build_density_fields(
        frame, int(cfg["A_type_id"]), int(cfg["B_type_id"]),
        float(cfg["target_voxel_size"]), float(cfg["smoothing_sigma"]),
        grid_shape_override=cfg.get("grid_shape"), deposition=str(cfg["deposition"]),
        log=None,
    )
    warnings.extend(fields.warnings)
    comp = composition_field(fields, float(cfg["density_cutoff_fraction"]), log=None)
    warnings.extend(comp.warnings)

    length = resolve_characteristic_length(comp, sq_q_star)
    warnings.extend(length.warnings)
    hmax = float(max(comp.grid.voxel))
    if length.length / hmax < 8:
        warnings.append(
            f"characteristic_length/voxel={length.length / hmax:.1f} < 8; morphology "
            "geometry and curvature are under-resolved."
        )

    surface = extract_surface_geometry(comp, level=float(cfg.get("interface_level", 0.0)))
    warnings.extend(surface.warnings)
    mixing = compute_general_mixing(fields, comp, frame, cfg)
    profile = compute_normal_profiles(frame, fields, comp, surface, length.length, cfg)
    warnings.extend(profile.warnings)
    domains = analyze_domain_geometry(comp, surface, cfg)
    warnings.extend(domains.warnings)

    topo = compute_topology(
        comp, [float(t) for t in cfg["topology_thresholds"]],
        connectivity=int(cfg["topology_connectivity"]), lattice_mode=None,
        threshold_mode=str(cfg.get("topology_threshold_mode", "normalized_plateaus")),
        log=None,
    )
    warnings.extend(topo.warnings)
    phase0 = next((r for r in topo.rows
                   if r["phase"] == domains.minority_phase and abs(r["phi_c"]) < 1e-12), {})
    chi_topo = domains.metrics["domain_interface_level_chi"]
    chi_surface = surface.gauss_bonnet_solid_chi_estimate
    chi_difference = (float(chi_surface - chi_topo)
                      if isinstance(chi_topo, (int, float)) else float("nan"))

    row: Dict[str, Any] = {
        "timestep": int(frame.timestep),
        "morphology": morphology,
        "minority_phase": domains.minority_phase,
        "n_atoms": int(frame.n_atoms),
        "fraction_A": fields.diagnostics["fraction_A"],
        "phi_mean": comp.stats["phi_mean"],
        "phi_rms": comp.stats["phi_rms"],
        "phi_min": comp.stats["phi_min"],
        "phi_max": comp.stats["phi_max"],
        "fraction_invalid_voxels": comp.stats["fraction_invalid_voxels"],
        "q_star_characteristic": length.q_star,
        "characteristic_length": length.length,
        "surface_area": surface.area,
        "surface_area_per_volume": surface.area_per_volume,
        "surface_H_mean_signed_to_A": surface.H_mean,
        "surface_H_abs_mean": surface.H_abs_mean,
        "surface_H_rms": surface.H_rms,
        "surface_K_mean": surface.K_mean,
        "surface_K_negative_area_fraction": surface.K_negative_area_fraction,
        "surface_K_integral": surface.K_integral,
        "surface_gauss_bonnet_solid_chi_estimate": chi_surface,
        "surface_normal_anisotropy": surface.normal_anisotropy,
        **{f"f_mix_voxel_{k}": v for k, v in mixing.f_mix_voxel.items()},
        **{f"f_mix_particle_{k}": v for k, v in mixing.f_mix_particle.items()},
        **{f"wrong_side_A_{k}": v for k, v in mixing.wrong_side_A.items()},
        **{f"wrong_side_B_{k}": v for k, v in mixing.wrong_side_B.items()},
        "overlap_O": mixing.overlap_O,
        **profile.metrics,
        **domains.metrics,
        # These threshold-sweep values use normalized plateau coordinates; the
        # separately reported domain_interface_level_chi is evaluated at the
        # actual physical interface_level used for the mesh.
        "topo_minority_midpoint_beta0": phase0.get("beta0", float("nan")),
        "topo_minority_midpoint_n_wrapping": phase0.get("n_wrapping", float("nan")),
        "topo_minority_midpoint_n_islands": phase0.get("n_islands", float("nan")),
        "topo_minority_midpoint_chi_total": phase0.get("chi_total", float("nan")),
        "topo_minority_midpoint_connection_score": phase0.get("connection_score", float("nan")),
        "surface_vs_topology_chi_difference": chi_difference,
        "n_warnings": len(warnings),
    }
    component_rows = [
        {"timestep": int(frame.timestep), "morphology": morphology,
         "minority_phase": domains.minority_phase, **r}
        for r in domains.component_rows
    ]
    topology_rows = [{"timestep": int(frame.timestep), "morphology": morphology, **r}
                     for r in topo.rows]
    resolution = {
        "grid": fields.diagnostics,
        "composition": comp.stats,
        "characteristic_length": {
            "q_star": length.q_star, "length": length.length,
            "source": length.source, **length.diagnostics,
        },
        "surface": surface.diagnostics,
        "normal_profile": profile.diagnostics,
        "domains": domains.diagnostics,
        "mixing": mixing.diagnostics,
        "routing": {
            "morphology": morphology,
            "modules": [
                "density_field", "composition", "surface_mesh", "normal_profiles",
                "general_mixing", "domain_geometry", "topology",
            ],
            "lamellar_modules_skipped": [
                "orientation", "resample", "interface_detection", "height_fields",
                "waviness", "spacing", "lamellar width", "lamellar penetration",
            ],
        },
    }
    result = MorphologyFrameAnalysis(
        timestep=int(frame.timestep), row=row, component_rows=component_rows,
        topology_rows=topology_rows, warnings=warnings, resolution=resolution,
        profile=profile, domains=domains, mixing=mixing, length_scale=length,
    )
    if keep_detail:
        result.fields = fields
        result.comp = comp
        result.surface = surface
        result.topology = topo
    if log:
        log(
            f"frame {frame.timestep}: morphology={morphology}, phase={domains.minority_phase}, "
            f"A/V={surface.area_per_volume:.4f}, H_rms={surface.H_rms:.4f}, "
            f"width={profile.metrics['normal_width_lam_mean']:.4f}, "
            f"components={int(domains.metrics['domain_component_count'])}, "
            f"{len(warnings)} warnings"
        )
    return result
