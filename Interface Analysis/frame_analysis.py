"""Per-frame orchestration: fields -> phi -> n0 -> slab -> interfaces -> metrics.

Every frame is analyzed independently (no time-averaged density field — time
averaging would erase transient waviness and defects). The result carries a
flat scalar `row` for aggregation, per-interface rows, the collected
warnings, and (optionally) the full field/map detail for diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .composition import CompositionField, composition_field
from .density_field import DensityFields, build_density_fields
from .density_profiles import DensityProfileResult, compute_density_profile
from .height_fields import HeightMap, prepare_height_map
from .interface_detection import COL_CLASS_NAMES, COL_GOOD, InterfaceSet, detect_interfaces
from .io_utils import LogFn
from .orientation import OrientationResult, resolve_orientation
from .penetration import PenetrationResult, compute_penetration
from .resample import SlabField, resample_slab
from .spacing import SpacingResult, compute_spacing
from .surface_geometry import (
    AreaResult,
    OrientationOrderResult,
    compute_area,
    compute_orientational_order,
)
from .topology import TopologyResult, compute_topology
from .trajectory import FrameData
from .waviness import WavinessResult, compute_waviness
from .width import WidthResult, compute_width


@dataclass
class FrameAnalysis:
    """Everything computed for one frame."""

    timestep: int
    row: Dict[str, Any]                      # flat scalars for per-frame CSV
    interface_rows: List[Dict[str, Any]]     # one row per interface
    topology_rows: List[Dict[str, Any]]      # threshold sweep rows (+ timestep)
    warnings: List[str]
    # full detail (heavy; only retained when keep_detail=True)
    fields: Optional[DensityFields] = None
    comp: Optional[CompositionField] = None
    orientation: Optional[OrientationResult] = None
    slab: Optional[SlabField] = None
    iset: Optional[InterfaceSet] = None
    hmaps: Optional[List[HeightMap]] = None
    waviness: Optional[List[WavinessResult]] = None
    spacing: Optional[SpacingResult] = None
    area: Optional[AreaResult] = None
    orient_order: Optional[OrientationOrderResult] = None
    width: Optional[WidthResult] = None
    penetration: Optional[PenetrationResult] = None
    density_profile: Optional[DensityProfileResult] = None
    topology: Optional[TopologyResult] = None
    resolution: Dict[str, Any] = field(default_factory=dict)


def analyze_frame(
    frame: FrameData,
    cfg: Dict[str, Any],
    sq_q_star: Optional[float] = None,
    keep_detail: bool = False,
    log: Optional[LogFn] = None,
) -> FrameAnalysis:
    """Run the full interface analysis for one frame."""
    warnings: List[str] = []

    # 1. Density and composition fields
    fields = build_density_fields(
        frame, int(cfg["A_type_id"]), int(cfg["B_type_id"]),
        float(cfg["target_voxel_size"]), float(cfg["smoothing_sigma"]),
        grid_shape_override=cfg.get("grid_shape"),
        deposition=str(cfg["deposition"]), log=None,
    )
    warnings += fields.warnings
    comp = composition_field(fields, float(cfg["density_cutoff_fraction"]), log=None)
    warnings += comp.warnings

    # 2. Lamellar normal
    ori = resolve_orientation(comp, cfg, sq_q_star=sq_q_star, log=None)
    warnings += ori.warnings

    # Optional architecture-resolved A_head/A_tail/B_tail profile.  It
    # uses the same detected reciprocal-lattice mode as the interface slab,
    # so oblique lamellae are never forced onto x/y/z.
    density_profile = None
    if bool(cfg.get("density_profile_enabled", False)):
        density_profile = compute_density_profile(frame, ori.lattice_mode, cfg)
        warnings += density_profile.warnings

    # 3. Oblique slab + interface detection
    slab = resample_slab(comp, ori.n0, ori.L0, cfg, log=None,
                         lattice_mode=ori.lattice_mode)
    warnings += slab.warnings
    iset = detect_interfaces(slab, cfg, log=None)
    warnings += iset.warnings

    # resolution-adequacy check that needs L0 (task: warn when the grid
    # cannot resolve the lamellar period)
    h_max = float(np.max(comp.grid.voxel))
    if ori.L0 > 0 and ori.L0 / h_max < 8:
        warnings.append(
            f"L0/voxel = {ori.L0 / h_max:.1f} < 8: the lamellar period is "
            "under-resolved; reduce target_voxel_size."
        )

    # 4. Height maps and geometry
    # Intrinsic/geometric interface metrics are defined on unambiguous,
    # expected-crossing columns only. Defected columns remain in the defect
    # map/topology but cannot silently bias width, waviness, spacing, or area.
    clean_columns = iset.column_class == COL_GOOD
    hmaps = [prepare_height_map(np.where(clean_columns, iset.h[j], np.nan),
                                slab.du, slab.dv)
             for j in range(iset.n_interfaces)]
    wav_results = [compute_waviness(hm) for hm in hmaps]
    L0_sq = (2.0 * np.pi / sq_q_star) if (sq_q_star and sq_q_star > 0) else None
    spacing = compute_spacing(iset, L0_sq=L0_sq)
    warnings += spacing.warnings
    max_infill = float(cfg.get("max_height_infill_fraction", 0.20))
    area = compute_area(
        hmaps, comp=comp, box_volume=float(np.prod(comp.grid.L)),
        max_filled_fraction=max_infill,
    )
    orient_order = compute_orientational_order(comp, ori.n0)

    # 5. Width and penetration
    width = compute_width(slab, iset, cfg, smoothing_sigma=float(cfg["smoothing_sigma"]), log=None)
    warnings += width.warnings
    pen = compute_penetration(fields, comp, slab, iset, frame, cfg, log=None)
    warnings += pen.warnings

    # 6. Topology
    topo = compute_topology(comp, [float(t) for t in cfg["topology_thresholds"]],
                            connectivity=int(cfg["topology_connectivity"]),
                            lattice_mode=ori.lattice_mode,
                            threshold_mode=str(cfg.get("topology_threshold_mode",
                                                       "absolute_phi")), log=None)
    warnings += topo.warnings

    # ---- flat per-frame row ----
    valid_wavs = [w for w, hm in zip(wav_results, hmaps)
                  if np.isfinite(w.rms) and hm.filled_fraction <= max_infill]
    rms_list = [w.rms for w in valid_wavs]
    p2p_list = [w.peak_to_peak for w in valid_wavs]
    slope_rms_list = [w.slope_rms for w in valid_wavs if np.isfinite(w.slope_rms)]
    excess_list = [e for e in area.per_interface_excess if np.isfinite(e)]

    topoA0 = next((r for r in topo.rows
                   if r["phase"] == "A" and abs(r["phi_c"]) < 1e-12), {})

    row: Dict[str, Any] = {
        "timestep": frame.timestep,
        "n_atoms": frame.n_atoms,
        "fraction_A": fields.diagnostics["fraction_A"],
        "phi_mean": comp.stats["phi_mean"],
        "phi_rms": comp.stats["phi_rms"],
        "phi_min": comp.stats["phi_min"],
        "phi_max": comp.stats["phi_max"],
        "fraction_invalid_voxels": comp.stats["fraction_invalid_voxels"],
        "n0_x": float(ori.n0[0]), "n0_y": float(ori.n0[1]), "n0_z": float(ori.n0[2]),
        "q_star_fft": ori.q_star_mag,
        "L0_fft": ori.L0,
        "L0_slab_used": slab.L0,
        "normal_window_periodic": slab.w_periodic,
        "orientation_confidence": ori.confidence,
        "orientation_secondary_ratio": ori.secondary_ratio,
        "n_interfaces": iset.n_interfaces,
        "n_valid_interfaces": iset.n_valid_interfaces,
        "valid_interface_fraction": (iset.n_valid_interfaces / iset.n_interfaces
                                     if iset.n_interfaces else 0.0),
        "bad_column_fraction": iset.bad_column_fraction,
        **{f"col_frac_{name}": iset.class_fractions.get(name, 0.0)
           for name in COL_CLASS_NAMES.values()},
        "waviness_rms": float(np.mean(rms_list)) if rms_list else float("nan"),
        "waviness_p2p": float(np.mean(p2p_list)) if p2p_list else float("nan"),
        "slope_rms": float(np.mean(slope_rms_list)) if slope_rms_list else float("nan"),
        "spacing_mean": spacing.mean,
        "spacing_std": spacing.std,
        "spacing_cv": spacing.cv,
        "spacing_clean_coverage": spacing.clean_coverage,
        "spacing_vs_sq_rel_diff": (spacing.rel_diff_vs_sq
                                   if spacing.rel_diff_vs_sq is not None else float("nan")),
        "excess_area_mean": float(np.mean(excess_list)) if excess_list else float("nan"),
        "area_total_height_field": area.total_height_field_area,
        "area_height_field_sampled_window": area.total_height_field_area,
        "area_marching_cubes": (area.marching_cubes_area
                                if area.marching_cubes_area is not None else float("nan")),
        "area_per_volume": (area.area_per_volume
                            if area.area_per_volume is not None else float("nan")),
        "S_n": orient_order.S_n,
        "misorientation_mean_deg": orient_order.mean_angle_deg,
        "misorientation_std_deg": orient_order.std_angle_deg,
        **{f"misorient_frac_gt_{k}": v for k, v in orient_order.fraction_above.items()},
        "width_lam_mean": width.mean_lam,
        "width_lam_std": width.std_lam,
        "width_lam_corrected_mean": width.mean_lam_corrected,
        "width_10_90": width.mean_w1090,
        "width_10_90_deconv": width.w1090_deconv_sigma,
        "width_failed_fraction": width.failed_fraction,
        "width_clean_column_fraction": width.diagnostics.get("clean_column_fraction",
                                                              float("nan")),
        "width_naive_over_aligned": width.waviness_broadening_ratio,
        **{f"f_mix_voxel_{k}": v for k, v in pen.f_mix_voxel.items()},
        **{f"f_mix_particle_{k}": v for k, v in pen.f_mix_particle.items()},
        **{f"wrong_side_A_{k}": v for k, v in pen.wrong_side_A.items()},
        **{f"wrong_side_B_{k}": v for k, v in pen.wrong_side_B.items()},
        "P_B_to_A": pen.P_B_to_A,
        "P_A_to_B": pen.P_A_to_B,
        "penetration_clean_coverage": pen.diagnostics.get("clean_eligible_coverage",
                                                           float("nan")),
        "overlap_O": pen.overlap_O,
        "density_profile_domain_contrast": (
            density_profile.metrics["density_profile_domain_contrast"]
            if density_profile is not None else float("nan")
        ),
        "density_profile_A_tail_interface_enrichment": (
            density_profile.metrics["density_profile_A_tail_interface_enrichment"]
            if density_profile is not None else float("nan")
        ),
        "density_profile_midpoint_crossings": (
            density_profile.metrics["density_profile_midpoint_crossings"]
            if density_profile is not None else float("nan")
        ),
        "density_profile_alignment_shift_bins": (
            density_profile.shift_bins if density_profile is not None else float("nan")
        ),
        "topo_A0_beta0": topoA0.get("beta0", float("nan")),
        "topo_A0_n_islands": topoA0.get("n_islands", float("nan")),
        "topo_A0_chi_total": topoA0.get("chi_total", float("nan")),
        "topo_A0_excess_connections": topoA0.get("n_excess_connections", float("nan")),
        "topo_A0_connection_score": topoA0.get("connection_score", float("nan")),
        "n_warnings": len(warnings),
    }

    # ---- per-interface rows ----
    interface_rows: List[Dict[str, Any]] = []
    for j in range(iset.n_interfaces):
        wv = wav_results[j]
        interface_rows.append({
            "timestep": frame.timestep,
            "interface": j,
            "type": "A_side_up" if iset.plane_type[j] > 0 else "B_side_up",
            "ideal_position_w": float(iset.plane_ideal[j]),
            "mean_position_w": float(iset.plane_mean[j]),
            "valid_fraction": float(iset.valid_fraction[j]),
            "rms_waviness": wv.rms,
            "peak_to_peak": wv.peak_to_peak,
            "slope_mean": wv.slope_mean,
            "slope_rms": wv.slope_rms,
            "slope_max": wv.slope_max,
            "corr_length": wv.corr_length,
            "dominant_lambda": wv.dominant_lambda,
            "anisotropy_ratio": wv.anisotropy_ratio,
            "area": area.per_interface_area[j],
            "excess_area": area.per_interface_excess[j],
            "filled_fraction": hmaps[j].filled_fraction,
        })

    topology_rows = [{"timestep": frame.timestep, **r} for r in topo.rows]

    resolution = {
        "grid": fields.diagnostics,
        "composition": comp.stats,
        "orientation": ori.diagnostics,
        "slab": slab.diagnostics,
        "detection": iset.diagnostics,
        "density_profile": (density_profile.diagnostics
                            if density_profile is not None else {"enabled": False}),
    }

    result = FrameAnalysis(
        timestep=frame.timestep, row=row, interface_rows=interface_rows,
        topology_rows=topology_rows, warnings=warnings, resolution=resolution,
        density_profile=density_profile,
    )
    if keep_detail:
        result.fields = fields
        result.comp = comp
        result.orientation = ori
        result.slab = slab
        result.iset = iset
        result.hmaps = hmaps
        result.waviness = wav_results
        result.spacing = spacing
        result.area = area
        result.orient_order = orient_order
        result.width = width
        result.penetration = pen
        result.density_profile = density_profile
        result.topology = topo
    else:
        # width/penetration hold modest arrays needed for pooled profiles
        result.width = width
        result.penetration = pen
        result.spacing = spacing

    if log:
        log(f"frame {frame.timestep}: n0={np.round(ori.n0, 3).tolist()}, "
            f"L0={ori.L0:.2f}, ifaces={iset.n_valid_interfaces}/{iset.n_interfaces}, "
            f"f_bad={iset.bad_column_fraction:.3f}, w_rms={row['waviness_rms']:.3f}, "
            f"lam={width.mean_lam_corrected:.3f}, P_A->B={pen.P_A_to_B:.3f}, "
            f"{len(warnings)} warnings")
    return result
