"""Morphology-independent intrinsic profiles along local interface normals.

Triangles are sampled proportional to their area.  At each selected triangle
centroid, the composition and density fields are interpolated along the local
normal pointing toward the A-rich phase.  Centering every profile on the
instantaneous phi-level surface removes translational, capillary-wave, and
large-scale shape broadening before the tanh width is fitted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
from scipy.optimize import curve_fit
from scipy.special import erf, erfinv

from .composition import CompositionField, sample_field_at_points
from .density_field import DensityFields, deposit_counts, smooth_field
from .density_profiles import _category_masks, _resolve_bead_indices
from .surface_mesh import SurfaceGeometry
from .trajectory import FrameData
from .width import GAUSS_1090, W1090_PER_LAM, _fit_profile


@dataclass
class NormalProfileResult:
    s: np.ndarray
    mean: Dict[str, np.ndarray]
    std: Dict[str, np.ndarray]
    n_samples: int
    lam_values: np.ndarray
    r2_values: np.ndarray
    metrics: Dict[str, float]
    diagnostics: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)


def _integrate_trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    """NumPy 1.x/2.x compatible trapezoidal integral."""
    fn = getattr(np, "trapezoid", None)
    if fn is None:  # NumPy < 2.0
        fn = np.trapz
    return float(fn(y, x))


def _area_systematic_indices(areas: np.ndarray, n_target: int) -> np.ndarray:
    """Deterministic equal-area systematic sample of face indices."""
    areas = np.asarray(areas, dtype=float)
    good = np.isfinite(areas) & (areas > 0)
    idx = np.flatnonzero(good)
    if idx.size == 0:
        return np.zeros(0, dtype=int)
    # Even when every face could be visited once, use equal-area quantiles:
    # marching-cubes triangles are not exactly equal in area, and an
    # unweighted all-face average would bias the profile toward small faces.
    n_target = min(int(n_target), int(idx.size))
    cumulative = np.cumsum(areas[idx])
    targets = (np.arange(n_target, dtype=float) + 0.5) * cumulative[-1] / n_target
    return idx[np.searchsorted(cumulative, targets, side="left")]


def _sample_lines(field: np.ndarray, comp: CompositionField, centers: np.ndarray,
                  normals: np.ndarray, s: np.ndarray) -> np.ndarray:
    points = centers[:, None, :] + normals[:, None, :] * s[None, :, None]
    grid = comp.grid
    points = grid.box_lo + np.mod(points - grid.box_lo, grid.L)
    values = sample_field_at_points(field, grid, points.reshape(-1, 3), order=1)
    return values.reshape(len(centers), len(s))


def _derivative_moment_width(s: np.ndarray, profile: np.ndarray) -> float:
    """Second-moment width of |d profile/ds| (Muller--Munster definition)."""
    good = np.isfinite(s) & np.isfinite(profile)
    if np.sum(good) < 5:
        return float("nan")
    x, y = np.asarray(s[good], dtype=float), np.asarray(profile[good], dtype=float)
    grad = np.abs(np.gradient(y, x))
    norm = _integrate_trapezoid(grad, x)
    if norm <= 1e-12:
        return float("nan")
    center = _integrate_trapezoid(x * grad, x) / norm
    variance = _integrate_trapezoid((x - center) ** 2 * grad, x) / norm
    return float(np.sqrt(max(variance, 0.0)))


def _fit_erf_width(s: np.ndarray, profile: np.ndarray) -> Optional[tuple[float, float]]:
    """Fit a+b*erf((s-s0)/(sqrt(2)*sigma)); return (sigma, R2)."""
    good = np.isfinite(s) & np.isfinite(profile)
    if np.sum(good) < 7:
        return None
    x, y = np.asarray(s[good], dtype=float), np.asarray(profile[good], dtype=float)
    span = float(x[-1] - x[0])
    dx = float(np.median(np.diff(x)))

    def model(xx, a, b, s0, sigma):
        return a + b * erf((xx - s0) / (np.sqrt(2.0) * sigma))

    a0 = 0.5 * float(y[0] + y[-1])
    b0 = max(0.5 * float(y[-1] - y[0]), 0.05)
    try:
        pars, _ = curve_fit(
            model, x, y, p0=(a0, b0, 0.0, max(span / 10.0, dx)),
            bounds=([-1.5, 0.0, x[0], max(dx / 4.0, 1e-8)],
                    [1.5, 2.0, x[-1], max(2.0 * span, dx)]),
            maxfev=10000,
        )
    except (RuntimeError, ValueError, FloatingPointError):
        return None
    pred = model(x, *pars)
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(abs(pars[3])), float(r2)


def _architecture_fields(frame: FrameData, fields: DensityFields,
                         cfg: Mapping[str, Any]) -> tuple[Dict[str, np.ndarray], Dict[str, Any], List[str]]:
    chain_length = int(cfg["density_profile_chain_length"])
    pure_length = int(cfg["density_profile_pure_a_length"])
    bead, source, warnings = _resolve_bead_indices(
        frame, chain_length,
        bool(cfg.get("density_profile_allow_inferred_bead_order", False)),
    )
    masks = _category_masks(
        frame, bead, pure_length, int(cfg["A_type_id"]), int(cfg["B_type_id"]),
    )
    out: Dict[str, np.ndarray] = {}
    sigma = float(cfg["smoothing_sigma"])
    method = str(cfg["deposition"])
    for name, mask in masks.items():
        counts = deposit_counts(frame.coords[mask], fields.grid, method=method)
        out[f"rho_{name}"] = smooth_field(counts, sigma, fields.grid) / fields.grid.voxel_volume
    return out, {
        "topology_source": source,
        "chain_length": chain_length,
        "pure_a_length": pure_length,
    }, warnings


def compute_normal_profiles(
    frame: FrameData,
    fields: DensityFields,
    comp: CompositionField,
    surface: SurfaceGeometry,
    characteristic_length: float,
    cfg: Mapping[str, Any],
) -> NormalProfileResult:
    """Sample local-normal profiles and fit intrinsic widths for one frame."""
    max_samples = int(cfg.get("surface_normal_samples", 2048))
    idx = _area_systematic_indices(surface.face_areas, max_samples)
    if idx.size < 8:
        raise RuntimeError("too few valid surface triangles for local-normal profiles")
    centers = surface.centroids[idx]
    normals = surface.normals_to_A[idx]
    good = np.linalg.norm(normals, axis=1) > 0.5
    centers, normals = centers[good], normals[good]
    if len(centers) < 8:
        raise RuntimeError("too few well-conditioned surface normals for profiles")

    frac = float(cfg.get("normal_profile_halfwidth_fraction", 0.25))
    configured = cfg.get("normal_profile_halfwidth", "auto")
    if isinstance(configured, str) and configured.lower() == "auto":
        s_max = frac * float(characteristic_length)
        halfwidth_source = f"{frac:g} * characteristic_length"
    else:
        s_max = float(configured)
        halfwidth_source = "explicit normal_profile_halfwidth"
    n_points = int(cfg.get("normal_profile_points", 65))
    if n_points % 2 == 0:
        n_points += 1
    s = np.linspace(-s_max, s_max, n_points)

    sample_fields: Dict[str, np.ndarray] = {
        "phi": comp.phi,
        "rho_A": fields.rho_A,
        "rho_B": fields.rho_B,
    }
    arch_diag: Dict[str, Any] = {"enabled": False}
    warnings: List[str] = []
    if bool(cfg.get("density_profile_enabled", False)):
        arch, arch_diag, arch_warnings = _architecture_fields(frame, fields, cfg)
        sample_fields.update(arch)
        arch_diag["enabled"] = True
        warnings.extend(arch_warnings)

    samples = {
        name: _sample_lines(values, comp, centers, normals, s)
        for name, values in sample_fields.items()
    }
    mean = {name: np.nanmean(values, axis=0) for name, values in samples.items()}
    std = {name: np.nanstd(values, axis=0, ddof=1) for name, values in samples.items()}

    rho_total = samples["rho_A"] + samples["rho_B"]
    for name in ("rho_A", "rho_B"):
        key = "fraction_" + name.removeprefix("rho_")
        values = np.divide(samples[name], rho_total, out=np.full_like(rho_total, np.nan),
                           where=rho_total > 1e-12)
        mean[key] = np.nanmean(values, axis=0)
        std[key] = np.nanstd(values, axis=0, ddof=1)
    for name in ("rho_A_head", "rho_A_tail", "rho_B_tail"):
        if name in samples:
            values = np.divide(samples[name], rho_total,
                               out=np.full_like(rho_total, np.nan), where=rho_total > 1e-12)
            key = "fraction_" + name.removeprefix("rho_")
            mean[key] = np.nanmean(values, axis=0)
            std[key] = np.nanstd(values, axis=0, ddof=1)

    lam0 = max(float(cfg.get("smoothing_sigma", 0.75)), float(np.mean(comp.grid.voxel)))
    min_r2 = float(cfg.get("width_min_r2", 0.90))
    lam: List[float] = []
    r2s: List[float] = []
    n_attempted = len(centers)
    for profile in samples["phi"]:
        fit = _fit_profile(s, profile, lam0, s_max)
        if fit is None:
            continue
        _, b, _, ll, r2 = fit
        if b > 0 and r2 >= min_r2:
            lam.append(ll)
            r2s.append(r2)
    lam_values = np.asarray(lam, dtype=float)
    r2_values = np.asarray(r2s, dtype=float)
    pooled_fit = _fit_profile(s, mean["phi"], lam0, s_max)

    positive = s > 0
    negative = s < 0
    rhoA_mean, rhoB_mean = mean["rho_A"], mean["rho_B"]
    intA = _integrate_trapezoid(rhoA_mean, s)
    intB = _integrate_trapezoid(rhoB_mean, s)
    P_A_to_B = (_integrate_trapezoid(rhoA_mean[negative], s[negative]) / intA
                if intA > 0 and np.sum(negative) > 1 else float("nan"))
    P_B_to_A = (_integrate_trapezoid(rhoB_mean[positive], s[positive]) / intB
                if intB > 0 and np.sum(positive) > 1 else float("nan"))

    enrichment = float("nan")
    if "fraction_A_tail" in mean:
        window = float(cfg.get("density_profile_interface_window_fraction", 0.05)) * 2.0 * s_max
        near = np.abs(s) <= window
        b_core = s <= -0.70 * s_max
        if np.any(near) and np.any(b_core):
            enrichment = float(np.nanmean(mean["fraction_A_tail"][near])
                               - np.nanmean(mean["fraction_A_tail"][b_core]))

    width_mean = float(np.mean(lam_values)) if lam_values.size else float("nan")
    width_std = (float(np.std(lam_values, ddof=1)) if lam_values.size > 1
                 else (0.0 if lam_values.size else float("nan")))
    width_median = float(np.median(lam_values)) if lam_values.size else float("nan")
    width_1090 = W1090_PER_LAM * width_mean
    width_deconv = (float(np.sqrt(max(width_1090 ** 2
                                      - (GAUSS_1090 * float(cfg["smoothing_sigma"])) ** 2, 0.0)))
                    if np.isfinite(width_1090) else float("nan"))
    failed = 1.0 - len(lam_values) / max(n_attempted, 1)
    moment_sigma = _derivative_moment_width(s, mean["phi"])
    erf_fit = _fit_erf_width(s, mean["phi"])
    erf_sigma = float(erf_fit[0]) if erf_fit else float("nan")
    erf_r2 = float(erf_fit[1]) if erf_fit else float("nan")
    erf_1090_factor = 2.0 * np.sqrt(2.0) * float(erfinv(0.8))
    smoothing_sigma = float(cfg["smoothing_sigma"])
    erf_sigma_deconv = (float(np.sqrt(max(erf_sigma ** 2 - smoothing_sigma ** 2, 0.0)))
                        if np.isfinite(erf_sigma) else float("nan"))
    if failed > 0.5:
        warnings.append(
            f"{100 * failed:.0f}% of local-normal tanh fits failed; the configured morphology "
            "may contain defects, the profile window may cross another interface, or the field "
            "may be under-resolved."
        )

    metrics = {
        "normal_width_lam_mean": width_mean,
        "normal_width_lam_std_spatial": width_std,
        "normal_width_lam_median": width_median,
        "normal_width_10_90": width_1090,
        "normal_width_10_90_deconv": width_deconv,
        "normal_width_failed_fraction": failed,
        "normal_width_n_attempted": float(n_attempted),
        "normal_width_n_succeeded": float(len(lam_values)),
        "normal_width_pooled_lam": (float(pooled_fit[3]) if pooled_fit else float("nan")),
        "normal_width_pooled_r2": (float(pooled_fit[4]) if pooled_fit else float("nan")),
        "normal_width_derivative_moment_sigma_pooled": moment_sigma,
        "normal_width_erf_sigma_pooled": erf_sigma,
        "normal_width_erf_sigma_deconv_pooled": erf_sigma_deconv,
        "normal_width_erf_10_90_pooled": erf_1090_factor * erf_sigma,
        "normal_width_erf_r2_pooled": erf_r2,
        "normal_P_A_to_B": P_A_to_B,
        "normal_P_B_to_A": P_B_to_A,
        "normal_A_tail_interface_enrichment": enrichment,
    }
    return NormalProfileResult(
        s=s, mean=mean, std=std, n_samples=len(centers),
        lam_values=lam_values, r2_values=r2_values, metrics=metrics,
        diagnostics={
            "method": "area-systematic local-normal sampling of instantaneous phi isosurface",
            "normal_direction": "s > 0 points toward increasing phi (A-rich side)",
            "profile_halfwidth": s_max,
            "profile_halfwidth_source": halfwidth_source,
            "n_profile_points": len(s),
            "n_surface_samples": len(centers),
            "fit_model": "a + b*tanh((s-s0)/lambda), b>0",
            "independent_width_estimator": (
                "second moment of |d<phi>/ds| plus an error-function fit whose sigma "
                "is that derivative variance; Gaussian smoothing deconvolves in quadrature"
            ),
            "minimum_r2": min_r2,
            "architecture": arch_diag,
            "penetration_note": (
                "normal_P values integrate the local interface window and are distinct from "
                "the lamellar half-period penetration metric"
            ),
        }, warnings=warnings,
    )


def profile_frame(profile: NormalProfileResult, timestep: int):
    """Return a tidy per-frame DataFrame without importing pandas at module load."""
    import pandas as pd
    data: Dict[str, Any] = {"timestep": int(timestep), "s": profile.s}
    for name, values in profile.mean.items():
        data[f"{name}_mean_surface"] = values
        data[f"{name}_std_surface"] = profile.std[name]
    return pd.DataFrame(data)
