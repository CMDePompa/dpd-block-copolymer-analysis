"""Intrinsic interface width from LOCALLY ALIGNED tanh profile fits.

The laterally averaged composition profile is broadened by waviness (the
interface moves under the average), so it must NOT be used as the width.
Instead, for every sampled column and every valid interface:

  1. the local zero crossing h_j(u, v) is taken from interface detection;
  2. phi is sampled along the column (global normal direction) on a window
     s in [-s_max, s_max] CENTERED on the local crossing;
  3. each profile is fit with  phi(s) = a + b * tanh((s - s0) / lam),
     where the offset a accommodates asymmetric bulk composition plateaus;
  4. the aligned profiles are also averaged (per interface and pooled) and
     the average is fit once — the "aligned mean profile" width;
  5. for contrast, the NAIVE laterally averaged (uncentered) profile is fit
     too; the difference between (4) and (5) is the waviness-broadening
     artifact, reported explicitly.

Width definitions (all reported):
  * lam       — tanh length from the fit;
  * w_10_90   — 10-90% width = 2 * atanh(0.8) * lam  (= 2.19722 * lam);
  * lam_obliq — obliquity-corrected lam * cos(theta_local): profiles run
    along the GLOBAL normal, so a locally tilted interface is crossed
    obliquely and its apparent width is inflated by 1/cos(theta). cos(theta)
    comes from the height-field slope at the same column.

Caveat recorded in results: the measured width includes broadening from the
density smoothing sigma (and, weakly, CIC deposition). Widths are therefore
"apparent field widths at the configured sigma"; the calibration mode
quantifies d(width)/d(sigma). An approximate quadrature deconvolution
w_1090_deconv = sqrt(max(w_1090^2 - (2.5631*sigma)^2, 0)) is reported as a
rough intrinsic estimate (2.5631 = 10-90 width of a unit Gaussian PSF).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import curve_fit

from .interface_detection import COL_GOOD, InterfaceSet
from .io_utils import LogFn
from .resample import SlabField

W1090_PER_LAM = 2.0 * np.arctanh(0.8)      # = 2.19722, conversion lam -> 10-90 width
GAUSS_1090 = 2.5631                        # 10-90 width of a unit-sigma Gaussian CDF


def _tanh_model(s: np.ndarray, a: float, b: float, s0: float, lam: float) -> np.ndarray:
    return a + b * np.tanh((s - s0) / lam)


@dataclass
class WidthResult:
    lam_values: np.ndarray            # all successful local fits (raw, uncorrected)
    lam_corrected: np.ndarray         # obliquity-corrected (same order)
    r2_values: np.ndarray
    mean_lam: float
    std_lam: float
    median_lam: float
    iqr_lam: float
    mean_lam_corrected: float
    mean_w1090: float                 # from corrected lam
    w1090_deconv_sigma: float         # approx. smoothing-deconvolved 10-90 width
    failed_fraction: float
    n_attempted: int
    n_succeeded: int
    width_maps: List[np.ndarray]      # per-interface (nu, nv) maps of corrected lam (NaN off-sample)
    s_grid: np.ndarray                # common profile abscissa
    aligned_mean_profile: np.ndarray  # pooled aligned <phi>(s), rising-orientation
    aligned_fit: Dict[str, float]     # fit of the aligned mean profile
    naive_profile_w: np.ndarray       # w grid of the naive laterally averaged profile
    naive_profile: np.ndarray         # <phi>(w) averaged over all columns (uncentered)
    naive_fit: Dict[str, float]       # fit of the naive profile around one interface
    waviness_broadening_ratio: float  # naive lam / aligned lam (>= 1 when waviness broadens)
    smoothing_sigma: float
    diagnostics: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)


def _fit_profile(s: np.ndarray, y: np.ndarray, lam0: float, s_max: float
                 ) -> Optional[Tuple[float, float, float, float, float]]:
    """Fit a + b*tanh((s-s0)/lam); returns (a, b, s0, lam, r2) or None."""
    if len(s) < 8 or not np.all(np.isfinite(y)):
        return None
    a0 = 0.5 * (y[-1] + y[0])
    b0 = 0.5 * (y[-1] - y[0])
    if abs(b0) < 1e-6:
        return None
    try:
        popt, _ = curve_fit(
            _tanh_model, s, y,
            p0=[a0, b0, 0.0, max(lam0, 1e-3)],
            bounds=([-1.5, -2.5, -0.5 * s_max, 1e-3],
                    [1.5, 2.5, 0.5 * s_max, 4.0 * s_max]),
            maxfev=400,
        )
    except (RuntimeError, ValueError):
        return None
    resid = y - _tanh_model(s, *popt)
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot <= 0:
        return None
    r2 = 1.0 - float(np.sum(resid ** 2)) / ss_tot
    a, b, s0, lam = (float(p) for p in popt)
    # a fit that pegs lam at the upper bound is not a measurement
    if lam >= 3.9 * s_max:
        return None
    return a, b, s0, lam, r2


def compute_width(
    slab: SlabField,
    iset: InterfaceSet,
    cfg: Dict[str, Any],
    smoothing_sigma: float,
    log: Optional[LogFn] = None,
) -> WidthResult:
    """Local tanh widths for all valid interfaces of one frame's slab."""
    phi = slab.phi_uvw
    nu, nv, nw = phi.shape
    w = slab.w
    dw = slab.dw
    L0 = slab.L0
    W = slab.W

    s_max = float(cfg["width_fit_halfwidth_fraction"]) * L0
    min_r2 = float(cfg["width_min_r2"])
    max_cols = int(cfg["width_fit_max_columns"])
    do_obliq = bool(cfg["obliquity_correction"])
    clean_columns = iset.column_class == COL_GOOD

    # Regular column subsampling (regular grid keeps the width MAP plottable).
    total_cols = nu * nv
    stride = max(1, int(np.ceil(np.sqrt(total_cols / max_cols))))
    iu_sel = np.arange(0, nu, stride)
    iv_sel = np.arange(0, nv, stride)

    n_half = max(4, int(round(s_max / dw)))
    s_grid = np.arange(-n_half, n_half + 1) * dw

    lam_list: List[float] = []
    lamc_list: List[float] = []
    r2_list: List[float] = []
    width_maps = [np.full((nu, nv), np.nan) for _ in range(iset.n_interfaces)]
    aligned_sum = np.zeros_like(s_grid)
    aligned_cnt = 0
    n_attempted = 0
    n_succeeded = 0

    # Pre-compute per-interface height-field slopes for obliquity correction.
    slopes: List[Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]] = []
    for j in range(iset.n_interfaces):
        hj = np.where(clean_columns, iset.h[j], np.nan)
        if np.mean(np.isfinite(hj)) < 0.25:
            slopes.append(None)
            continue
        from .height_fields import fill_gaps, gradient_valid_mask
        filled, _ = fill_gaps(hj)
        gu, gv = np.gradient(filled, slab.du, slab.dv, edge_order=1)
        slopes.append((gu, gv, gradient_valid_mask(np.isfinite(hj))))

    lam0 = max(2 * dw, 0.1 * L0)  # initial guess
    for j in range(iset.n_interfaces):
        sgn = float(iset.plane_type[j])   # +1 rising, -1 falling
        for iu in iu_sel:
            for iv in iv_sel:
                if not clean_columns[iu, iv]:
                    continue
                hj = iset.h[j, iu, iv]
                if not np.isfinite(hj):
                    continue
                # periodic sampling of the column at s_grid around the crossing
                idx = (hj - w[0] + s_grid) / dw
                if slab.w_periodic:
                    idx_work = np.mod(idx, nw)
                    i0 = np.floor(idx_work).astype(int)
                    frac = idx_work - i0
                    i1 = (i0 + 1) % nw
                else:
                    # Do not join unrelated endpoints of a non-closing
                    # oblique column. Edge interfaces are excluded.
                    if float(np.min(idx)) < 0.0 or float(np.max(idx)) > nw - 1:
                        continue
                    i0 = np.floor(idx).astype(int)
                    i0 = np.clip(i0, 0, nw - 1)
                    frac = idx - i0
                    i1 = np.minimum(i0 + 1, nw - 1)
                cos_t = 1.0
                if do_obliq:
                    if slopes[j] is None:
                        continue
                    gu, gv, slope_valid = slopes[j]
                    if not slope_valid[iu, iv]:
                        continue
                    cos_t = 1.0 / np.sqrt(1.0 + gu[iu, iv] ** 2 + gv[iu, iv] ** 2)
                n_attempted += 1
                col = phi[iu, iv, :]
                prof = (1.0 - frac) * col[i0] + frac * col[i1]

                fit = _fit_profile(s_grid, prof, lam0, s_max)
                if fit is None:
                    continue
                a, b, s0, lam, r2 = fit
                if r2 < min_r2 or b * sgn <= 0:
                    continue
                n_succeeded += 1
                lam_list.append(lam)
                lamc_list.append(lam * cos_t)
                r2_list.append(r2)
                width_maps[j][iu, iv] = lam * cos_t
                # Accumulate the aligned profile in rising orientation by
                # REVERSING THE s AXIS for falling interfaces (s -> -s), which
                # preserves the composition offset 'a'; negating phi would
                # flip and cancel the offset in the average.
                aligned_sum += prof if sgn > 0 else prof[::-1]
                aligned_cnt += 1

    warnings: List[str] = []
    lam_arr = np.asarray(lam_list)
    lamc_arr = np.asarray(lamc_list)
    r2_arr = np.asarray(r2_list)
    failed_fraction = 1.0 - (n_succeeded / n_attempted) if n_attempted else 1.0
    if failed_fraction > 0.3:
        warnings.append(
            f"{100 * failed_fraction:.0f}% of local width fits failed "
            f"(R^2 < {min_r2} or no convergence): profiles are noisy or the fit "
            "window overlaps neighbouring interfaces; widths may be biased toward "
            "the cleanest interface patches."
        )

    # Aligned mean profile + its single fit.
    aligned_mean = aligned_sum / aligned_cnt if aligned_cnt else np.full_like(s_grid, np.nan)
    aligned_fit = _fit_dict(_fit_profile(s_grid, aligned_mean, lam0, s_max)) \
        if aligned_cnt else _fit_dict(None)

    # Naive laterally averaged profile over the whole slab (uncentered).
    naive = phi.mean(axis=(0, 1))
    naive_fit = _naive_fit(naive, w, iset, s_max, dw, lam0, W,
                           periodic=slab.w_periodic)

    aligned_lam = aligned_fit.get("lam", float("nan"))
    naive_lam = naive_fit.get("lam", float("nan"))
    broadening = float(naive_lam / aligned_lam) if (np.isfinite(naive_lam)
                                                    and np.isfinite(aligned_lam)
                                                    and aligned_lam > 0) else float("nan")

    mean_lamc = float(np.mean(lamc_arr)) if lamc_arr.size else float("nan")
    mean_w1090 = W1090_PER_LAM * mean_lamc
    w1090_deconv = float(np.sqrt(max(mean_w1090 ** 2 - (GAUSS_1090 * smoothing_sigma) ** 2, 0.0))) \
        if np.isfinite(mean_w1090) else float("nan")

    q1, q3 = (np.percentile(lam_arr, [25, 75]) if lam_arr.size else (np.nan, np.nan))
    result = WidthResult(
        lam_values=lam_arr, lam_corrected=lamc_arr, r2_values=r2_arr,
        mean_lam=float(np.mean(lam_arr)) if lam_arr.size else float("nan"),
        std_lam=float(np.std(lam_arr)) if lam_arr.size else float("nan"),
        median_lam=float(np.median(lam_arr)) if lam_arr.size else float("nan"),
        iqr_lam=float(q3 - q1),
        mean_lam_corrected=mean_lamc,
        mean_w1090=mean_w1090,
        w1090_deconv_sigma=w1090_deconv,
        failed_fraction=failed_fraction,
        n_attempted=n_attempted, n_succeeded=n_succeeded,
        width_maps=width_maps, s_grid=s_grid,
        aligned_mean_profile=aligned_mean, aligned_fit=aligned_fit,
        naive_profile_w=w.copy(), naive_profile=naive, naive_fit=naive_fit,
        waviness_broadening_ratio=broadening,
        smoothing_sigma=float(smoothing_sigma),
        diagnostics={
            "s_max": s_max, "column_stride": stride,
            "n_columns_sampled": int(len(iu_sel) * len(iv_sel)),
            "clean_column_fraction": float(np.mean(clean_columns)),
            "excluded_bad_column_fraction": float(1.0 - np.mean(clean_columns)),
            "w_periodic": slab.w_periodic,
            "w1090_per_lam": W1090_PER_LAM,
            "deconvolution_note": ("w1090_deconv = sqrt(w1090^2 - (2.5631*sigma)^2); "
                                   "approximate — ignores CIC/interpolation broadening"),
        },
        warnings=warnings,
    )
    if log:
        log(f"width: {n_succeeded}/{n_attempted} fits ok; lam={result.mean_lam:.3f} "
            f"(corrected {mean_lamc:.3f}), 10-90={mean_w1090:.3f}, "
            f"naive/aligned broadening={broadening:.2f}")
        for msg in warnings:
            log(f"WARN: {msg}")
    return result


def _fit_dict(fit) -> Dict[str, float]:
    if fit is None:
        return {"a": float("nan"), "b": float("nan"), "s0": float("nan"),
                "lam": float("nan"), "r2": float("nan")}
    a, b, s0, lam, r2 = fit
    return {"a": a, "b": b, "s0": s0, "lam": lam, "r2": r2}


def _naive_fit(naive: np.ndarray, w: np.ndarray, iset: InterfaceSet,
               s_max: float, dw: float, lam0: float, W: float,
               periodic: bool = True) -> Dict[str, float]:
    """Fit the uncentered laterally averaged profile around the best interface."""
    if iset.n_interfaces == 0:
        return _fit_dict(None)
    nw = len(w)
    n_half = max(4, int(round(s_max / dw)))
    s = np.arange(-n_half, n_half + 1) * dw
    j = None
    center = float("nan")
    clean = iset.column_class == COL_GOOD
    for candidate in np.argsort(iset.valid_fraction)[::-1]:
        vals = iset.h[int(candidate)][clean & np.isfinite(iset.h[int(candidate)])]
        c = float(np.mean(vals)) if vals.size else float("nan")
        if not np.isfinite(c):
            continue
        idx_test = (c - w[0] + s) / dw
        if periodic or (float(np.min(idx_test)) >= 0.0 and
                        float(np.max(idx_test)) <= nw - 1):
            j = int(candidate)
            center = float(c)
            break
    if j is None:
        return _fit_dict(None)
    idx = (center - w[0] + s) / dw
    if periodic:
        idx = np.mod(idx, nw)
        i0 = np.floor(idx).astype(int)
        frac = idx - i0
        i1 = (i0 + 1) % nw
    else:
        i0 = np.clip(np.floor(idx).astype(int), 0, nw - 1)
        frac = idx - i0
        i1 = np.minimum(i0 + 1, nw - 1)
    prof = (1 - frac) * naive[i0] + frac * naive[i1]
    if iset.plane_type[j] < 0:      # falling: reverse the s axis (preserves offset)
        prof = prof[::-1]
    return _fit_dict(_fit_profile(s, prof, lam0, s_max))
