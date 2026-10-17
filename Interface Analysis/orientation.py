"""Global lamellar normal n0 from the composition field, with S(q) cross-check.

Primary estimator: the dominant reciprocal-lattice mode of phi via a 3D FFT.
For a box with edges L, FFT bin (m1, m2, m3) corresponds to the physical
wavevector q = 2*pi*(m1/Lx, m2/Ly, m3/Lz). The normal is n0 = q*/|q*| for the
non-DC bin of largest spectral power (summed over the +/- Friedel pair).

The FFT is intrinsically limited to integer lattice modes, so |q*| is
quantized. We refine the magnitude with an intensity-weighted centroid over
the peak's immediate lattice neighbourhood, and we report the raw integer
mode for provenance. The DIRECTION is what n0 needs and it is robust.

Cross-checks and confidence:
  * n0 is compared with the S(q) pipeline's q_star magnitude (direction is
    not available from the isotropic S(q), only |q*|);
  * a secondary-peak ratio flags polydomain / poorly aligned samples;
  * an orientation "confidence" is the fraction of phi spectral power carried
    by the dominant Friedel pair.

This module never assumes lamellae are axis-aligned.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .composition import CompositionField
from .io_utils import LogFn


@dataclass
class OrientationResult:
    n0: np.ndarray                 # (3,) unit normal
    q_star_vec: np.ndarray         # (3,) physical wavevector of the dominant mode
    q_star_mag: float              # |q_star| (centroid-refined)
    q_star_mag_lattice: float      # |q_star| at the raw integer lattice mode
    lattice_mode: Tuple[int, int, int]
    L0: float                      # 2*pi / q_star_mag
    method: str
    confidence: float              # fraction of AC spectral power in the dominant pair
    secondary_ratio: float         # 2nd distinct pair power / 1st
    diagnostics: Dict[str, Any]
    warnings: List[str]


def _friedel_power(power: np.ndarray, mode: Tuple[int, int, int]) -> float:
    """Summed power of a mode and its Friedel mate -m (periodic indices)."""
    n = power.shape
    a = tuple(m % nn for m, nn in zip(mode, n))
    b = tuple((-m) % nn for m, nn in zip(mode, n))
    if a == b:
        return float(power[a])
    return float(power[a] + power[b])


def estimate_orientation_fft(
    comp: CompositionField,
    sq_q_star: Optional[float] = None,
    secondary_peak_warn_ratio: float = 0.5,
    qstar_crosscheck_tol: float = 0.15,
    log: Optional[LogFn] = None,
) -> OrientationResult:
    """Estimate n0 from the dominant reciprocal-lattice mode of phi."""
    phi = comp.phi
    grid = comp.grid
    L = grid.L
    n = phi.shape

    phi0 = phi - phi.mean()
    power = np.abs(np.fft.fftn(phi0)) ** 2
    total_ac_power = float(power.sum() - power[0, 0, 0])
    power[0, 0, 0] = 0.0

    # Integer mode index per axis (signed).
    freqs = [np.fft.fftfreq(nn, d=L[i] / nn) * L[i] for i, nn in enumerate(n)]  # = signed integer m
    m_axes = [np.rint(f).astype(int) for f in freqs]

    # Dominant bin.
    flat = int(np.argmax(power))
    k0 = np.unravel_index(flat, n)
    mode = tuple(int(m_axes[ax][k0[ax]]) for ax in range(3))
    q_vec_lat = 2.0 * np.pi * np.asarray(mode, dtype=float) / L
    q_mag_lat = float(np.linalg.norm(q_vec_lat))

    dominant_power = _friedel_power(power, mode)
    confidence = dominant_power / total_ac_power if total_ac_power > 0 else 0.0

    # Centroid refinement of |q| over the 3x3x3 lattice neighbourhood around
    # the peak (direction from the integer mode; magnitude sharpened). Only
    # bins on the same side as the peak are used to avoid cancelling the pair.
    q_mag_ref = _centroid_qmag(power, m_axes, L, k0, mode)

    n0 = q_vec_lat / q_mag_lat if q_mag_lat > 0 else np.array([0.0, 0.0, 1.0])
    # Canonical sign: first nonzero component positive (n0 and -n0 are equivalent).
    for c in n0:
        if abs(c) > 1e-9:
            if c < 0:
                n0 = -n0
                q_vec_lat = -q_vec_lat
            break

    q_star_mag = q_mag_ref if q_mag_ref > 0 else q_mag_lat
    L0 = 2.0 * np.pi / q_star_mag if q_star_mag > 0 else float("nan")

    # Secondary distinct pair (exclude the dominant pair's two bins).
    secondary_ratio, second_mode = _secondary_ratio(power, m_axes, mode, dominant_power)

    warnings: List[str] = []
    if confidence < 0.15:
        warnings.append(
            f"orientation confidence low ({confidence:.2f}): the dominant phi mode carries "
            f"<15% of composition power. Sample may be poorly ordered or polydomain; "
            f"n0={np.round(n0, 3).tolist()} is unreliable."
        )
    if secondary_ratio > secondary_peak_warn_ratio:
        warnings.append(
            f"secondary phi mode {second_mode} carries {secondary_ratio:.2f} of the dominant "
            f"mode's power (> {secondary_peak_warn_ratio}): possible multiple lamellar "
            "orientations (polydomain). A single global n0 may not describe the whole box."
        )

    method = "fft_phi_dominant_mode"
    qstar_rel_diff = None
    if sq_q_star is not None and np.isfinite(sq_q_star) and sq_q_star > 0:
        qstar_rel_diff = abs(q_star_mag - sq_q_star) / sq_q_star
        if qstar_rel_diff > qstar_crosscheck_tol:
            warnings.append(
                f"FFT |q*|={q_star_mag:.4f} disagrees with S(q) q_star={sq_q_star:.4f} "
                f"by {100 * qstar_rel_diff:.1f}% (> {100 * qstar_crosscheck_tol:.0f}%). "
                "FFT sets the direction; check grid/box or S(q) window."
            )
        method = "fft_phi_dominant_mode+sq_crosscheck"

    diagnostics: Dict[str, Any] = {
        "lattice_mode": list(mode),
        "q_star_vec": q_vec_lat.tolist(),
        "q_star_mag_lattice": q_mag_lat,
        "q_star_mag_centroid": q_star_mag,
        "L0_from_qstar": L0,
        "confidence_power_fraction": confidence,
        "secondary_pair_ratio": secondary_ratio,
        "secondary_mode": list(second_mode) if second_mode else None,
        "sq_q_star": sq_q_star,
        "sq_qstar_rel_diff": qstar_rel_diff,
        "note": ("FFT gives lamellar DIRECTION robustly; |q*| is lattice-quantized and "
                 "centroid-refined. S(q) provides an independent |q*| magnitude only."),
    }
    if log:
        log(f"n0={np.round(n0, 4).tolist()} (mode {mode}), |q*|={q_star_mag:.4f}, "
            f"L0={L0:.3f}, confidence={confidence:.3f}, secondary={secondary_ratio:.3f}")
        for w in warnings:
            log(f"WARN: {w}")

    return OrientationResult(
        n0=n0, q_star_vec=q_vec_lat, q_star_mag=q_star_mag,
        q_star_mag_lattice=q_mag_lat, lattice_mode=mode, L0=L0,
        method=method, confidence=confidence, secondary_ratio=secondary_ratio,
        diagnostics=diagnostics, warnings=warnings,
    )


def _centroid_qmag(power, m_axes, L, k0, mode) -> float:
    """Intensity-weighted |q| over lattice bins adjacent to the peak (same hemisphere)."""
    n = power.shape
    num = 0.0
    den = 0.0
    for dz0 in (-1, 0, 1):
        for dz1 in (-1, 0, 1):
            for dz2 in (-1, 0, 1):
                idx = ((k0[0] + dz0) % n[0], (k0[1] + dz1) % n[1], (k0[2] + dz2) % n[2])
                m = np.array([m_axes[0][idx[0]], m_axes[1][idx[1]], m_axes[2][idx[2]]], dtype=float)
                if np.dot(m, np.asarray(mode, float)) <= 0:  # keep same hemisphere as the peak
                    continue
                q = 2.0 * np.pi * m / L
                w = power[idx]
                num += w * np.linalg.norm(q)
                den += w
    return float(num / den) if den > 0 else 0.0


def _secondary_ratio(power, m_axes, mode, dominant_power) -> Tuple[float, Optional[Tuple[int, int, int]]]:
    """Ratio of the strongest OTHER distinct Friedel pair's power to the dominant's."""
    n = power.shape
    p = power.copy()
    for mm in (mode, tuple(-x for x in mode)):
        p[tuple(x % nn for x, nn in zip(mm, n))] = 0.0
    if dominant_power <= 0:
        return 0.0, None
    # search a modest number of top bins for the next distinct direction
    order = np.argsort(p.ravel())[::-1][:32]
    mode_dir = np.asarray(mode, float)
    mode_dir /= np.linalg.norm(mode_dir)
    for flat in order:
        k = np.unravel_index(int(flat), n)
        m = tuple(int(m_axes[ax][k[ax]]) for ax in range(3))
        if m == (0, 0, 0):
            continue
        mdir = np.asarray(m, float)
        if np.linalg.norm(mdir) == 0:
            continue
        mdir /= np.linalg.norm(mdir)
        if abs(float(np.dot(mdir, mode_dir))) > 0.98:  # same axis (a harmonic) — skip
            continue
        return _friedel_power(power, m) / dominant_power, m
    return 0.0, None


def resolve_orientation(
    comp: CompositionField,
    cfg: Dict[str, Any],
    sq_q_star: Optional[float] = None,
    log: Optional[LogFn] = None,
) -> OrientationResult:
    """Dispatch on cfg['n0_mode']: 'fixed', 'fft', or 'auto' (fft + S(q) crosscheck)."""
    mode = cfg.get("n0_mode", "auto")
    if mode == "fixed":
        v = np.asarray(cfg["n0_fixed"], dtype=float)
        n0 = v / np.linalg.norm(v)
        for c in n0:
            if abs(c) > 1e-9:
                if c < 0:
                    n0 = -n0
                break
        # still run FFT to report confidence/L0 for provenance
        fft_res = estimate_orientation_fft(
            comp, sq_q_star=sq_q_star,
            secondary_peak_warn_ratio=float(cfg["secondary_peak_warn_ratio"]),
            qstar_crosscheck_tol=float(cfg["qstar_crosscheck_tol"]), log=None,
        )
        angle = np.degrees(np.arccos(np.clip(abs(float(np.dot(n0, fft_res.n0))), 0, 1)))
        warnings = [f"n0 fixed by config to {np.round(n0, 4).tolist()}; "
                    f"FFT estimate would be {np.round(fft_res.n0, 4).tolist()} "
                    f"({angle:.1f} deg away)."]
        if log:
            log(warnings[0])
        return OrientationResult(
            n0=n0, q_star_vec=n0 * fft_res.q_star_mag, q_star_mag=fft_res.q_star_mag,
            q_star_mag_lattice=fft_res.q_star_mag_lattice, lattice_mode=fft_res.lattice_mode,
            L0=fft_res.L0, method="fixed_from_config", confidence=fft_res.confidence,
            secondary_ratio=fft_res.secondary_ratio,
            diagnostics={**fft_res.diagnostics, "fixed_vs_fft_angle_deg": angle},
            warnings=warnings + fft_res.warnings,
        )

    crosscheck = sq_q_star if mode == "auto" else None
    return estimate_orientation_fft(
        comp, sq_q_star=crosscheck,
        secondary_peak_warn_ratio=float(cfg["secondary_peak_warn_ratio"]),
        qstar_crosscheck_tol=float(cfg["qstar_crosscheck_tol"]), log=log,
    )
