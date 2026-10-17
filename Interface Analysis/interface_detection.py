"""Detect A/B interfaces as zero crossings of phi along n0, and label them.

Per column (fixed u, v) the composition phi(w) is a periodic 1-D signal. Zero
crossings are found with a Schmitt-trigger (hysteresis) filter that rejects
sub-threshold noise wiggles, and each crossing is located to sub-sample
accuracy by linear interpolation. Crossings carry a direction: rising
(-> A, phi - to +) or falling (-> B, phi + to -). Rising and falling
interfaces are labeled on SEPARATE sub-lattices spaced by L0, which correctly
handles asymmetric lamellae (unequal A and B layer thicknesses) where rising
and falling planes are not L0/2 apart.

When the normal window is a verified box-periodic closure, its boundary is
handled by a wrapped L0/2 guard. Otherwise the column is bounded and an L0/2
edge guard is discarded; unrelated endpoints are never joined.

Columns are classified (good / missing / extra / ambiguous / close-pair /
low-amplitude) to produce an operational defect map and a bad-column fraction.
This is meaningful for a mostly-lamellar morphology; it is not a universal
defect definition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .io_utils import LogFn
from .resample import SlabField

# Column classification codes (also used for the defect map).
COL_GOOD = 0
COL_MISSING = 1        # fewer crossings than expected (merged/broken layer)
COL_EXTRA = 2          # more crossings than expected (bridge/island)
COL_AMBIGUOUS = 3      # a crossing could not be confidently assigned to a plane
COL_CLOSE_PAIR = 4     # two crossings closer than close_pair_fraction * L0/2
COL_LOW_AMP = 5        # |phi| never exceeds hysteresis (no confident phase)

COL_CLASS_NAMES = {
    COL_GOOD: "good",
    COL_MISSING: "missing_crossing",
    COL_EXTRA: "extra_crossing",
    COL_AMBIGUOUS: "ambiguous",
    COL_CLOSE_PAIR: "close_pair",
    COL_LOW_AMP: "low_amplitude",
}


@dataclass
class InterfaceSet:
    """Labeled interfaces with per-interface height fields and column diagnostics."""

    h: np.ndarray                  # (M, nu, nv) crossing position along w; NaN where missing
    plane_type: np.ndarray         # (M,) +1 rising (->A), -1 falling (->B)
    plane_ideal: np.ndarray        # (M,) ideal plane position along w
    plane_mean: np.ndarray         # (M,) mean detected height (over valid columns)
    valid_fraction: np.ndarray     # (M,) fraction of columns with a valid crossing
    column_class: np.ndarray       # (nu, nv) int code from COL_*
    expected_crossings: int
    crossing_count: np.ndarray     # (nu, nv) number of confident crossings per column
    bad_column_fraction: float
    class_fractions: Dict[str, float]
    n_interfaces: int
    n_valid_interfaces: int
    W: float                       # sampled normal extent
    w_periodic: bool               # whether first/last w samples are neighbours
    diagnostics: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)


def find_crossings_1d(
    col: np.ndarray,
    w: np.ndarray,
    hysteresis: float,
    dw: float,
    periodic: bool = True,
    guard: Optional[float] = None,
) -> List[Tuple[float, int]]:
    """Zero crossings of a 1-D column with Schmitt-trigger hysteresis.

    Returns a list of (w_position, direction) with direction +1 (rising, ->A)
    or -1 (falling, ->B). Positions are linearly interpolated. When periodic,
    the column is padded with a wrapped guard so boundary crossings and the
    initial confirmed sign are handled without wrap bookkeeping.
    """
    n = len(col)
    if n < 2:
        return []
    if periodic:
        gpad = int(np.ceil((guard if guard is not None else 0.0) / dw)) if dw > 0 else 0
        gpad = max(1, min(gpad, n))
        col_ext = np.concatenate([col[-gpad:], col, col[:gpad]])
        w_ext = np.concatenate([w[0] - dw * np.arange(gpad, 0, -1), w, w[-1] + dw * np.arange(1, gpad + 1)])
        # Keep exactly one period [w0 - dw/2, w0 + W - dw/2); W = n*dw so this
        # is [w0 - dw/2, w[-1] + dw/2). Every crossing on the circle appears once.
        w_lo, w_hi = w[0] - 0.5 * dw, w[-1] + 0.5 * dw
    else:
        col_ext, w_ext = col, w
        w_lo, w_hi = w[0] - 1e9, w[-1] + 1e9

    # Initial confirmed sign: first sample exceeding the hysteresis band.
    S = 0
    for value in col_ext:
        if value > hysteresis:
            S = 1
            break
        if value < -hysteresis:
            S = -1
            break
    if S == 0:
        return []  # low-amplitude column: never leaves the noise band

    crossings: List[Tuple[float, int]] = []
    last_zero: Optional[float] = None
    last_dir = 0
    for i in range(1, len(col_ext)):
        a, b = col_ext[i - 1], col_ext[i]
        if a < 0.0 <= b or (a < 0.0 and b > 0.0):
            t = (0.0 - a) / (b - a) if b != a else 0.0
            last_zero = w_ext[i - 1] + t * (w_ext[i] - w_ext[i - 1])
            last_dir = 1
        elif a > 0.0 >= b or (a > 0.0 and b < 0.0):
            t = (0.0 - a) / (b - a) if b != a else 0.0
            last_zero = w_ext[i - 1] + t * (w_ext[i] - w_ext[i - 1])
            last_dir = -1
        # Confirm a crossing only when the opposite threshold is exceeded.
        if S == -1 and b > hysteresis and last_zero is not None and last_dir == 1:
            crossings.append((float(last_zero), 1))
            S = 1
            last_zero = None
        elif S == 1 and b < -hysteresis and last_zero is not None and last_dir == -1:
            crossings.append((float(last_zero), -1))
            S = -1
            last_zero = None

    # Keep crossings inside the central window (deduplicates the wrap guard).
    return [(wc, d) for (wc, d) in crossings if w_lo <= wc < w_hi]


def _circular_mean_mod(positions: np.ndarray, period: float) -> float:
    """Mean of positions modulo *period*, returned in [0, period)."""
    if len(positions) == 0:
        return 0.0
    theta = 2.0 * np.pi * (np.asarray(positions) % period) / period
    ang = np.arctan2(np.mean(np.sin(theta)), np.mean(np.cos(theta)))
    return float((ang % (2.0 * np.pi)) / (2.0 * np.pi) * period)


def _wrap_pm(x: np.ndarray, period: float) -> np.ndarray:
    """Wrap x into the symmetric interval [-period/2, period/2)."""
    return (np.asarray(x) + period / 2.0) % period - period / 2.0


def _build_periodic_planes(theta: float, L0: float, n_periods: int, w0: float, W: float, ptype: int
                           ) -> List[Tuple[float, int]]:
    """Exactly n_periods ideal planes of one type on the periodic circle [w0, w0+W).

    Planes sit at theta + k*L0 (k = 0..n_periods-1) mapped into the window by
    modulo W. Because W = n_periods*L0 these are the n_periods distinct planes
    on the circle — there is no window boundary, so every interface is treated
    as interior (no edge loss under strong waviness).
    """
    if L0 <= 0 or n_periods < 1:
        return []
    planes = []
    for k in range(n_periods):
        c = w0 + ((theta + k * L0 - w0) % W)
        planes.append((c, ptype))
    return planes


def _build_linear_planes(theta: float, L0: float, w_lo: float, w_hi: float,
                         ptype: int) -> List[Tuple[float, int]]:
    """Ideal planes inside a bounded, non-periodic normal window."""
    if L0 <= 0 or w_hi <= w_lo:
        return []
    k0 = int(np.ceil((w_lo - theta) / L0))
    k1 = int(np.floor((w_hi - theta) / L0))
    return [(theta + k * L0, ptype) for k in range(k0, k1 + 1)]


def detect_interfaces(
    slab: SlabField,
    cfg: Dict[str, Any],
    log: Optional[LogFn] = None,
) -> InterfaceSet:
    """Detect and label interfaces across the whole slab; build height fields."""
    phi = slab.phi_uvw
    nu, nv, nw = phi.shape
    w = slab.w
    dw = slab.dw
    L0 = slab.L0
    hyst = float(cfg["phi_hysteresis"])
    assign_tol = float(cfg["assignment_tolerance"]) * (L0 / 2.0)
    close_frac = float(cfg.get("close_pair_fraction", 0.30)) * (L0 / 2.0)
    min_amp = float(cfg["min_column_amplitude"])
    guard = 0.5 * L0

    # 1. Per-column crossings.
    col_cross: Dict[Tuple[int, int], List[Tuple[float, int]]] = {}
    rise_pos: List[float] = []
    fall_pos: List[float] = []
    low_amp = np.zeros((nu, nv), dtype=bool)
    for iu in range(nu):
        for iv in range(nv):
            col = phi[iu, iv, :]
            if np.max(np.abs(col)) < min_amp:
                low_amp[iu, iv] = True
            cs = find_crossings_1d(
                col, w, hyst, dw, periodic=slab.w_periodic,
                guard=guard if slab.w_periodic else None,
            )
            if not slab.w_periodic:
                # Edge guard makes every retained crossing safe for the
                # default +/-L0/2 penetration window and avoids interpreting
                # a truncated phase at either endpoint as a defect.
                cs = [(wc, d) for wc, d in cs
                      if w[0] + guard <= wc <= w[-1] - guard]
            col_cross[(iu, iv)] = cs
            for wc, d in cs:
                (rise_pos if d == 1 else fall_pos).append(wc)

    warnings: List[str] = []
    if not rise_pos and not fall_pos:
        warnings.append("no interface crossings detected in any column; "
                        "the slab may be single-phase, mislabeled, or below hysteresis.")
        if log:
            log(f"WARN: {warnings[-1]}")
        return _empty_interface_set(nu, nv, low_amp, warnings, slab.W,
                                    slab.w_periodic)

    # 2. Global sub-lattice phases (rising and falling handled separately) and
    #    ideal planes of each type on either a verified periodic circle or a
    #    guarded bounded interval.
    W = slab.W
    w0 = float(w[0])
    theta_r = _circular_mean_mod(np.array(rise_pos), L0) if rise_pos else None
    theta_f = _circular_mean_mod(np.array(fall_pos), L0) if fall_pos else None
    planes: List[Tuple[float, int]] = []
    if slab.w_periodic:
        if theta_r is not None:
            planes += _build_periodic_planes(theta_r, L0, slab.n_periods, w0, W, +1)
        if theta_f is not None:
            planes += _build_periodic_planes(theta_f, L0, slab.n_periods, w0, W, -1)
    else:
        w_lo = float(w[0] + guard)
        w_hi = float(w[-1] - guard)
        if theta_r is not None:
            planes += _build_linear_planes(theta_r, L0, w_lo, w_hi, +1)
        if theta_f is not None:
            planes += _build_linear_planes(theta_f, L0, w_lo, w_hi, -1)
    planes.sort(key=lambda p: p[0])
    M = len(planes)
    plane_ideal = np.array([p[0] for p in planes], dtype=float)
    plane_type = np.array([p[1] for p in planes], dtype=int)

    # 3. Assign crossings to planes by signed CIRCULAR offset; build heights.
    #    Height = ideal plane position + circular offset (locally unwrapped),
    #    which is free of window-boundary loss under strong waviness.
    h = np.full((M, nu, nv), np.nan, dtype=float)
    crossing_count = np.zeros((nu, nv), dtype=np.int32)
    column_class = np.zeros((nu, nv), dtype=np.int32)

    for (iu, iv), cs in col_cross.items():
        if low_amp[iu, iv]:
            column_class[iu, iv] = COL_LOW_AMP
            continue
        crossing_count[iu, iv] = len(cs)
        assigned: Dict[int, List[float]] = {}     # plane index -> list of heights
        unassigned = 0
        for wc, d in cs:
            same = np.where(plane_type == d)[0]
            if same.size == 0:
                unassigned += 1
                continue
            offsets = (_wrap_pm(wc - plane_ideal[same], W) if slab.w_periodic
                       else wc - plane_ideal[same])
            k = int(np.argmin(np.abs(offsets)))
            j = int(same[k])
            if abs(offsets[k]) <= assign_tol:
                assigned.setdefault(j, []).append(plane_ideal[j] + float(offsets[k]))
            else:
                unassigned += 1
        # Fill heights (nearest-to-ideal wins if a plane got multiple crossings).
        multi = False
        for j, vals in assigned.items():
            if len(vals) > 1:
                multi = True
            h[j, iu, iv] = min(vals, key=lambda x: abs(x - plane_ideal[j]))

        # Close-pair detection.
        close_pair = False
        if len(cs) >= 2:
            wpos = np.sort([c[0] for c in cs])
            gaps = np.diff(wpos)
            min_gap = float(np.min(gaps))
            if slab.w_periodic:
                min_gap = min(min_gap, float((wpos[0] + W) - wpos[-1]))
            if min_gap < close_frac:
                close_pair = True

        # Crossing count must take priority.  Otherwise an extra crossing is
        # necessarily multi-assigned/unassigned first and COL_EXTRA can never
        # be reached.
        if len(cs) < M:
            column_class[iu, iv] = COL_MISSING
        elif len(cs) > M:
            column_class[iu, iv] = COL_EXTRA
        elif unassigned > 0 or multi:
            column_class[iu, iv] = COL_AMBIGUOUS
        elif close_pair:
            column_class[iu, iv] = COL_CLOSE_PAIR
        else:
            column_class[iu, iv] = COL_GOOD

    # 4. Per-interface statistics.
    valid_fraction = np.array([np.mean(np.isfinite(h[j])) for j in range(M)], dtype=float)
    plane_mean = np.array([np.nanmean(h[j]) if np.any(np.isfinite(h[j])) else np.nan
                           for j in range(M)], dtype=float)
    n_valid_interfaces = int(np.sum(valid_fraction >= 0.5))

    class_counts = {name: int(np.sum(column_class == code))
                    for code, name in COL_CLASS_NAMES.items()}
    total_cols = nu * nv
    class_fractions = {k: v / total_cols for k, v in class_counts.items()}
    bad_column_fraction = 1.0 - class_fractions["good"]

    if bad_column_fraction > 0.5:
        warnings.append(
            f"{100 * bad_column_fraction:.0f}% of columns are non-ideal "
            f"(expected {M} crossings/column). The height-field picture is weak here; "
            "inspect the defect map and topology before trusting waviness/spacing."
        )
    med_count = int(np.median(crossing_count[~low_amp])) if np.any(~low_amp) else 0
    if M > 0 and med_count not in (M, M - 1, M + 1):
        warnings.append(
            f"median crossings/column ({med_count}) differs from the number of ideal "
            f"planes ({M}); L0 or n_periods may be off — check the S(q) crosscheck."
        )

    diagnostics: Dict[str, Any] = {
        "n_planes": M,
        "n_rising": int(np.sum(plane_type == 1)),
        "n_falling": int(np.sum(plane_type == -1)),
        "theta_rising": theta_r,
        "theta_falling": theta_f,
        "assignment_tolerance_w": assign_tol,
        "close_pair_threshold_w": close_frac,
        "hysteresis": hyst,
        "expected_crossings_per_column": M,
        "median_crossings_per_column": med_count,
        "class_counts": class_counts,
        "mean_valid_fraction": float(np.mean(valid_fraction)) if M else 0.0,
        "w_periodic": slab.w_periodic,
    }
    if log:
        log(f"interfaces: {M} planes ({diagnostics['n_rising']}R/{diagnostics['n_falling']}F), "
            f"{n_valid_interfaces} valid; bad-column fraction {bad_column_fraction:.3f} "
            f"(good {class_fractions['good']:.2f}, missing {class_fractions['missing_crossing']:.2f}, "
            f"extra {class_fractions['extra_crossing']:.2f}, ambiguous {class_fractions['ambiguous']:.2f})")
        for wmsg in warnings:
            log(f"WARN: {wmsg}")

    return InterfaceSet(
        h=h, plane_type=plane_type, plane_ideal=plane_ideal, plane_mean=plane_mean,
        valid_fraction=valid_fraction, column_class=column_class,
        expected_crossings=M, crossing_count=crossing_count,
        bad_column_fraction=bad_column_fraction, class_fractions=class_fractions,
        n_interfaces=M, n_valid_interfaces=n_valid_interfaces,
        W=W, w_periodic=slab.w_periodic,
        diagnostics=diagnostics, warnings=warnings,
    )


def _empty_interface_set(nu, nv, low_amp, warnings, W=0.0,
                         w_periodic=False) -> InterfaceSet:
    column_class = np.where(low_amp, COL_LOW_AMP, COL_MISSING).astype(np.int32)
    class_counts = {name: int(np.sum(column_class == code)) for code, name in COL_CLASS_NAMES.items()}
    total = nu * nv
    return InterfaceSet(
        h=np.full((0, nu, nv), np.nan), plane_type=np.zeros(0, int),
        plane_ideal=np.zeros(0), plane_mean=np.zeros(0), valid_fraction=np.zeros(0),
        column_class=column_class, expected_crossings=0,
        crossing_count=np.zeros((nu, nv), np.int32), bad_column_fraction=1.0,
        class_fractions={k: v / total for k, v in class_counts.items()},
        n_interfaces=0, n_valid_interfaces=0,
        W=float(W), w_periodic=bool(w_periodic),
        diagnostics={"n_planes": 0}, warnings=warnings,
    )
