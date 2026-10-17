"""Local lamellar spacing from neighbouring equivalent interfaces.

Equivalent interfaces are the same TYPE (both A-side rising, or both B-side
falling) and adjacent in that sub-lattice. Their local forward separation

    L_j(u, v) = h_{j+2}(u, v) - h_j(u, v)

is the local lamellar period. We compute it per (u, v) column where both
interfaces are valid, aggregate to per-pair and whole-sample statistics, and
compare with the S(q) estimate L0 = 2*pi/q_star.

Using same-type pairs (spacing = one full period) rather than adjacent
opposite-type interfaces (which give the A or B layer thickness, unequal for
asymmetric composition) is essential: it measures the period, not the block
thickness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .interface_detection import COL_GOOD, InterfaceSet


@dataclass
class SpacingResult:
    local_maps: List[np.ndarray]          # per equivalent-pair L_j(u, v) maps (NaN where invalid)
    pair_indices: List[tuple]             # (j, j+2) index pairs contributing
    all_values: np.ndarray                # pooled valid local spacings
    mean: float
    median: float
    std: float
    iqr: float
    cv: float
    minimum: float
    maximum: float
    p10: float
    p90: float
    L0_sq: Optional[float]                # S(q)-derived period, if provided
    rel_diff_vs_sq: Optional[float]       # |mean - L0_sq| / L0_sq
    clean_coverage: float                 # usable clean pairs / measurable pairs
    per_pair: List[Dict[str, Any]]
    warnings: List[str] = field(default_factory=list)


def compute_spacing(
    iset: InterfaceSet,
    L0_sq: Optional[float] = None,
    sq_flag_tol: float = 0.15,
) -> SpacingResult:
    """Local lamellar spacing from consecutive same-type interface pairs."""
    h = iset.h
    ptype = iset.plane_type
    M = iset.n_interfaces
    warnings: List[str] = []

    local_maps: List[np.ndarray] = []
    pair_indices: List[tuple] = []
    per_pair: List[Dict[str, Any]] = []
    pooled: List[np.ndarray] = []

    clean_columns = iset.column_class == COL_GOOD
    n_candidate = 0
    n_clean = 0

    # Pair every interface with the next interface of the same type.  On a
    # genuinely periodic normal window this includes the two wrap pairs that
    # the old j,j+2 loop omitted.  A bounded window uses forward pairs only.
    pairs: List[tuple] = []
    for j in range(M):
        candidates = [k for k in range(j + 1, M) if ptype[k] == ptype[j]]
        wraps = False
        if candidates:
            k = candidates[0]
        elif iset.w_periodic:
            wrapped = [k for k in range(0, j) if ptype[k] == ptype[j]]
            if not wrapped:
                continue
            k = wrapped[0]
            wraps = True
        else:
            continue
        pairs.append((j, k, wraps))

    for j, k, wraps in pairs:
        a, b = h[j], h[k]
        measurable = np.isfinite(a) & np.isfinite(b)
        both = measurable & clean_columns
        n_candidate += int(measurable.sum())
        n_clean += int(both.sum())
        if both.sum() < 4:
            continue
        Lmap = np.full(a.shape, np.nan)
        delta = b[both] - a[both]
        if wraps:
            delta = delta + iset.W
        Lmap[both] = delta
        vals = Lmap[both]
        # guard against a wrapped/misassigned pair giving a negative spacing
        if np.median(vals) <= 0:
            warnings.append(f"pair ({j},{k}) has non-positive median spacing; skipped.")
            continue
        local_maps.append(Lmap)
        pair_indices.append((j, k))
        pooled.append(vals)
        per_pair.append({
            "pair": (int(j), int(k)),
            "type": "A" if ptype[j] > 0 else "B",
            "wrap_pair": bool(wraps),
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "std": float(np.std(vals)),
            "n": int(vals.size),
        })

    clean_coverage = n_clean / n_candidate if n_candidate else 0.0
    if clean_coverage < 0.5 and n_candidate:
        warnings.append(
            f"only {100 * clean_coverage:.0f}% of measurable spacing pairs are in "
            "good columns; spacing describes clean patches, not the whole interface."
        )
    if not pooled:
        warnings.append("no equivalent interface pairs with sufficient overlap; spacing undefined.")
        return _empty_spacing(L0_sq, warnings, clean_coverage)

    allv = np.concatenate(pooled)
    q1, q3 = np.percentile(allv, [25, 75])
    mean = float(np.mean(allv))
    result = SpacingResult(
        local_maps=local_maps, pair_indices=pair_indices, all_values=allv,
        mean=mean, median=float(np.median(allv)), std=float(np.std(allv)),
        iqr=float(q3 - q1), cv=float(np.std(allv) / mean) if mean else float("nan"),
        minimum=float(allv.min()), maximum=float(allv.max()),
        p10=float(np.percentile(allv, 10)), p90=float(np.percentile(allv, 90)),
        L0_sq=L0_sq, rel_diff_vs_sq=None, clean_coverage=clean_coverage,
        per_pair=per_pair, warnings=warnings,
    )
    if L0_sq is not None and np.isfinite(L0_sq) and L0_sq > 0:
        result.rel_diff_vs_sq = abs(mean - L0_sq) / L0_sq
        if result.rel_diff_vs_sq > sq_flag_tol:
            warnings.append(
                f"interface-derived spacing {mean:.3f} differs from S(q) L0={L0_sq:.3f} "
                f"by {100 * result.rel_diff_vs_sq:.1f}% (> {100 * sq_flag_tol:.0f}%). "
                "The FFT n0/L0 and the S(q) magnitude disagree; check orientation crosscheck."
            )
    return result


def _empty_spacing(L0_sq, warnings, clean_coverage=0.0) -> SpacingResult:
    nan = float("nan")
    return SpacingResult(
        local_maps=[], pair_indices=[], all_values=np.zeros(0),
        mean=nan, median=nan, std=nan, iqr=nan, cv=nan, minimum=nan, maximum=nan,
        p10=nan, p90=nan, L0_sq=L0_sq, rel_diff_vs_sq=None,
        clean_coverage=float(clean_coverage), per_pair=[], warnings=warnings,
    )
