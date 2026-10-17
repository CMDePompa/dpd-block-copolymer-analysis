"""Periodic-aware topology of thresholded composition sets.

For a threshold phi_c, the A-rich set is M_A = {phi > phi_c} and the B-rich
set is M_B = {phi < phi_c}; both are analyzed over a sweep of thresholds
(never a single segmentation).

Everything here is explicitly periodic:

* :func:`periodic_label` — scipy labeling followed by union-find merging
  across the periodic faces. The union-find carries integer displacement
  potentials, so it also detects WRAPPING (percolation): a merge that closes
  a cycle with a nonzero box displacement marks the component as wrapping in
  those directions. Exact for face (6) connectivity; for 26-connectivity the
  cross-boundary diagonal adjacencies are included as well.

* :func:`euler_characteristic_periodic` — exact Euler characteristic of the
  voxel cubical complex on the 3-torus, chi = V - E + F - C (vertex/edge/
  face/cube counts with periodic identification, 6-connected foreground
  convention). Verified against known solids in the tests.

Interpretation for lamellae (per component chi is the workhorse):
  * intact lamellar slab (wraps laterally): chi = 0;
  * tunnels/bridges generally drive chi downward, but coexisting islands and
    cavities can offset that change; max(0,-chi) is therefore a connection
    score, not a literal defect count;
  * isolated blob (island): chi = +1; an enclosed cavity adds +1.

Betti numbers: beta0 is exact (periodic components). beta2 is estimated as
the number of NON-WRAPPING complement components (enclosed cavities) and
beta1 = beta0 + beta2 - chi is then derived. This decomposition is EXACT
ONLY when no component wraps the box; intact lamellae themselves wrap and
carry nontrivial H1/H2, so for lamellar systems the per-component chi and
the wrap flags are the trustworthy quantities. This limitation is recorded
in the output rather than hidden.

The expected component count for PERFECT lamellae with lattice mode m is
gcd(|m1|, |m2|, |m3|) (nonzero entries), reported for reference.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage

from .composition import CompositionField
from .io_utils import LogFn


# ---------------------------------------------------------------------------
# Weighted union-find with displacement potentials (wrap detection)
# ---------------------------------------------------------------------------
class _OffsetUnionFind:
    """Union-find over label ids carrying integer 3-vector potentials.

    pot(x) is the displacement of x's reference copy relative to its root's.
    union(a, b, delta) imposes pot(b) - pot(a) = delta; a contradiction
    (cycle with nonzero net displacement) marks the root as wrapping in the
    axes where the cycle vector is nonzero.
    """

    def __init__(self, n_labels: int) -> None:
        self.parent = np.arange(n_labels + 1)
        self.pot = np.zeros((n_labels + 1, 3), dtype=np.int64)
        self.wraps = np.zeros((n_labels + 1, 3), dtype=bool)

    def find(self, x: int) -> Tuple[int, np.ndarray]:
        path = []
        while self.parent[x] != x:
            path.append(x)
            x = int(self.parent[x])
        # path compression with potential accumulation (reverse order)
        total = np.zeros(3, dtype=np.int64)
        for node in reversed(path):
            total = total + self.pot[node]
            self.parent[node] = x
            self.pot[node] = total
        return x, (self.pot[path[0]].copy() if path else np.zeros(3, dtype=np.int64))

    def union(self, a: int, b: int, delta: np.ndarray) -> None:
        ra, _ = self.find(a)
        rb, _ = self.find(b)
        pa = self.pot[a] if a != ra else np.zeros(3, dtype=np.int64)
        pb = self.pot[b] if b != rb else np.zeros(3, dtype=np.int64)
        if ra != rb:
            # attach rb under ra with pot(rb) = pa + delta - pb
            self.parent[rb] = ra
            self.pot[rb] = pa + delta - pb
            self.wraps[ra] |= self.wraps[rb]
        else:
            cycle = pb - pa - delta
            self.wraps[ra] |= (cycle != 0)


@dataclass
class PeriodicComponents:
    labels: np.ndarray             # (n,) periodic component id per voxel-label region, 0 = background
    n_components: int
    volumes: np.ndarray            # (n_components,) voxel counts, sorted descending
    wraps: np.ndarray              # (n_components, 3) bool, aligned with volumes
    chi_per_component: List[int]
    label_map: Dict[int, int]      # scipy label -> component index (1-based order of volumes)


def _neighbor_shifts(connectivity: int) -> List[Tuple[int, int, int]]:
    if connectivity == 6:
        return [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    shifts = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if (dx, dy, dz) == (0, 0, 0):
                    continue
                if (dx, dy, dz) < (0, 0, 0):   # unique half-space
                    continue
                shifts.append((dx, dy, dz))
    return shifts


def periodic_label(mask: np.ndarray, connectivity: int = 6,
                   compute_chi: bool = True) -> PeriodicComponents:
    """Label a boolean field on the 3-torus with wrap detection."""
    structure = ndimage.generate_binary_structure(3, 1 if connectivity == 6 else 3)
    labels, n_raw = ndimage.label(mask, structure=structure)
    if n_raw == 0:
        return PeriodicComponents(labels=labels, n_components=0,
                                  volumes=np.zeros(0, int), wraps=np.zeros((0, 3), bool),
                                  chi_per_component=[], label_map={})

    uf = _OffsetUnionFind(n_raw)
    n = mask.shape
    idx = [np.arange(nn) for nn in n]
    for shift in _neighbor_shifts(connectivity):
        s = np.array(shift)
        rolled = np.roll(labels, tuple(-s), axis=(0, 1, 2))
        pair = (labels > 0) & (rolled > 0)
        if not pair.any():
            continue
        # box-crossing vector per cell along each axis (0 or +-1)
        cross = np.zeros(mask.shape + (3,), dtype=np.int8)
        crossing_any = np.zeros(mask.shape, dtype=bool)
        for d in range(3):
            if s[d] == 0:
                continue
            cr = ((idx[d] + s[d]) // n[d]).astype(np.int8)  # +1 at the top edge (s>0)
            sl = [None, None, None]
            shape = [1, 1, 1]
            shape[d] = n[d]
            cr3 = cr.reshape(shape)
            cross[..., d] = cr3
            crossing_any |= (cr3 != 0)
        sel = pair & crossing_any
        if not sel.any():
            continue
        a_lab = labels[sel]
        b_lab = rolled[sel]
        deltas = cross[sel]
        # unique constraint triples to keep the python loop tiny
        arr = np.column_stack([a_lab, b_lab, deltas])
        for a, b, d0, d1, d2 in np.unique(arr, axis=0):
            uf.union(int(a), int(b), np.array([d0, d1, d2], dtype=np.int64))

    # resolve roots
    root_of = np.zeros(n_raw + 1, dtype=np.int64)
    for lab in range(1, n_raw + 1):
        root_of[lab], _ = uf.find(lab)
    roots, inv = np.unique(root_of[1:], return_inverse=True)
    comp_of_label = inv + 1                       # 1-based component per scipy label

    comp_labels = np.zeros_like(labels)
    nzm = labels > 0
    comp_labels[nzm] = comp_of_label[labels[nzm] - 1]

    n_comp = len(roots)
    volumes = np.bincount(comp_labels[nzm], minlength=n_comp + 1)[1:]
    wraps = np.zeros((n_comp, 3), dtype=bool)
    for k, r in enumerate(roots):
        wraps[k] = uf.wraps[r]

    # sort by volume descending
    order = np.argsort(volumes)[::-1]
    remap = np.zeros(n_comp + 1, dtype=np.int64)
    remap[order + 1] = np.arange(1, n_comp + 1)
    comp_labels = remap[comp_labels]
    volumes = volumes[order]
    wraps = wraps[order]
    label_map = {int(lab): int(remap[comp_of_label[lab - 1]]) for lab in range(1, n_raw + 1)}

    chi_list: List[int] = []
    if compute_chi:
        for k in range(1, n_comp + 1):
            chi_list.append(euler_characteristic_periodic(comp_labels == k))

    return PeriodicComponents(labels=comp_labels, n_components=n_comp,
                              volumes=volumes, wraps=wraps,
                              chi_per_component=chi_list, label_map=label_map)


def euler_characteristic_periodic(mask: np.ndarray) -> int:
    """Exact chi = V - E + F - C of the voxel complex on the 3-torus."""
    mask = mask.astype(bool)
    C = int(mask.sum())
    if C == 0:
        return 0
    F = 0
    for d in range(3):
        F += int(np.logical_or(mask, np.roll(mask, -1, axis=d)).sum())
    E = 0
    for d in range(3):
        a1, a2 = [x for x in range(3) if x != d]
        m = mask | np.roll(mask, -1, axis=a1) | np.roll(mask, -1, axis=a2) \
            | np.roll(np.roll(mask, -1, axis=a1), -1, axis=a2)
        E += int(m.sum())
    m = mask.copy()
    for d in range(3):
        m = m | np.roll(m, -1, axis=d)
    V = int(m.sum())
    return V - E + F - C


# ---------------------------------------------------------------------------
# Threshold sweep
# ---------------------------------------------------------------------------
def expected_lamellae_gcd(lattice_mode: Optional[Sequence[int]]) -> Optional[int]:
    """gcd(|m_i|) of the nonzero lattice-mode entries = perfect-lamellae count."""
    if lattice_mode is None:
        return None
    vals = [abs(int(v)) for v in lattice_mode if int(v) != 0]
    if not vals:
        return None
    g = vals[0]
    for v in vals[1:]:
        g = math.gcd(g, v)
    return g


@dataclass
class TopologyResult:
    rows: List[Dict[str, Any]]                # one per (threshold, phase)
    connectivity_view: Optional[np.ndarray]   # component labels at phi_c = 0 (A set) for plotting
    diagnostics: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)


def compute_topology(
    comp: CompositionField,
    thresholds: Sequence[float],
    connectivity: int = 6,
    lattice_mode: Optional[Sequence[int]] = None,
    threshold_mode: str = "absolute_phi",
    log: Optional[LogFn] = None,
) -> TopologyResult:
    """Sweep thresholds; label both phases periodically; tabulate topology."""
    phi = comp.phi
    n_total = phi.size
    valid = ~comp.invalid_mask
    n_valid = int(valid.sum())
    invalid_fraction = float(comp.invalid_mask.mean())
    topology_reliable = invalid_fraction == 0.0
    expected = expected_lamellae_gcd(lattice_mode)
    rows: List[Dict[str, Any]] = []
    warnings: List[str] = []
    view = None

    if threshold_mode not in ("absolute_phi", "normalized_plateaus"):
        raise ValueError(f"unknown topology threshold mode: {threshold_mode!r}")
    plateau_low = float("nan")
    plateau_high = float("nan")
    midpoint = 0.0
    halfspan = 1.0
    threshold_field = phi
    if threshold_mode == "normalized_plateaus" and n_valid:
        plateau_low, plateau_high = np.percentile(phi[valid], [5.0, 95.0])
        midpoint = 0.5 * (plateau_high + plateau_low)
        halfspan = 0.5 * (plateau_high - plateau_low)
        if halfspan <= 1e-8:
            warnings.append("composition plateau span is too small for normalized topology thresholds.")
            halfspan = 1.0
        threshold_field = (phi - midpoint) / halfspan
    if invalid_fraction > 0.0:
        warnings.append(
            f"{100 * invalid_fraction:.2f}% invalid-density voxels are excluded from "
            "topology; connectivity results are marked unreliable because even a "
            "small excluded region can change topology."
        )

    for phi_c in thresholds:
        actual_phi_threshold = (midpoint + float(phi_c) * halfspan
                                if threshold_mode == "normalized_plateaus"
                                else float(phi_c))
        for phase, mask in (("A", valid & (threshold_field > phi_c)),
                            ("B", valid & (threshold_field < phi_c))):
            vol_frac = float(mask.mean())
            if mask.sum() == 0 or int(mask.sum()) == n_valid:
                rows.append({
                    "phi_c": float(phi_c), "phase": phase, "volume_fraction": vol_frac,
                    "phi_threshold_actual": actual_phi_threshold,
                    "threshold_mode": threshold_mode,
                    "invalid_fraction": invalid_fraction,
                    "topology_reliable": topology_reliable,
                    "n_components": 0 if mask.sum() == 0 else 1,
                    "n_wrapping": 0, "n_islands": 0,
                    "largest_volume_fraction": vol_frac,
                    "island_volume_median_voxels": float("nan"),
                    "chi_total": 0 if mask.sum() == 0 else euler_characteristic_periodic(mask),
                    "chi_largest": float("nan"),
                    "n_excess_connections": float("nan"),
                    "connection_score": float("nan"),
                    "beta0": 0 if mask.sum() == 0 else 1,
                    "beta2_cavities": 0, "beta1_est": float("nan"),
                    "expected_components_perfect": expected,
                    "note": "empty or full set at this threshold",
                })
                continue
            pc = periodic_label(mask, connectivity=connectivity)
            # complement analysis for cavities
            pc_bg = periodic_label(valid & ~mask, connectivity=connectivity,
                                   compute_chi=False)
            n_cavities = int(np.sum(~pc_bg.wraps.any(axis=1))) if pc_bg.n_components else 0
            chi_total = int(np.sum(pc.chi_per_component)) if pc.chi_per_component else 0
            wraps_any = pc.wraps.any(axis=1)
            n_wrapping = int(wraps_any.sum())
            n_islands = int((~wraps_any).sum())
            island_vols = pc.volumes[~wraps_any]
            excess = int(np.sum([max(0, -c) for c in pc.chi_per_component]))
            beta1_est = pc.n_components + n_cavities - chi_total
            rows.append({
                "phi_c": float(phi_c), "phase": phase, "volume_fraction": vol_frac,
                "phi_threshold_actual": actual_phi_threshold,
                "threshold_mode": threshold_mode,
                "invalid_fraction": invalid_fraction,
                "topology_reliable": topology_reliable,
                "n_components": pc.n_components,
                "n_wrapping": n_wrapping,
                "n_islands": n_islands,
                "largest_volume_fraction": float(pc.volumes[0] / n_total),
                "island_volume_median_voxels": float(np.median(island_vols)) if island_vols.size else float("nan"),
                "chi_total": chi_total,
                "chi_largest": int(pc.chi_per_component[0]) if pc.chi_per_component else float("nan"),
                "n_excess_connections": excess,
                "connection_score": excess,
                "beta0": pc.n_components,
                "beta2_cavities": n_cavities,
                "beta1_est": beta1_est,
                "expected_components_perfect": expected,
                "note": ("connection_score=max(0,-chi) is a topology heuristic, not "
                         "a literal bridge count; beta1_est is exact only if nothing "
                         "wraps, so use chi/wrap flags for lamellae"),
            })
            if phase == "A" and abs(phi_c) < 1e-12:
                view = pc.labels

    diagnostics = {
        "connectivity": connectivity,
        "threshold_mode": threshold_mode,
        "plateau_percentiles_5_95": [plateau_low, plateau_high],
        "plateau_midpoint": midpoint,
        "plateau_halfspan": halfspan,
        "invalid_fraction": invalid_fraction,
        "expected_components_perfect_lamellae": expected,
        "chi_interpretation": ("chi=0 intact wrapping slab; negative chi indicates "
                               "handles/connections but is not a literal defect count"),
    }
    if log:
        mid = [r for r in rows if abs(r["phi_c"]) < 1e-12 and r["phase"] == "A"]
        if mid:
            r = mid[0]
            log(f"topology(A, phi_c=0): beta0={r['beta0']} (perfect={expected}), "
                f"wrapping={r['n_wrapping']}, islands={r['n_islands']}, "
                f"chi={r['chi_total']}, excess_connections={r['n_excess_connections']}")
    return TopologyResult(rows=rows, connectivity_view=view,
                          diagnostics=diagnostics, warnings=warnings)
