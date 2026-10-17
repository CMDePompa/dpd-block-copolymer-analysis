"""Interpenetration and compositional-mixing metrics.

Four complementary metrics, each with a distinct physical meaning. None
alone defines a defect. Interpretation guidance lives in the README; this
module measures spatial mixing.

1. Mixed-volume fraction f_mix(phi_mix): fraction of valid voxels with
   |phi| < phi_mix. Grows with interface area, interface width, AND bulk
   mixing — a blunt but robust global indicator. Evaluated at several
   thresholds.

2. Particle-based mixing: phi interpolated at bead positions;
   * mixed-bead fraction (|phi(r_i)| < phi_mix),
   * WRONG-SIDE fractions: A beads in strongly B-rich regions
     (phi < -phi_mix) and B beads in strongly A-rich regions (phi > +phi_mix).
   These fractions quantify spatial mixing; they do not alone identify
   topological defects.

3. Directional integrated wrong-side density. Sign convention: s is the
   signed distance from the local zero crossing ALONG the global normal,
   ORIENTED so s > 0 is the A-rich side of that interface. Then

     P_B->A = sum(rho_B, 0 < s <= s_half) / sum(rho_B, |s| <= s_half)
     P_A->B = sum(rho_A, -s_half <= s < 0) / sum(rho_A, |s| <= s_half)

   with s_half = penetration_halfwindow (default L0/2, so windows of
   neighbouring interfaces tile one period; each voxel contributes to the
   two interfaces bracketing it). Perfect SHARP segregation -> 0; full
   mixing -> 0.5. NOTE: because the window spans a half period on each side,
   its far edge reaches the NEXT interface, whose diffuse tail contributes a
   nonzero baseline (~0.1 for clean tanh interfaces at these L0/width
   ratios). P values are therefore comparable across systems analysed with
   the same parameters, but are not pure single-interface tail integrals.

4. Density-overlap O = <rho_A rho_B> / sqrt(<rho_A^2><rho_B^2>): a global
   scalar in [0, 1]; 0 for disjoint domains, 1 for identical fields.

Also produced: interface-frame mean profiles rho_A(s), rho_B(s), phi(s), and
lateral maps of the unlike-density fraction evaluated at each lamella's
midplane (where B does not belong in an A layer, and vice versa).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .composition import CompositionField, sample_field_at_points
from .config import is_auto
from .density_field import DensityFields
from .interface_detection import COL_GOOD, InterfaceSet
from .io_utils import LogFn
from .resample import SlabField, sample_on_slab
from .trajectory import FrameData


@dataclass
class PenetrationResult:
    f_mix_voxel: Dict[str, float]          # threshold -> voxel mixed fraction
    f_mix_particle: Dict[str, float]       # threshold -> bead mixed fraction
    wrong_side_A: Dict[str, float]         # threshold -> fraction of A beads with phi < -t
    wrong_side_B: Dict[str, float]         # threshold -> fraction of B beads with phi > +t
    P_B_to_A: float
    P_A_to_B: float
    overlap_O: float
    s_bins: np.ndarray                     # signed-distance bin centers
    rhoA_of_s: np.ndarray                  # mean rho_A(s) in the interface frame
    rhoB_of_s: np.ndarray
    phi_of_s: np.ndarray
    map_fB_in_A_core: np.ndarray           # (nu, nv) unlike fraction at A-lamella midplanes
    map_fA_in_B_core: np.ndarray           # (nu, nv) unlike fraction at B-lamella midplanes
    diagnostics: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)


def compute_penetration(
    fields: DensityFields,
    comp: CompositionField,
    slab: SlabField,
    iset: InterfaceSet,
    frame: FrameData,
    cfg: Dict[str, Any],
    log: Optional[LogFn] = None,
) -> PenetrationResult:
    """All four mixing metrics + interface-frame density profiles for one frame.

    Requires the frame's density fields, composition field, oblique slab, and
    detected interfaces (for the signed-distance frame), plus the raw frame
    (for particle-based metrics). See the module docstring for definitions
    and sign conventions.
    """
    thresholds = [float(t) for t in cfg["mixing_thresholds"]]
    A_id, B_id = int(cfg["A_type_id"]), int(cfg["B_type_id"])
    warnings: List[str] = []

    # ---- 1. voxel mixed fraction ----
    valid = ~comp.invalid_mask
    absphi = np.abs(comp.phi[valid])
    n_valid = max(int(valid.sum()), 1)
    f_mix_voxel = {f"{t:g}": float(np.sum(absphi < t) / n_valid) for t in thresholds}

    # ---- 2. particle-based mixing ----
    phi_at_beads = sample_field_at_points(comp.phi, comp.grid, frame.coords, order=1)
    isA = frame.types == A_id
    isB = frame.types == B_id
    nA, nB = max(int(isA.sum()), 1), max(int(isB.sum()), 1)
    f_mix_particle = {f"{t:g}": float(np.mean(np.abs(phi_at_beads) < t)) for t in thresholds}
    wrong_side_A = {f"{t:g}": float(np.sum(phi_at_beads[isA] < -t) / nA) for t in thresholds}
    wrong_side_B = {f"{t:g}": float(np.sum(phi_at_beads[isB] > t) / nB) for t in thresholds}

    # ---- 4. density overlap (global) ----
    rA, rB = fields.rho_A, fields.rho_B
    denom = np.sqrt(float(np.sum(rA ** 2)) * float(np.sum(rB ** 2)))
    overlap_O = float(np.sum(rA * rB) / denom) if denom > 0 else float("nan")

    # ---- 3. directional penetration in the interface frame ----
    s_half = (slab.L0 / 2.0 if (is_auto(cfg.get("penetration_halfwindow"))
                                or cfg.get("penetration_halfwindow") is None)
              else float(cfg["penetration_halfwindow"]))
    rhoA_slab = sample_on_slab(fields.rho_A, comp.grid, slab, order=1)
    rhoB_slab = sample_on_slab(fields.rho_B, comp.grid, slab, order=1)

    nu, nv, nw = slab.shape
    w = slab.w
    dw = slab.dw
    W = slab.W
    nbins = max(8, int(round(2 * s_half / dw)))
    s_edges = np.linspace(-s_half, s_half, nbins + 1)
    s_bins = 0.5 * (s_edges[:-1] + s_edges[1:])
    accA = np.zeros(nbins)
    accB = np.zeros(nbins)
    accPhi = np.zeros(nbins)
    accN = np.zeros(nbins)

    h = iset.h
    ptype = iset.plane_type
    clean_columns = iset.column_class == COL_GOOD
    n_measurable = 0
    n_clean_eligible = 0
    for j in range(iset.n_interfaces):
        sgn = float(ptype[j])              # rising: A side is s>0 already
        hj = h[j]
        measurable = np.isfinite(hj)
        vmask = measurable & clean_columns
        n_measurable += int(measurable.sum())
        if not slab.w_periodic:
            vmask &= ((hj - s_half >= w[0]) & (hj + s_half <= w[-1]))
        n_clean_eligible += int(vmask.sum())
        if not vmask.any():
            continue
        # signed distance of every w sample from this interface, oriented A-side positive
        # s(u,v,w) = wrap(w - h_j(u,v)) * sgn
        delta = w[None, None, :] - hj[:, :, None]
        if slab.w_periodic:
            delta = (delta + W / 2.0) % W - W / 2.0
        s_signed = delta * sgn
        sel = vmask[:, :, None] & (np.abs(s_signed) <= s_half)
        which = np.clip(((s_signed[sel] + s_half) / (2 * s_half) * nbins).astype(int), 0, nbins - 1)
        accA += np.bincount(which, weights=rhoA_slab[sel], minlength=nbins)
        accB += np.bincount(which, weights=rhoB_slab[sel], minlength=nbins)
        accPhi += np.bincount(which, weights=slab.phi_uvw[sel], minlength=nbins)
        accN += np.bincount(which, minlength=nbins)

    nz = accN > 0
    rhoA_of_s = np.divide(accA, accN, out=np.full(nbins, np.nan), where=nz)
    rhoB_of_s = np.divide(accB, accN, out=np.full(nbins, np.nan), where=nz)
    phi_of_s = np.divide(accPhi, accN, out=np.full(nbins, np.nan), where=nz)

    pos = s_bins > 0
    neg = s_bins < 0
    totB = float(np.nansum(accB))
    totA = float(np.nansum(accA))
    P_B_to_A = float(np.nansum(accB[pos]) / totB) if totB > 0 else float("nan")
    P_A_to_B = float(np.nansum(accA[neg]) / totA) if totA > 0 else float("nan")
    if not nz.all():
        warnings.append("some signed-distance bins are empty; penetration profile is patchy.")
    clean_coverage = n_clean_eligible / n_measurable if n_measurable else 0.0
    if n_measurable and clean_coverage < 0.5:
        warnings.append(
            f"only {100 * clean_coverage:.0f}% of measurable interface columns are "
            "clean and edge-safe; directional penetration describes clean patches only."
        )

    # ---- midplane unlike-fraction maps ----
    map_fB_in_A, map_fA_in_B = _midplane_maps(slab, iset, rhoA_slab, rhoB_slab)

    diagnostics = {
        "s_half": s_half,
        "n_s_bins": nbins,
        "clean_eligible_coverage": clean_coverage,
        "w_periodic": slab.w_periodic,
        "sign_convention": "s > 0 is the A-rich side of each interface (oriented per plane type)",
        "window_note": ("s_half = L0/2 tiles one period; each voxel contributes to the two "
                        "interfaces bracketing it"),
        "interpretation_note": ("wrong-side fractions measure spatial mixing and do not alone "
                                "identify topological defects"),
    }
    if log:
        log(f"penetration: P_B->A={P_B_to_A:.4f}, P_A->B={P_A_to_B:.4f}, O={overlap_O:.4f}, "
            f"f_mix(0.5)={f_mix_voxel.get('0.5', float('nan')):.4f}, "
            f"wrongA(0.5)={wrong_side_A.get('0.5', float('nan')):.4f}")
        for msg in warnings:
            log(f"WARN: {msg}")

    return PenetrationResult(
        f_mix_voxel=f_mix_voxel, f_mix_particle=f_mix_particle,
        wrong_side_A=wrong_side_A, wrong_side_B=wrong_side_B,
        P_B_to_A=P_B_to_A, P_A_to_B=P_A_to_B, overlap_O=overlap_O,
        s_bins=s_bins, rhoA_of_s=rhoA_of_s, rhoB_of_s=rhoB_of_s, phi_of_s=phi_of_s,
        map_fB_in_A_core=map_fB_in_A, map_fA_in_B_core=map_fA_in_B,
        diagnostics=diagnostics, warnings=warnings,
    )


def _midplane_maps(slab: SlabField, iset: InterfaceSet,
                   rhoA_slab: np.ndarray, rhoB_slab: np.ndarray):
    """Unlike-density fraction at each lamella's midplane, averaged over layers.

    The layer between adjacent planes (j, j+1) is A-rich when plane j is
    rising (+1). The midplane position per column is the mean of the two
    local crossing positions (circular-wrap aware); the unlike fraction is
    interpolated there along w.
    """
    nu, nv, nw = slab.shape
    w0 = slab.w[0]
    dw = slab.dw
    W = slab.W
    h = iset.h
    ptype = iset.plane_type
    M = iset.n_interfaces

    sumB_A = np.zeros((nu, nv)); cntB_A = np.zeros((nu, nv))
    sumA_B = np.zeros((nu, nv)); cntA_B = np.zeros((nu, nv))

    clean_columns = iset.column_class == COL_GOOD
    n_pairs = M if iset.w_periodic else max(0, M - 1)
    for j in range(n_pairs):
        jn = (j + 1) % M
        a, b = h[j], h[jn]
        both = np.isfinite(a) & np.isfinite(b) & clean_columns
        if both.sum() < 4:
            continue
        gap = ((b - a) % W if iset.w_periodic else b - a)
        mid = (a + 0.5 * gap)                 # midplane along w (may exceed window; wrap below)
        mid = np.where(both, mid, 0.0)        # dummy value on invalid cells (never used)
        idx = np.mod((mid - w0) / dw, nw)
        i0 = np.floor(idx).astype(int) % nw
        frac = idx - np.floor(idx)
        i1 = (i0 + 1) % nw
        ii, jj = np.where(both)
        rA = (1 - frac[ii, jj]) * rhoA_slab[ii, jj, i0[ii, jj]] + frac[ii, jj] * rhoA_slab[ii, jj, i1[ii, jj]]
        rB = (1 - frac[ii, jj]) * rhoB_slab[ii, jj, i0[ii, jj]] + frac[ii, jj] * rhoB_slab[ii, jj, i1[ii, jj]]
        tot = rA + rB
        good = tot > 1e-12
        if ptype[j] > 0:      # rising -> the layer after j is A-rich
            sumB_A[ii[good], jj[good]] += rB[good] / tot[good]
            cntB_A[ii[good], jj[good]] += 1
        else:                 # falling -> B-rich layer
            sumA_B[ii[good], jj[good]] += rA[good] / tot[good]
            cntA_B[ii[good], jj[good]] += 1

    map_fB_in_A = np.divide(sumB_A, cntB_A, out=np.full((nu, nv), np.nan), where=cntB_A > 0)
    map_fA_in_B = np.divide(sumA_B, cntA_B, out=np.full((nu, nv), np.nan), where=cntA_B > 0)
    return map_fB_in_A, map_fA_in_B
