"""Morphology-independent mixing metrics on fields and particles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

import numpy as np

from .composition import CompositionField, sample_field_at_points
from .density_field import DensityFields
from .trajectory import FrameData


@dataclass
class GeneralMixingResult:
    f_mix_voxel: Dict[str, float]
    f_mix_particle: Dict[str, float]
    wrong_side_A: Dict[str, float]
    wrong_side_B: Dict[str, float]
    overlap_O: float
    diagnostics: Dict[str, object]


def compute_general_mixing(fields: DensityFields, comp: CompositionField,
                           frame: FrameData, cfg: Mapping[str, object]) -> GeneralMixingResult:
    thresholds = [float(t) for t in cfg["mixing_thresholds"]]
    valid = ~comp.invalid_mask
    absphi = np.abs(comp.phi[valid])
    f_mix_voxel = {f"{t:g}": float(np.mean(absphi < t)) for t in thresholds}

    phi_beads = sample_field_at_points(comp.phi, comp.grid, frame.coords, order=1)
    isA = frame.types == int(cfg["A_type_id"])
    isB = frame.types == int(cfg["B_type_id"])
    f_mix_particle = {f"{t:g}": float(np.mean(np.abs(phi_beads) < t)) for t in thresholds}
    wrong_A = {f"{t:g}": float(np.mean(phi_beads[isA] < -t)) for t in thresholds}
    wrong_B = {f"{t:g}": float(np.mean(phi_beads[isB] > t)) for t in thresholds}

    rA, rB = fields.rho_A, fields.rho_B
    denom = np.sqrt(float(np.sum(rA * rA)) * float(np.sum(rB * rB)))
    overlap = float(np.sum(rA * rB) / denom) if denom > 0 else float("nan")
    return GeneralMixingResult(
        f_mix_voxel=f_mix_voxel, f_mix_particle=f_mix_particle,
        wrong_side_A=wrong_A, wrong_side_B=wrong_B, overlap_O=overlap,
        diagnostics={
            "wrong_side_definition": "A: phi<-threshold; B: phi>+threshold",
            "interpretation": (
                "wrong-side fractions measure spatial mixing; interpret them with "
                "the declared morphology and other interface metrics"
            ),
        },
    )
