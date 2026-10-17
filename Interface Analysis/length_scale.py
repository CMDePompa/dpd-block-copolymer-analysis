"""Characteristic morphology length from S(q) or a conservative FFT fallback."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .composition import CompositionField


@dataclass
class LengthScaleResult:
    q_star: float
    length: float
    source: str
    diagnostics: Dict[str, object]
    warnings: List[str] = field(default_factory=list)


def resolve_characteristic_length(comp: CompositionField,
                                  sq_q_star: Optional[float]) -> LengthScaleResult:
    if sq_q_star is not None and np.isfinite(sq_q_star) and sq_q_star > 0:
        q = float(sq_q_star)
        return LengthScaleResult(
            q_star=q, length=2.0 * np.pi / q, source="S(q) pipeline",
            diagnostics={"rule": "L*=2*pi/q_star from configured S(q) results"},
        )

    phi = comp.phi - float(np.mean(comp.phi))
    power = np.abs(np.fft.fftn(phi)) ** 2
    power[0, 0, 0] = 0.0
    index = np.unravel_index(int(np.argmax(power)), power.shape)
    signed = []
    for i, n in zip(index, power.shape):
        signed.append(i if i <= n // 2 else i - n)
    mode = np.asarray(signed, dtype=float)
    qvec = 2.0 * np.pi * mode / comp.grid.L
    q = float(np.linalg.norm(qvec))
    if q <= 0:
        raise RuntimeError("could not determine a nonzero morphology length scale")
    return LengthScaleResult(
        q_star=q, length=2.0 * np.pi / q, source="3-D composition FFT fallback",
        diagnostics={
            "rule": "magnitude of strongest non-DC composition Fourier bin",
            "dominant_lattice_mode": [int(v) for v in mode],
            "note": (
                "for nonlamellar morphologies the direction of this one peak is not used; "
                "only its magnitude sets a local profile window"
            ),
        },
        warnings=[
            "No usable S(q) q_star was found; the characteristic length uses a quantized "
            "single-bin FFT fallback. Configure sq_results_dir for production analysis."
        ],
    )
