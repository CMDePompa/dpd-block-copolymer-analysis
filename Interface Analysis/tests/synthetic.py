"""Synthetic lamellar systems with known ground truth for unit tests.

Two families:

* analytic phi fields on a grid (deterministic, noise-free) — for interface
  detection, waviness, spacing, width, defect, and topology tests;
* particle frames (FrameData) sampled from a lamellar composition law — for
  deposition/smoothing tests and end-to-end runs;

plus a LAMMPS dump writer for trajectory-reader tests.

Ground-truth conventions used by the generators:
  phi(r) = offset + amp * tanh( sin(q . r - q * h(r)) / (|q| * lam) )

  * q = 2*pi * (m / L) with integer m — always commensurate with the box;
  * lam: intrinsic interface half-width (tanh length in PHYSICAL units,
    exact at the crossing where |d sin/ds| = |q|);
  * h(r): imposed interface displacement field along the normal (waviness);
  * offset/amp: composition asymmetry (offset != 0 mimics A beads dissolved
    in the B-rich phase).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

from ..density_field import GridSpec
from ..trajectory import FrameData


def unit_normal(m: Sequence[int], L: Sequence[float]) -> np.ndarray:
    """Unit normal of the lamellar family with integer Miller-like indices m."""
    q = 2.0 * np.pi * np.asarray(m, dtype=float) / np.asarray(L, dtype=float)
    return q / np.linalg.norm(q)


def make_phi_grid(
    shape: Tuple[int, int, int] = (48, 48, 48),
    L: Tuple[float, float, float] = (24.0, 24.0, 24.0),
    m: Tuple[int, int, int] = (0, 0, 2),
    lam: float = 0.8,
    amp: float = 0.9,
    offset: float = 0.0,
    waviness_amp: float = 0.0,
    waviness_kint: Tuple[int, int, int] = (1, 0, 0),
    phase: float = 0.3,
    box_lo: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Tuple[np.ndarray, GridSpec, dict]:
    """Analytic lamellar phi field with known width, waviness, and spacing.

    Returns (phi, grid, truth) where truth records every ground-truth value:
    q_vec, |q|, L0, n0, lam, expected 10-90 width, imposed waviness RMS, etc.

    The waviness displaces interfaces along the normal by
    h(r) = waviness_amp * cos(k . r + 0.7), with k an integer lattice vector
    PERPENDICULAR to q (enforced), so the height field is single valued and
    periodic; its lateral RMS is waviness_amp / sqrt(2).
    """
    shape = tuple(int(n) for n in shape)
    L_arr = np.asarray(L, dtype=float)
    lo = np.asarray(box_lo, dtype=float)
    grid = GridSpec(box_lo=lo, box_hi=lo + L_arr, shape=shape, voxel=L_arr / np.asarray(shape))

    q_vec = 2.0 * np.pi * np.asarray(m, dtype=float) / L_arr
    q_mag = float(np.linalg.norm(q_vec))
    n0 = q_vec / q_mag

    k_vec = 2.0 * np.pi * np.asarray(waviness_kint, dtype=float) / L_arr
    if waviness_amp != 0.0 and abs(float(np.dot(k_vec, n0))) > 1e-12:
        raise ValueError("waviness_kint must be perpendicular to m for a single-valued height field")

    axes = [grid.axis_centers(i) for i in range(3)]
    X, Y, Z = np.meshgrid(*axes, indexing="ij")
    qr = q_vec[0] * X + q_vec[1] * Y + q_vec[2] * Z
    if waviness_amp != 0.0:
        h = waviness_amp * np.cos(k_vec[0] * X + k_vec[1] * Y + k_vec[2] * Z + 0.7)
        qr = qr - q_mag * h
    phi = offset + amp * np.tanh(np.sin(qr + phase) / (q_mag * lam))

    truth = {
        "m": tuple(int(v) for v in m),
        "q_vec": q_vec,
        "q_mag": q_mag,
        "L0": 2.0 * np.pi / q_mag,
        "n0": n0,
        "lam": lam,
        "width_10_90": 2.0 * np.arctanh(0.8) * lam,
        "amp": amp,
        "offset": offset,
        "waviness_amp": waviness_amp,
        "waviness_rms": waviness_amp / np.sqrt(2.0) if waviness_amp else 0.0,
        "waviness_k": k_vec,
        "phase": phase,
        "n_interfaces_per_period": 2,
    }
    return phi, grid, truth


def make_lamellar_frame(
    L: float = 24.0,
    rho: float = 3.0,
    m: Tuple[int, int, int] = (0, 0, 2),
    lam: float = 0.8,
    fA_B_block: float = 0.0,
    phase: float = 0.3,
    seed: int = 7,
    timestep: int = 1000,
) -> Tuple[FrameData, dict]:
    """Particle frame sampled from a lamellar composition law.

    Positions are uniform (dense-melt approximation, total density exactly
    uniform on average); each particle is A with probability
      P_A(r) = pmin + (pmax - pmin) * (1 + tanh(sin(q.r + phase)/(|q| lam)))/2,
    where pmin = fA_B_block mimics A beads dissolved in the B-rich phase
    (mixed A/B domain) and pmax = 1 (pure A lamella).

    The EXPECTED phi field is then offset + amp*tanh(...) with
    amp = (1 - pmin)/ (1 + ... ) — recorded in truth as expected_amp/offset
    computed from pmin: phi = 2*P_A - 1.
    """
    rng = np.random.default_rng(seed)
    n = int(round(rho * L ** 3))
    coords = rng.uniform(0.0, L, size=(n, 3))
    q_vec = 2.0 * np.pi * np.asarray(m, dtype=float) / L
    q_mag = float(np.linalg.norm(q_vec))
    s = coords @ q_vec + phase
    core = 0.5 * (1.0 + np.tanh(np.sin(s) / (q_mag * lam)))
    p_A = fA_B_block + (1.0 - fA_B_block) * core
    is_A = rng.uniform(size=n) < p_A
    types = np.where(is_A, 1, 2).astype(np.int32)
    frame = FrameData(
        timestep=timestep,
        box_lo=np.zeros(3),
        box_hi=np.full(3, float(L)),
        types=types,
        coords=coords,
        mol_ids=None,
    )
    # phi = 2 P_A - 1 = (2 fA_B_block - 1) + 2 (1 - fA_B_block) * core
    #     = offset + amp * tanh(...) with:
    truth = {
        "q_vec": q_vec,
        "q_mag": q_mag,
        "L0": 2.0 * np.pi / q_mag,
        "n0": q_vec / q_mag,
        "lam": lam,
        "expected_amp": 1.0 - fA_B_block,
        "expected_offset": fA_B_block,
        "fraction_A_expected": float(np.mean(p_A)),
        "phase": phase,
    }
    return frame, truth


def write_lammpstrj(
    path: str | Path,
    frames: Iterable[FrameData],
    coord_style: str = "xu",
    include_mol: bool = True,
    unwrap_shift: float = 0.0,
    truncate_last_atoms: int = 0,
) -> Path:
    """Write frames as a LAMMPS dump for reader tests.

    coord_style: 'x' (wrapped), 'xu' (unwrapped; unwrap_shift adds whole-box
    offsets to exercise folding), or 'xs' (scaled).
    truncate_last_atoms: drop this many atom lines from the LAST frame to
    fabricate a truncated/incomplete file.
    """
    path = Path(path)
    frames = list(frames)
    with path.open("w", newline="\n") as fh:
        for fi, fr in enumerate(frames):
            n = fr.n_atoms
            L = fr.L
            cols = {"x": "x y z", "xu": "xu yu zu", "xs": "xs ys zs"}[coord_style]
            mol_col = "mol " if include_mol else ""
            bead_col = "bead_index " if fr.bead_indices is not None else ""
            fh.write("ITEM: TIMESTEP\n%d\n" % fr.timestep)
            fh.write("ITEM: NUMBER OF ATOMS\n%d\n" % n)
            fh.write("ITEM: BOX BOUNDS pp pp pp\n")
            for d in range(3):
                fh.write("%.16e %.16e\n" % (fr.box_lo[d], fr.box_hi[d]))
            fh.write(f"ITEM: ATOMS id type {mol_col}{bead_col}{cols}\n")
            n_write = n - (truncate_last_atoms if fi == len(frames) - 1 else 0)
            for i in range(n_write):
                xyz = fr.coords[i].copy()
                if coord_style == "xu":
                    # push some atoms out of the box by whole box lengths
                    xyz = xyz + unwrap_shift * L * ((i % 3) - 1)
                elif coord_style == "xs":
                    xyz = (xyz - fr.box_lo) / L
                atom_id = int(fr.atom_ids[i]) if fr.atom_ids is not None else i + 1
                mol = f"{int(fr.mol_ids[i]) if fr.mol_ids is not None else 1} " if include_mol else ""
                bead = f"{int(fr.bead_indices[i])} " if fr.bead_indices is not None else ""
                fh.write(f"{atom_id} {int(fr.types[i])} {mol}{bead}"
                         f"{xyz[0]:.10g} {xyz[1]:.10g} {xyz[2]:.10g}\n")
    return path
