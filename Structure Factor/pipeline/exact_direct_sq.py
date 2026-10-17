"""Exact full-lattice structure factors from particle coordinates (sq_method: "exact").

This engine evaluates the collective density modes F(k) at EVERY reciprocal
lattice vector k = (n_x*dq_x, n_y*dq_y, n_z*dq_z) with |k| inside the compute
range — no spatial gridding, no assignment window, no shell subsampling — so
it is exact for any structure orientation.

This is the standard direct particle-sum structure factor, i.e. Eq. 10 of
Gavrilov, Kudryavtsev & Chertovich, J. Chem. Phys. 139, 224901 (2013):

    S(q) = (1/n) < | sum_j exp(i q . r_j) |^2 >

with q on the box reciprocal lattice and n the total particle count. Our
S_AA is that quantity on the A-bead subset; tests/test_v4.py asserts
agreement with a literal transcription of Eq. 10.

Note the full lattice matters: restricting k, m, p to positive integers (as
Eq. 10 is literally written, and as freud's octant-folded sampling does)
drops mixed-sign vectors such as (1,-1,2), which carry real Bragg intensity
for obliquely oriented lamellae. A freud-based engine was evaluated on this
basis and rejected; see README4.md.

All spectra use N = N_A + N_B, matching the cic engine:

    S_AA(q)   = <|F_A(q)|^2>/N
    S_BB(q)   = <|F_B(q)|^2>/N
    S_AB(q)   = <Re[F_A(q) F_B(q)*]>/N
    S_tot(q)  = <|F_A(q) + F_B(q)|^2>/N
    S_comp(q) = <|F_A(q) - F_B(q)|^2>/N

Angle brackets denote the average over every reciprocal-lattice vector in the
radial q bin.

Implementation: the plane-wave factorizes per axis,
    exp(i k.r) = exp(i n_x dq_x x) * exp(i n_y dq_y y) * exp(i n_z dq_z z),
so F on the full (2n+1)^3 grid is a chunked complex matrix product
(BLAS zgemm) instead of a naive per-vector sum. For N = 96,000 particles and
q_max = 4 on an L = 32 box this is a few seconds per frame.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from .particle_mesh_sq import concentration_spectra


class ExactDirectSq:
    """Reusable exact full-lattice S(q) calculator.

    Mirrors the ParticleMeshSq interface:
    ``bin_edges`` / ``bin_centers`` / ``counts`` and
    ``compute(frame, A_type_id, B_type_id)`` -> dict with q, S_AA, S_BB,
    S_AB, S_tot, S_comp.
    """

    def __init__(
        self,
        L,
        q_compute_min: float,
        q_compute_max: float,
        q_bins: int,
        chunk_particles: int = 4096,
    ):
        self.L = np.asarray(L, dtype=np.float64)
        self.chunk_particles = int(chunk_particles)
        self.bin_edges = np.linspace(
            float(q_compute_min), float(q_compute_max), int(q_bins) + 1
        )
        self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2.0

        self._dq = 2.0 * np.pi / self.L
        n_max = np.floor(float(q_compute_max) / self._dq).astype(int)
        # Full integer ranges per axis: -n_max .. +n_max (complete lattice,
        # no octant folding, no half-space bookkeeping).
        self._n_ranges = [
            np.arange(-n_max[d], n_max[d] + 1, dtype=np.int64) for d in range(3)
        ]
        kx = self._n_ranges[0] * self._dq[0]
        ky = self._n_ranges[1] * self._dq[1]
        kz = self._n_ranges[2] * self._dq[2]
        qmag = np.sqrt(
            kx[:, None, None] ** 2
            + ky[None, :, None] ** 2
            + kz[None, None, :] ** 2
        )

        eps = 1e-9 * max(float(q_compute_max), 1.0)
        valid = (
            (qmag >= self.bin_edges[0] - eps)
            & (qmag <= self.bin_edges[-1] + eps)
            & (qmag > 0)
        )
        self._valid = valid.ravel()
        q_selected = qmag.ravel()[self._valid]
        bin_index = np.searchsorted(self.bin_edges, q_selected, side="right") - 1
        self._bin_index = np.clip(
            bin_index, 0, len(self.bin_centers) - 1
        ).astype(np.int32)
        self.counts = np.bincount(
            self._bin_index, minlength=len(self.bin_centers)
        ).astype(np.int64)
        self._grid_shape = qmag.shape

    def _collective_modes(self, coords: np.ndarray) -> np.ndarray:
        """F(k) = sum_j exp(i k.r_j) on the selected lattice vectors."""
        Mx, My, Mz = (len(r) for r in self._n_ranges)
        F = np.zeros((Mx, My * Mz), dtype=np.complex128)
        for start in range(0, len(coords), self.chunk_particles):
            stop = min(start + self.chunk_particles, len(coords))
            chunk = coords[start:stop]
            ex = np.exp(
                1j * self._dq[0] * chunk[:, 0:1] * self._n_ranges[0][None, :]
            )  # (n, Mx)
            ey = np.exp(
                1j * self._dq[1] * chunk[:, 1:2] * self._n_ranges[1][None, :]
            )  # (n, My)
            ez = np.exp(
                1j * self._dq[2] * chunk[:, 2:3] * self._n_ranges[2][None, :]
            )  # (n, Mz)
            # (n, My*Mz) outer product per particle, then contract over
            # particles with one zgemm: (Mx, n) @ (n, My*Mz).
            w = (ey[:, :, None] * ez[:, None, :]).reshape(len(chunk), My * Mz)
            F += ex.T @ w
        return F.reshape(Mx * My * Mz)[self._valid]

    def _shell_mean(self, values: np.ndarray, N: int) -> np.ndarray:
        sums = np.bincount(
            self._bin_index, weights=values, minlength=len(self.bin_centers)
        )
        out = np.full(len(self.bin_centers), np.nan, dtype=np.float64)
        occ = self.counts > 0
        out[occ] = sums[occ] / (self.counts[occ] * N)
        return out

    def compute(
        self,
        frame,
        A_type_id: int = 1,
        B_type_id: int = 2,
    ) -> Dict[str, np.ndarray]:
        if not np.allclose(frame.L, self.L, rtol=1e-8, atol=1e-10):
            raise ValueError("Box changed after reciprocal grid initialization")
        A = np.asarray(frame.coords[frame.types == A_type_id], dtype=np.float64)
        B = np.asarray(frame.coords[frame.types == B_type_id], dtype=np.float64)
        N = len(A) + len(B)
        if not len(A) or not len(B):
            raise ValueError("Both A and B particles are required")

        F_A = self._collective_modes(A)
        F_B = self._collective_modes(B)

        S_AA = self._shell_mean(np.abs(F_A) ** 2, N)
        S_BB = self._shell_mean(np.abs(F_B) ** 2, N)
        S_AB = self._shell_mean(np.real(F_A * np.conj(F_B)), N)
        S_tot = self._shell_mean(np.abs(F_A + F_B) ** 2, N)
        S_comp = self._shell_mean(np.abs(F_A - F_B) ** 2, N)
        S_cc, S_cc_normalized, x_A, x_B = concentration_spectra(
            S_AA, S_BB, S_AB, len(A), len(B)
        )
        return {
            "q": self.bin_centers.copy(),
            "S_AA": S_AA,
            "S_BB": S_BB,
            "S_AB": S_AB,
            "S_tot": S_tot,
            "S_comp": S_comp,
            "S_cc": S_cc,
            "S_cc_normalized": S_cc_normalized,
            "x_A": x_A,
            "x_B": x_B,
        }


def make_exact_sq_object(
    L,
    q_compute_min: float,
    q_compute_max: float,
    q_bins: int,
) -> ExactDirectSq:
    return ExactDirectSq(L, q_compute_min, q_compute_max, q_bins)
