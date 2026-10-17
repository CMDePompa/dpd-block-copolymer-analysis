"""Validated particle-mesh structure factors for A/B polymer systems.

The production spectrum is computed from Cloud-In-Cell (CIC) particle
assignment followed by an FFT on the periodic reciprocal lattice.  The CIC
assignment window is deconvolved before shell averaging.

All spectra use N = N_A + N_B:

    S_AA(q)   = <|F_A(q)|^2>/N
    S_BB(q)   = <|F_B(q)|^2>/N
    S_AB(q)   = <Re[F_A(q) F_B(q)*]>/N
    S_tot(q)  = <|F_A(q) + F_B(q)|^2>/N
    S_comp(q) = <|F_A(q) - F_B(q)|^2>/N

For comparisons across systems whose global A/B composition changes, the
engines also report the Bhatia--Thornton-style concentration spectrum

    S_cc(q) = x_B^2 S_AA + x_A^2 S_BB - 2 x_A x_B S_AB

and S_cc_normalized = S_cc/(x_A*x_B). The normalized curve removes the
trivial amplitude change caused solely by changing the global A/B fractions.

Angle brackets denote an average over every reciprocal-lattice vector in the
radial q bin.  ``exact_direct_composition_sq`` provides a deliberately slower
particle-coordinate calculation for low-q validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy.signal import find_peaks as _scipy_find_peaks
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False


def _is_auto(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() == "auto"


def concentration_spectra(
    S_AA: np.ndarray,
    S_BB: np.ndarray,
    S_AB: np.ndarray,
    n_A: int,
    n_B: int,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Return concentration spectra and the global A/B bead fractions."""
    N = int(n_A) + int(n_B)
    if N <= 0 or n_A <= 0 or n_B <= 0:
        raise ValueError("Both A and B particles are required")
    x_A = float(n_A) / N
    x_B = float(n_B) / N
    S_cc = x_B**2 * S_AA + x_A**2 * S_BB - 2.0 * x_A * x_B * S_AB
    S_cc_normalized = S_cc / (x_A * x_B)
    return S_cc, S_cc_normalized, x_A, x_B


def resolve_q_grid(L, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve q bins from the physical reciprocal lattice of the box."""
    Lx, Ly, Lz = (float(v) for v in L)
    dq_x, dq_y, dq_z = 2.0 * np.pi / np.array([Lx, Ly, Lz])
    dq_min = min(dq_x, dq_y, dq_z)
    q_compute_max = float(cfg["q_compute_max"])
    notes: Dict[str, str] = {"q_compute_max": "explicit from config"}

    raw = cfg.get("q_compute_min", "auto")
    q_compute_min = dq_min if _is_auto(raw) else float(raw)
    notes["q_compute_min"] = (
        f"auto: smallest reciprocal spacing = {dq_min:.8g}"
        if _is_auto(raw) else "explicit from config"
    )

    raw = cfg.get("q_peak_search_min", "auto")
    q_peak_search_min = dq_min if _is_auto(raw) else float(raw)
    notes["q_peak_search_min"] = (
        f"auto: smallest reciprocal spacing = {dq_min:.8g}"
        if _is_auto(raw) else "explicit from config"
    )

    raw = cfg.get("q_peak_search_max")
    q_peak_search_max = q_compute_max if raw is None else float(raw)
    notes["q_peak_search_max"] = (
        "using q_compute_max" if raw is None else "explicit from config"
    )

    raw_bins = cfg.get("q_bins", "auto")
    raw_width = cfg.get("q_bin_width", "auto")
    q_range = q_compute_max - q_compute_min
    if not _is_auto(raw_bins):
        q_bins = int(raw_bins)
    else:
        requested_width = (
            dq_min / 4.0 if _is_auto(raw_width) else float(raw_width)
        )
        q_bins = max(1, int(np.ceil(q_range / requested_width)))
    q_bin_width = q_range / q_bins
    notes["q_bins"] = (
        "explicit from config" if not _is_auto(raw_bins)
        else "derived from q-bin width"
    )
    notes["q_bin_width"] = f"effective uniform width = {q_bin_width:.8g}"

    shell_volume = 4.0 * np.pi * (
        q_compute_max**3 - q_compute_min**3
    ) / 3.0
    n_kvec_estimate = int(shell_volume / (dq_x * dq_y * dq_z))
    return {
        "Lx": Lx, "Ly": Ly, "Lz": Lz,
        "dq_x": float(dq_x), "dq_y": float(dq_y), "dq_z": float(dq_z),
        "q_compute_min": q_compute_min,
        "q_compute_max": q_compute_max,
        "q_bin_width": q_bin_width,
        "q_bins": q_bins,
        "q_peak_search_min": q_peak_search_min,
        "q_peak_search_max": q_peak_search_max,
        "n_kvec_estimate": n_kvec_estimate,
        "resolution_notes": notes,
    }


def _cic_density(
    coords: np.ndarray,
    box_lo: np.ndarray,
    L: np.ndarray,
    grid_n: int,
) -> np.ndarray:
    """Deposit point particles onto periodic grid nodes with CIC weights."""
    shape = (grid_n, grid_n, grid_n)
    scaled = np.mod((coords - box_lo) / L, 1.0) * grid_n
    base = np.floor(scaled).astype(np.int64)
    frac = scaled - base
    rho = np.zeros(grid_n**3, dtype=np.float64)

    for ox in (0, 1):
        wx = frac[:, 0] if ox else 1.0 - frac[:, 0]
        ix = (base[:, 0] + ox) % grid_n
        for oy in (0, 1):
            wy = frac[:, 1] if oy else 1.0 - frac[:, 1]
            iy = (base[:, 1] + oy) % grid_n
            for oz in (0, 1):
                wz = frac[:, 2] if oz else 1.0 - frac[:, 2]
                iz = (base[:, 2] + oz) % grid_n
                flat = (ix * grid_n + iy) * grid_n + iz
                rho += np.bincount(
                    flat, weights=wx * wy * wz, minlength=grid_n**3
                )
    return rho.reshape(shape)


class ParticleMeshSq:
    """Reusable reciprocal grid and CIC particle-mesh calculator."""

    def __init__(
        self,
        L,
        q_compute_min: float,
        q_compute_max: float,
        q_bins: int,
        grid_n: int = 128,
    ):
        self.L = np.asarray(L, dtype=np.float64)
        self.grid_n = int(grid_n)
        if self.grid_n < 16:
            raise ValueError("sq_grid must be at least 16")
        nyquist_min = np.min(np.pi * self.grid_n / self.L)
        if q_compute_max >= 0.70 * nyquist_min:
            raise ValueError(
                f"q_compute_max={q_compute_max:g} is too close to the grid "
                f"Nyquist limit {nyquist_min:g}; increase sq_grid"
            )

        self.bin_edges = np.linspace(
            q_compute_min, q_compute_max, int(q_bins) + 1
        )
        self.bin_centers = (
            self.bin_edges[:-1] + self.bin_edges[1:]
        ) / 2.0

        freq_xy = np.fft.fftfreq(self.grid_n)
        freq_z = np.fft.rfftfreq(self.grid_n)
        qx = 2.0 * np.pi * np.fft.fftfreq(
            self.grid_n, d=self.L[0] / self.grid_n
        )
        qy = 2.0 * np.pi * np.fft.fftfreq(
            self.grid_n, d=self.L[1] / self.grid_n
        )
        qz = 2.0 * np.pi * np.fft.rfftfreq(
            self.grid_n, d=self.L[2] / self.grid_n
        )
        qmag = np.sqrt(
            qx[:, None, None] ** 2
            + qy[None, :, None] ** 2
            + qz[None, None, :] ** 2
        )

        # rfftn omits the negative-z conjugate half.  Give interior z planes
        # weight two so shell averages equal a full reciprocal-lattice average.
        degeneracy_z = np.full(len(qz), 2, dtype=np.int64)
        degeneracy_z[0] = 1
        if self.grid_n % 2 == 0:
            degeneracy_z[-1] = 1
        degeneracy = np.broadcast_to(
            degeneracy_z[None, None, :], qmag.shape
        )

        # CIC amplitude window W(k) = product sinc(n_i/N)^2.
        wx = np.sinc(freq_xy) ** 2
        wy = np.sinc(freq_xy) ** 2
        wz = np.sinc(freq_z) ** 2
        cic_window = (
            wx[:, None, None] * wy[None, :, None] * wz[None, None, :]
        )

        eps = 1e-10
        valid = (
            (qmag >= self.bin_edges[0] - eps)
            & (qmag <= self.bin_edges[-1] + eps)
            & (qmag > 0)
        )
        q_selected = qmag[valid]
        bin_index = np.searchsorted(
            self.bin_edges, q_selected, side="right"
        ) - 1
        bin_index = np.clip(bin_index, 0, len(self.bin_centers) - 1)
        self._selected = np.flatnonzero(valid.ravel())
        self._bin_index = bin_index.astype(np.int32)
        self._mode_weight = degeneracy[valid].astype(np.float64)
        self._inv_cic_window = 1.0 / cic_window[valid]
        self.counts = np.bincount(
            self._bin_index,
            weights=self._mode_weight,
            minlength=len(self.bin_centers),
        ).astype(np.int64)

    def _shell_mean(self, values: np.ndarray, N: int) -> np.ndarray:
        sums = np.bincount(
            self._bin_index,
            weights=self._mode_weight * values,
            minlength=len(self.bin_centers),
        )
        out = np.full(len(self.bin_centers), np.nan, dtype=np.float64)
        valid = self.counts > 0
        out[valid] = sums[valid] / (self.counts[valid] * N)
        return out

    def compute(
        self,
        frame,
        A_type_id: int = 1,
        B_type_id: int = 2,
    ) -> Dict[str, np.ndarray]:
        if not np.allclose(frame.L, self.L, rtol=1e-8, atol=1e-10):
            raise ValueError("Box changed after reciprocal grid initialization")
        A = frame.coords[frame.types == A_type_id]
        B = frame.coords[frame.types == B_type_id]
        N = len(A) + len(B)
        if not len(A) or not len(B):
            raise ValueError("Both A and B particles are required")

        rho_A = _cic_density(A, frame.box_lo, self.L, self.grid_n)
        rho_B = _cic_density(B, frame.box_lo, self.L, self.grid_n)
        F_A = np.fft.rfftn(rho_A).ravel()[self._selected]
        F_B = np.fft.rfftn(rho_B).ravel()[self._selected]
        F_A = F_A * self._inv_cic_window
        F_B = F_B * self._inv_cic_window

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


def make_sq_object(
    L,
    q_compute_min: float,
    q_compute_max: float,
    q_bins: int,
    grid_n: int = 128,
) -> ParticleMeshSq:
    return ParticleMeshSq(
        L, q_compute_min, q_compute_max, q_bins, grid_n=grid_n
    )


def compute_particle_mesh_sq_single_frame(
    frame,
    A_type_id: int = 1,
    B_type_id: int = 2,
    sq: Optional[ParticleMeshSq] = None,
    q_compute_min: float = 0.1,
    q_compute_max: float = 4.0,
    q_bins: int = 150,
    grid_n: int = 128,
) -> Dict[str, np.ndarray]:
    if sq is None:
        sq = make_sq_object(
            frame.L, q_compute_min, q_compute_max, q_bins, grid_n
        )
    return sq.compute(frame, A_type_id=A_type_id, B_type_id=B_type_id)


def exact_direct_composition_sq(
    frame,
    bin_edges: np.ndarray,
    A_type_id: int = 1,
    B_type_id: int = 2,
    q_max: float = 1.0,
    chunk_size: int = 24,
) -> Dict[str, np.ndarray]:
    """Compute low-q composition Sq directly from particle coordinates."""
    L = np.asarray(frame.L, dtype=np.float64)
    dq = 2.0 * np.pi / L
    nmax = np.floor(float(q_max) / dq).astype(int)
    triplets = np.stack(
        np.meshgrid(
            np.arange(-nmax[0], nmax[0] + 1),
            np.arange(-nmax[1], nmax[1] + 1),
            np.arange(-nmax[2], nmax[2] + 1),
            indexing="ij",
        ),
        axis=-1,
    ).reshape(-1, 3)
    kvec = triplets * dq
    kmag = np.linalg.norm(kvec, axis=1)
    keep = (
        (kmag > 0)
        & (kmag >= bin_edges[0] - 1e-10)
        & (kmag <= min(float(q_max), bin_edges[-1]) + 1e-10)
    )
    kvec, kmag = kvec[keep], kmag[keep]
    signs = np.where(
        frame.types == A_type_id,
        1.0,
        np.where(frame.types == B_type_id, -1.0, 0.0),
    )
    use = signs != 0
    coords = np.asarray(frame.coords[use], dtype=np.float64)
    signs = signs[use]
    N = len(signs)
    values = np.empty(len(kvec), dtype=np.float64)
    for start in range(0, len(kvec), int(chunk_size)):
        stop = min(start + int(chunk_size), len(kvec))
        phase = coords @ kvec[start:stop].T
        amplitude = signs @ np.exp(1j * phase)
        values[start:stop] = np.abs(amplitude) ** 2 / N

    counts, _ = np.histogram(kmag, bins=bin_edges)
    sums, _ = np.histogram(kmag, bins=bin_edges, weights=values)
    spectrum = np.full(len(bin_edges) - 1, np.nan)
    occupied = counts > 0
    spectrum[occupied] = sums[occupied] / counts[occupied]
    return {
        "q": (bin_edges[:-1] + bin_edges[1:]) / 2.0,
        "S_comp": spectrum,
        "counts": counts.astype(int),
        "n_vectors": int(len(kvec)),
    }


def validate_mesh_against_exact(
    mesh_result: Dict[str, np.ndarray],
    exact_result: Dict[str, np.ndarray],
    q_min: float,
    q_max: float,
    peak_rel_tolerance: float = 0.10,
) -> Dict[str, Any]:
    """Compare CIC and exact spectra without inventing a peak in disorder.

    A significant exact peak is checked by bin and intensity. If the exact
    spectrum has no significant local maximum, validation uses the relative
    RMS error over the complete finite low-q spectrum instead.
    """
    q = np.asarray(exact_result["q"])
    exact = np.asarray(exact_result["S_comp"])
    mesh = np.asarray(mesh_result["S_comp"])
    mask = (
        (q >= q_min) & (q <= q_max)
        & np.isfinite(exact) & np.isfinite(mesh)
    )
    if not np.any(mask):
        return {
            "passed": False,
            "reason": "no common finite validation bins",
            "q_star_exact": np.nan,
            "q_star_mesh": np.nan,
            "peak_rel_error": np.nan,
        }
    indices = np.flatnonzero(mask)
    rms_denom = max(float(np.sqrt(np.mean(exact[indices] ** 2))), 1e-30)
    spectrum_rel_rmse = float(
        np.sqrt(np.mean((mesh[indices] - exact[indices]) ** 2)) / rms_denom
    )
    exact_peak = _find_dominant_low_q_peak(
        q, exact, q_min, q_max, len(q)
    )
    mesh_peak = _find_dominant_low_q_peak(
        q, mesh, q_min, q_max, len(q)
    )
    has_peak = bool(exact_peak.get("peak_detected", False))
    if has_peak:
        exact_peak_i = int(np.argmin(np.abs(q - exact_peak["q_star"])))
        mesh_peak_i = (
            int(np.argmin(np.abs(q - mesh_peak["q_star"])))
            if mesh_peak.get("peak_detected", False) else -1
        )
        denom = max(abs(float(exact[exact_peak_i])), 1e-30)
        peak_rel_error = abs(
            float(mesh[exact_peak_i]) - float(exact[exact_peak_i])
        ) / denom
        q_match = exact_peak_i == mesh_peak_i
        passed = q_match and peak_rel_error <= float(peak_rel_tolerance)
        reason = (
            "passed" if passed else
            "peak bin mismatch" if not q_match else
            "peak intensity mismatch"
        )
    else:
        exact_peak_i = -1
        mesh_peak_i = -1
        peak_rel_error = np.nan
        q_match = False
        passed = spectrum_rel_rmse <= float(peak_rel_tolerance)
        reason = "passed_no_peak_spectrum_rmse" if passed else "spectrum_rmse_mismatch"
    return {
        "passed": bool(passed),
        "reason": reason,
        "exact_peak_detected": has_peak,
        "q_star_exact": float(q[exact_peak_i]) if exact_peak_i >= 0 else np.nan,
        "q_star_mesh": float(q[mesh_peak_i]) if mesh_peak_i >= 0 else np.nan,
        "S_peak_exact": float(exact[exact_peak_i]) if exact_peak_i >= 0 else np.nan,
        "S_mesh_at_exact_peak": float(mesh[exact_peak_i]) if exact_peak_i >= 0 else np.nan,
        "peak_rel_error": float(peak_rel_error),
        "spectrum_rel_rmse": spectrum_rel_rmse,
        "peak_rel_tolerance": float(peak_rel_tolerance),
    }


def _check_s_tot_consistency(
    S_AA: np.ndarray,
    S_BB: np.ndarray,
    S_AB: np.ndarray,
    S_tot: np.ndarray,
) -> float:
    reconstructed = S_AA + S_BB + 2.0 * S_AB
    denom = np.maximum(np.abs(S_AA) + np.abs(S_BB), 1e-30)
    rel = np.abs(S_tot - reconstructed) / denom
    return float(np.nanmax(rel)) if np.isfinite(rel).any() else np.nan


def _find_dominant_low_q_peak(
    q: np.ndarray,
    S_comp: np.ndarray,
    q_min: float,
    q_max: float,
    q_bins: int,
    min_height_over_median: float = 1.5,
    min_relative_prominence: float = 0.10,
    min_occupied_bins: int = 5,
) -> Dict[str, Any]:
    """Find a significant finite-q local maximum, or explicitly report none.

    Older versions always returned the largest sampled bin. That behavior
    assigned a meaningless q_star to every disordered spectrum.
    """
    mask = (q >= q_min) & (q <= q_max) & np.isfinite(S_comp)
    if int(np.count_nonzero(mask)) < int(min_occupied_bins):
        return {
            "q_star": np.nan, "peak_height": np.nan,
            "at_lower_boundary": False, "at_upper_boundary": False,
            "n_candidates": 0, "q_candidates": [],
            "method": "no_peak",
            "peak_detected": False,
            "reason": "too_few_occupied_bins",
            "background_median": np.nan,
            "height_over_median": np.nan,
            "relative_prominence": np.nan,
        }
    q_sub, S_sub = q[mask], S_comp[mask]
    background = float(np.nanmedian(S_sub))
    positive_scale = max(abs(background), 1e-30)
    candidates: List[int] = []
    prominences = np.array([], dtype=float)
    if _SCIPY_AVAILABLE:
        peaks, props = _scipy_find_peaks(S_sub, prominence=0.0)
        candidates = peaks.tolist()
        prominences = np.asarray(props.get("prominences", []), dtype=float)
    else:
        candidates = [
            i for i in range(1, len(S_sub) - 1)
            if S_sub[i] > S_sub[i - 1] and S_sub[i] > S_sub[i + 1]
        ]
        prominences = np.asarray([
            S_sub[i] - max(S_sub[i - 1], S_sub[i + 1]) for i in candidates
        ])

    qualifying: List[Tuple[int, float, float, float]] = []
    for j, idx in enumerate(candidates):
        height = float(S_sub[idx])
        prominence = float(prominences[j]) if j < len(prominences) else 0.0
        height_ratio = height / positive_scale
        prominence_ratio = prominence / max(abs(height), 1e-30)
        if (
            height_ratio >= float(min_height_over_median)
            and prominence_ratio >= float(min_relative_prominence)
        ):
            qualifying.append((idx, prominence, height_ratio, prominence_ratio))

    if not qualifying:
        return {
            "q_star": np.nan,
            "peak_height": np.nan,
            "at_lower_boundary": False,
            "at_upper_boundary": False,
            "n_candidates": len(candidates),
            "q_candidates": [float(q_sub[i]) for i in candidates],
            "method": "no_peak",
            "peak_detected": False,
            "reason": "no_local_maximum_passed_significance_thresholds",
            "background_median": background,
            "height_over_median": np.nan,
            "relative_prominence": np.nan,
        }

    best, best_prominence, height_ratio, prominence_ratio = max(
        qualifying, key=lambda item: S_sub[item[0]]
    )
    q_star = float(q_sub[best])
    finite_q = np.sort(q_sub[np.isfinite(q_sub)])
    local_dq = float(np.nanmedian(np.diff(finite_q))) if len(finite_q) > 1 else 0.0
    return {
        "q_star": q_star,
        "peak_height": float(S_sub[best]),
        "at_lower_boundary": q_star < q_min + 2.0 * local_dq,
        "at_upper_boundary": q_star > q_max - 2.0 * local_dq,
        "n_candidates": len(qualifying),
        "q_candidates": [float(q_sub[item[0]]) for item in qualifying],
        "method": "significant_local_maximum",
        "peak_detected": True,
        "reason": "passed_significance_thresholds",
        "background_median": background,
        "height_over_median": float(height_ratio),
        "relative_prominence": float(prominence_ratio),
        "prominence": float(best_prominence),
    }


@dataclass
class MeshSqResult:
    q: np.ndarray
    S_AA: np.ndarray
    S_BB: np.ndarray
    S_AB: np.ndarray
    S_tot: np.ndarray
    S_comp: np.ndarray
    S_cc: np.ndarray
    S_cc_normalized: np.ndarray
    x_A: float
    x_B: float
    q_star: float
    L0: float
    peak_height: float
    n_frames_averaged: int
    peak_detected: bool = False
    peak_reason: str = ""
    peak_background_median: float = np.nan
    peak_height_over_median: float = np.nan
    peak_relative_prominence: float = np.nan
    q_star_at_lower_boundary: bool = False
    q_star_at_upper_boundary: bool = False
    n_peak_candidates: int = 0
    peak_detection_method: str = "unknown"
    s_tot_check_max_rel_err: float = 0.0
    warnings: List[str] = field(default_factory=list)
    normalization_notes: Dict[str, Any] = field(default_factory=dict)


def partial_sq_to_dataframe(result: MeshSqResult):
    import pandas as pd
    return pd.DataFrame({
        "q": result.q, "S_AA": result.S_AA, "S_BB": result.S_BB,
        "S_AB": result.S_AB, "S_tot": result.S_tot,
    })


def composition_sq_to_dataframe(result: MeshSqResult):
    import pandas as pd
    return pd.DataFrame({
        "q": result.q,
        "S_comp_contrast": result.S_comp,
        "S_cc": result.S_cc,
        "S_cc_normalized": result.S_cc_normalized,
    })
