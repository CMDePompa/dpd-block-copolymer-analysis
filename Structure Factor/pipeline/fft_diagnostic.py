"""Voxel-FFT composition-field diagnostic for cross-validation and window detection.

This module computes a radially averaged S(q) from the Fourier transform of the
composition field phi = (rhoA - rhoB) / (rhoA + rhoB).

WARNING — normalization note
----------------------------
The FFT S(q) here is NOT equivalent to the publishable S_comp(q) in absolute values.
  - The FFT S(q) is proportional to |FT[phi]|² where phi is binned onto a voxel
    grid. Its scale depends on grid size, particle density per voxel, and the
    empty-voxel treatment.
  - The engine S_comp(q) = (1/N)|F_A(q) − F_B(q)|², a properly normalized
    particle-based structure factor.
  - Both methods agree on the POSITION of the composition peak (q_star) but NOT
    on absolute peak heights.

Use the FFT diagnostic for:
  1. Cheap per-frame q_star estimates for the stationary-window heuristic.
  2. Cross-validation of the engine q_star (should agree within fft_crosscheck_tol).

Do NOT use FFT S(q) as a replacement for the engine S_comp in reported results.
"""

from __future__ import annotations

from typing import Dict

import numpy as np


def fft_diagnostic_frame(
    frame,
    A_type_id: int = 1,
    B_type_id: int = 2,
    grid_n: int = 64,
    phi_mode: str = "zero",
    phi_smooth_sigma: float = 1.0,
    n_radial_bins: int = 128,
    q_peak_search_min: float = 0.20,
    q_peak_search_max: float = 5.00,
) -> Dict[str, float]:
    """Compute cheap FFT-based diagnostics for one frame.

    Returns a dict with:
      q_star_fft      — peak position of radially averaged FFT S(q)
      peak_height_fft — S(q) at q_star (proportional units, not normalized)
      phi_rms         — RMS of composition field (order-parameter proxy)
      grad_rms        — RMS of grad(phi) (interface density proxy)
    """
    lo, hi = frame.box_lo, frame.box_hi
    phi = _make_phi_field(
        frame.coords, frame.types, lo, hi, grid_n,
        A_type_id, B_type_id, phi_mode=phi_mode, phi_smooth_sigma=phi_smooth_sigma,
    )

    phi_rms = float(np.sqrt(np.mean(phi ** 2)))
    L = hi - lo
    nx, ny, nz = phi.shape
    dx, dy, dz = L[0] / nx, L[1] / ny, L[2] / nz
    gx, gy, gz = np.gradient(phi, dx, dy, dz, edge_order=1)
    grad_rms = float(np.sqrt(np.mean(gx ** 2 + gy ** 2 + gz ** 2)))

    n_particles = int(
        np.count_nonzero(
            (frame.types == A_type_id) | (frame.types == B_type_id)
        )
    )
    S, QX, QY, QZ = _compute_spectrum(
        phi, lo, hi, n_particles=n_particles
    )
    q_centers, Sq_rad = _radial_average(S, QX, QY, QZ, n_radial_bins)

    q_star, peak_height = _peak_in_range(q_centers, Sq_rad, q_peak_search_min, q_peak_search_max)

    return {
        "q_star_fft": q_star,
        "peak_height_fft": peak_height,
        "phi_rms": phi_rms,
        "grad_rms": grad_rms,
    }


def _make_phi_field(
    coords: np.ndarray,
    types: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    grid_n: int,
    A_type: int,
    B_type: int,
    phi_mode: str = "zero",
    phi_smooth_sigma: float = 1.0,
) -> np.ndarray:
    """Build a smoothed, density-normalized contrast field on a CIC grid.

    phi_mode options:
      "zero"     — empty voxels set to 0 (default, cheapest)
      "nan_mean" — empty voxels filled with mean of occupied phi
      "smooth"   — Gaussian smoothing of rhoA, rhoB before phi (requires scipy)
    """
    from .particle_mesh_sq import _cic_density

    L = hi - lo
    isA = types == A_type
    isB = types == B_type
    rhoA = _cic_density(coords[isA], lo, L, grid_n)
    rhoB = _cic_density(coords[isB], lo, L, grid_n)

    mode = str(phi_mode).strip().lower()
    if mode not in {"zero", "nan_mean", "smooth"}:
        raise ValueError(
            f"Unknown phi_mode={phi_mode!r}; use zero, nan_mean, or smooth"
        )
    if mode == "smooth":
        # Smoothing is periodic and applied before the local normalization,
        # avoiding sparse-voxel +/-1 artifacts.
        try:
            from scipy.ndimage import gaussian_filter
        except ImportError as exc:
            raise ImportError("phi_mode=smooth requires scipy") from exc
        rhoA = gaussian_filter(rhoA, sigma=phi_smooth_sigma, mode="wrap")
        rhoB = gaussian_filter(rhoB, sigma=phi_smooth_sigma, mode="wrap")

    total = rhoA + rhoB
    occupied = total > 1e-14
    phi = np.zeros_like(total)
    phi[occupied] = (rhoA[occupied] - rhoB[occupied]) / total[occupied]
    if mode == "nan_mean" and np.any(occupied):
        phi[~occupied] = float(np.mean(phi[occupied]))
    return phi


def _compute_spectrum(
    phi: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    n_particles: int,
):
    nx, ny, nz = phi.shape
    L = hi - lo
    phi0 = phi - np.mean(phi)
    F = np.fft.fftn(phi0)
    mean_grid_density = n_particles / float(nx * ny * nz)
    S = np.abs(F) ** 2 * mean_grid_density**2 / max(n_particles, 1)
    qx = 2.0 * np.pi * np.fft.fftfreq(nx, d=L[0] / nx)
    qy = 2.0 * np.pi * np.fft.fftfreq(ny, d=L[1] / ny)
    qz = 2.0 * np.pi * np.fft.fftfreq(nz, d=L[2] / nz)
    S = np.fft.fftshift(S)
    qx = np.fft.fftshift(qx)
    qy = np.fft.fftshift(qy)
    qz = np.fft.fftshift(qz)
    QX, QY, QZ = np.meshgrid(qx, qy, qz, indexing="ij")
    return S, QX, QY, QZ


def _radial_average(
    S: np.ndarray,
    QX: np.ndarray,
    QY: np.ndarray,
    QZ: np.ndarray,
    nbins: int,
):
    qmag = np.sqrt(QX ** 2 + QY ** 2 + QZ ** 2).ravel()
    Sflat = S.ravel()
    qmax = float(qmag.max())
    edges = np.linspace(0.0, qmax, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    which = np.digitize(qmag, edges) - 1
    valid = (which >= 0) & (which < nbins)
    sums = np.bincount(which[valid], weights=Sflat[valid], minlength=nbins).astype(float)
    counts = np.bincount(which[valid], minlength=nbins).astype(float)
    Sq = np.zeros(nbins, dtype=np.float64)
    nz = counts > 0
    Sq[nz] = sums[nz] / counts[nz]
    return centers, Sq


def _peak_in_range(q, Sq, q_min, q_max):
    mask = (q >= q_min) & (q <= q_max) & np.isfinite(Sq)
    if not np.any(mask):
        return float("nan"), float("nan")
    q_sub, S_sub = q[mask], Sq[mask]
    i = int(np.argmax(S_sub))
    return float(q_sub[i]), float(S_sub[i])
