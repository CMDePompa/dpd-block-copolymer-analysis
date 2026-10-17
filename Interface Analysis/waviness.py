"""Waviness of an interface height field h(u, v): amplitude, slope, spectra.

All quantities operate on a single interface's height map (from
:mod:`height_fields`). The distinction from interface WIDTH is strict: width
is the intrinsic diffuseness of phi across the interface (see :mod:`width`);
waviness is the geometric undulation of the phi = 0 surface. They are never
combined into one number.

Metrics:
  * RMS waviness w_j and peak-to-peak Delta h_j;
  * local slope |grad h| (mean, RMS, max, and the distribution);
  * height-height correlation C_h(Delta u, Delta v) (radially averaged);
  * height power spectrum S_h(q_par): 2-D, isotropically averaged, and the
    two principal-axis directional cuts.

Lateral-plane periodicity: the oblique (u, v) window is not periodic in
general (see resample.py), so spectra and correlations remove the mean plane
(linear detrend) and apply a Hann window before the FFT to suppress the
window-edge discontinuity. This is recorded so the spectra are not
over-interpreted at the lowest q_par.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .height_fields import HeightMap, gradient_valid_mask


@dataclass
class WavinessResult:
    rms: float
    peak_to_peak: float
    slope_mean: float
    slope_rms: float
    slope_max: float
    slope_percentiles: Dict[str, float]
    q_par: np.ndarray              # isotropic spectrum wavenumbers
    S_iso: np.ndarray              # isotropically averaged height power spectrum
    Cr_r: np.ndarray               # radial lag for the correlation function
    Cr: np.ndarray                 # radially averaged height-height correlation
    corr_length: float             # lag where C_h falls to 1/e of C_h(0)
    dominant_lambda: float         # wavelength of the spectral peak (excl. q=0)
    anisotropy_ratio: float        # principal-axis spectral power ratio (>=1)
    diagnostics: Dict[str, Any]


def _detrend_plane(dh: np.ndarray) -> np.ndarray:
    """Remove the best-fit tilt plane a*u + b*v + c from a height fluctuation."""
    nu, nv = dh.shape
    u = np.arange(nu) - (nu - 1) / 2.0
    v = np.arange(nv) - (nv - 1) / 2.0
    U, V = np.meshgrid(u, v, indexing="ij")
    A = np.column_stack([U.ravel(), V.ravel(), np.ones(dh.size)])
    coef, *_ = np.linalg.lstsq(A, dh.ravel(), rcond=None)
    return dh - (A @ coef).reshape(nu, nv)


def _hann2d(shape: Tuple[int, int]) -> np.ndarray:
    wx = np.hanning(shape[0])
    wy = np.hanning(shape[1])
    return np.outer(wx, wy)


def compute_waviness(
    hmap: HeightMap,
    detrend: bool = True,
    window: bool = True,
) -> WavinessResult:
    """Full waviness characterization of one interface height map.

    RMS and peak-to-peak use only the MEASURED cells (no infill bias). Slopes
    and spectra use the gap-filled field (finite everywhere); the filled
    fraction is reported so heavily-infilled maps can be down-weighted.
    """
    du, dv = hmap.du, hmap.dv
    measured = hmap.h_raw[hmap.valid]
    if measured.size < 4:
        return _empty_waviness(du, dv, hmap.filled_fraction)

    mean_h = hmap.mean_height
    rms = float(np.sqrt(np.mean((measured - mean_h) ** 2)))
    peak_to_peak = float(measured.max() - measured.min())

    dh = hmap.dh  # filled, mean-subtracted
    gu, gv = np.gradient(dh, du, dv, edge_order=1)
    slope_mag = np.sqrt(gu ** 2 + gv ** 2)
    # Restrict slopes to cells whose complete finite-difference stencil was
    # measured.  A filled neighbour must not manufacture a local slope.
    smask = gradient_valid_mask(hmap.valid)
    slope_vals = slope_mag[smask]
    if slope_vals.size > 4:
        slope_mean = float(np.mean(slope_vals))
        slope_rms = float(np.sqrt(np.mean(slope_vals ** 2)))
        slope_max = float(np.max(slope_vals))
        slope_percentiles = {
            "p50": float(np.percentile(slope_vals, 50)),
            "p90": float(np.percentile(slope_vals, 90)),
            "p99": float(np.percentile(slope_vals, 99)),
        }
    else:
        slope_mean = slope_rms = slope_max = float("nan")
        slope_percentiles = {"p50": float("nan"), "p90": float("nan"),
                             "p99": float("nan")}

    # ---- Spectrum ----
    field = _detrend_plane(dh) if detrend else dh
    win = _hann2d(field.shape) if window else np.ones(field.shape)
    win_norm = np.sqrt(np.mean(win ** 2))
    F = np.fft.fft2(field * win) / (field.size * win_norm)
    P = np.abs(np.fft.fftshift(F)) ** 2
    nu, nv = field.shape
    qu = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(nu, d=du))
    qv = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(nv, d=dv))
    QU, QV = np.meshgrid(qu, qv, indexing="ij")
    Qmag = np.sqrt(QU ** 2 + QV ** 2)

    q_par, S_iso = _radial_average(P, Qmag)
    # Dominant wavelength from the exact 2-D spectral peak (grid-resolution
    # accurate), not the coarse radial bins. Exclude the DC bin.
    Pdc = P.copy()
    Pdc[nu // 2, nv // 2] = 0.0
    ipk2d = np.unravel_index(int(np.argmax(Pdc)), Pdc.shape)
    q_peak = float(Qmag[ipk2d])
    dominant_lambda = 2.0 * np.pi / q_peak if q_peak > 0 else float("nan")

    anisotropy_ratio = _anisotropy(P, QU, QV)

    # ---- Height-height correlation via Wiener-Khinchin (on the windowed field) ----
    Cr_r, Cr, corr_length = _height_correlation(field, du, dv)

    diagnostics = {
        "filled_fraction": hmap.filled_fraction,
        "n_measured": int(measured.size),
        "detrended": detrend,
        "windowed": window,
        "uv_periodic": False,
        "spectrum_note": ("non-periodic (u,v) window: Hann-windowed + detrended; "
                          "lowest-q_par bins are damped and semi-quantitative."),
        "mean_height": mean_h,
    }
    return WavinessResult(
        rms=rms, peak_to_peak=peak_to_peak,
        slope_mean=slope_mean, slope_rms=slope_rms, slope_max=slope_max,
        slope_percentiles=slope_percentiles,
        q_par=q_par, S_iso=S_iso, Cr_r=Cr_r, Cr=Cr, corr_length=corr_length,
        dominant_lambda=dominant_lambda, anisotropy_ratio=anisotropy_ratio,
        diagnostics=diagnostics,
    )


def _radial_average(P: np.ndarray, Qmag: np.ndarray, nbins: Optional[int] = None):
    q = Qmag.ravel()
    p = P.ravel()
    qmax = float(q.max())
    if nbins is None:
        nbins = max(8, min(P.shape) // 2)
    edges = np.linspace(0.0, qmax, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    which = np.clip(np.digitize(q, edges) - 1, 0, nbins - 1)
    sums = np.bincount(which, weights=p, minlength=nbins)
    counts = np.bincount(which, minlength=nbins).astype(float)
    S = np.zeros(nbins)
    nz = counts > 0
    S[nz] = sums[nz] / counts[nz]
    return centers, S


def _anisotropy(P: np.ndarray, QU: np.ndarray, QV: np.ndarray) -> float:
    """Ratio of spectral second moments along the two principal in-plane axes."""
    w = P.copy()
    w[P.shape[0] // 2, P.shape[1] // 2] = 0.0  # drop DC
    tot = w.sum()
    if tot <= 0:
        return 1.0
    Muu = float(np.sum(w * QU ** 2) / tot)
    Mvv = float(np.sum(w * QV ** 2) / tot)
    Muv = float(np.sum(w * QU * QV) / tot)
    cov = np.array([[Muu, Muv], [Muv, Mvv]])
    eig = np.linalg.eigvalsh(cov)
    eig = np.clip(eig, 1e-30, None)
    return float(np.sqrt(eig[-1] / eig[0]))


def _height_correlation(field: np.ndarray, du: float, dv: float):
    """Radially averaged height-height correlation C_h(r) via autocorrelation."""
    F = np.fft.fft2(field)
    ac = np.fft.ifft2(np.abs(F) ** 2).real / field.size
    ac = np.fft.fftshift(ac)
    nu, nv = field.shape
    ru = (np.arange(nu) - nu // 2) * du
    rv = (np.arange(nv) - nv // 2) * dv
    RU, RV = np.meshgrid(ru, rv, indexing="ij")
    Rmag = np.sqrt(RU ** 2 + RV ** 2)
    r_bins, C = _radial_average(ac, Rmag)
    C0 = C[0] if C[0] != 0 else 1.0
    Cn = C / C0
    # correlation length: first lag where Cn drops below 1/e
    corr_length = float("nan")
    below = np.where(Cn < np.exp(-1.0))[0]
    if below.size:
        i = below[0]
        if i > 0:
            # linear interpolation between i-1 and i
            r0, r1 = r_bins[i - 1], r_bins[i]
            c0, c1 = Cn[i - 1], Cn[i]
            corr_length = float(r0 + (np.exp(-1.0) - c0) * (r1 - r0) / (c1 - c0)) if c1 != c0 else r1
        else:
            corr_length = float(r_bins[i])
    return r_bins, Cn, corr_length


def _empty_waviness(du, dv, filled_fraction) -> WavinessResult:
    nan = float("nan")
    return WavinessResult(
        rms=nan, peak_to_peak=nan, slope_mean=nan, slope_rms=nan, slope_max=nan,
        slope_percentiles={"p50": nan, "p90": nan, "p99": nan},
        q_par=np.zeros(0), S_iso=np.zeros(0), Cr_r=np.zeros(0), Cr=np.zeros(0),
        corr_length=nan, dominant_lambda=nan, anisotropy_ratio=nan,
        diagnostics={"filled_fraction": filled_fraction, "n_measured": 0,
                     "note": "too few measured cells for waviness"},
    )
