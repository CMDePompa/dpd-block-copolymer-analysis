"""Prepare individual interface height maps h_j(u, v) for geometric analysis.

An interface height map comes from :class:`interface_detection.InterfaceSet`
as a 2-D array over the (u, v) lateral grid, with NaN where the crossing was
missing. Downstream geometry (slopes, area, spectra, correlations) needs a
gap-filled, finite field. This module provides:

* :func:`fill_gaps` — nearest-neighbour + smooth infill of missing columns,
  reporting the filled fraction (large fractions invalidate the map);
* :func:`prepare_height_map` — returns a :class:`HeightMap` bundling the raw
  and filled fields, the valid mask, and the mean height.

Gap-filling is for well-formed maps with a few missing columns. When a large
fraction is missing (a genuinely defective/branched interface), the height
field is not a valid single-valued representation and callers should fall
back to the phase-field / marching-cubes geometry — flagged here, not hidden.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy import ndimage


@dataclass
class HeightMap:
    """A single interface's height field over the (u, v) plane."""

    h_raw: np.ndarray          # (nu, nv), NaN where missing
    h_filled: np.ndarray       # (nu, nv), finite everywhere
    valid: np.ndarray          # (nu, nv) bool
    mean_height: float
    filled_fraction: float
    du: float
    dv: float

    @property
    def dh(self) -> np.ndarray:
        """Mean-subtracted, gap-filled height fluctuation delta h(u, v)."""
        return self.h_filled - self.mean_height


def gradient_valid_mask(valid: np.ndarray) -> np.ndarray:
    """Cells whose finite-difference stencil contains only measured values.

    The lateral window is non-periodic.  Interior cells require both axial
    neighbours; edge cells require the one neighbour used by ``np.gradient``.
    This prevents gap-filled values from silently driving slopes and areas.
    """
    valid = np.asarray(valid, dtype=bool)
    if valid.ndim != 2:
        raise ValueError("gradient_valid_mask expects a 2-D mask")
    out = valid.copy()
    for axis in (0, 1):
        ok = np.zeros_like(valid)
        n = valid.shape[axis]
        if n < 2:
            return np.zeros_like(valid)
        first = [slice(None), slice(None)]
        second = [slice(None), slice(None)]
        first[axis] = 0
        second[axis] = 1
        ok[tuple(first)] = valid[tuple(first)] & valid[tuple(second)]
        last = [slice(None), slice(None)]
        penult = [slice(None), slice(None)]
        last[axis] = n - 1
        penult[axis] = n - 2
        ok[tuple(last)] = valid[tuple(last)] & valid[tuple(penult)]
        if n > 2:
            cur = [slice(None), slice(None)]
            prev = [slice(None), slice(None)]
            nxt = [slice(None), slice(None)]
            cur[axis] = slice(1, -1)
            prev[axis] = slice(0, -2)
            nxt[axis] = slice(2, None)
            ok[tuple(cur)] = (valid[tuple(prev)] & valid[tuple(cur)]
                              & valid[tuple(nxt)])
        out &= ok
    return out


def fill_gaps(h: np.ndarray, smooth: float = 1.0) -> Tuple[np.ndarray, float]:
    """Fill NaNs by nearest-valid assignment, lightly smoothing filled regions.

    Returns (filled, filled_fraction). If nothing is valid, returns zeros.
    The valid samples are preserved exactly; only NaN cells are replaced.
    """
    valid = np.isfinite(h)
    frac = 1.0 - float(np.mean(valid))
    if not valid.any():
        return np.zeros_like(h), 1.0
    if valid.all():
        return h.copy(), 0.0

    # Nearest valid value for every cell (periodic-agnostic; edges are lateral
    # window, not periodic in general — nearest-neighbour is the safe choice).
    idx = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    filled = h[tuple(idx)]

    # Blend a smoothed version into the FILLED cells only, so infilled regions
    # are not piecewise-constant while measured cells stay exact.
    if smooth > 0:
        sm = ndimage.gaussian_filter(filled, sigma=smooth, mode="nearest")
        filled = np.where(valid, h, sm)
    return filled, frac


def prepare_height_map(
    h: np.ndarray,
    du: float,
    dv: float,
    smooth_fill: float = 1.0,
) -> HeightMap:
    """Bundle a raw interface height field with a gap-filled version."""
    valid = np.isfinite(h)
    filled, frac = fill_gaps(h, smooth=smooth_fill)
    mean_h = float(np.nanmean(h)) if valid.any() else 0.0
    return HeightMap(h_raw=h, h_filled=filled, valid=valid,
                     mean_height=mean_h, filled_fraction=frac, du=du, dv=dv)
