"""Resample the composition field onto an oblique (u, v, w) grid with w || n0.

Lamellae in these systems are NOT axis-aligned (all three test systems have
oblique normals), so interface heights must be measured along n0, not along a
box axis. We build an orthonormal frame (e_u, e_v, e_w = n0) and sample phi on
a regular (u, v, w) lattice centered in the box. Sampling uses
``scipy.ndimage.map_coordinates`` with ``mode='grid-wrap'`` — the exact
periodic interpolation mode — so columns that leave one face re-enter the
opposite face correctly.

Periodicity of the transformed block:
  * Along w (|| n0), an integer number of lamellar periods is NOT generally
    a periodic translation of the simulation box.  We only treat w as
    circular when W*n0 is an integer box-vector.  Otherwise downstream code
    uses a bounded column and discards measurements whose fitting window
    would cross an edge.
  * Along u, v (the lattice PLANE perpendicular to n0): a general oblique
    plane does not tile the cubic box with the sampled square window, so the
    (u, v) window is treated as NON-periodic. Height spectra and correlations
    therefore apply a 2D window/detrend (see waviness.py) and this caveat is
    recorded. For axis-aligned or simple-diagonal normals the window can be
    exactly periodic; we do not rely on that.

Interpolation order, wrapping, and the (non-)preservation of voxel spacing
are all recorded in the returned diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import map_coordinates

from .composition import CompositionField
from .config import is_auto
from .density_field import GridSpec
from .io_utils import LogFn


@dataclass
class SlabField:
    """phi sampled on an oblique regular grid aligned with n0."""

    phi_uvw: np.ndarray            # (nu, nv, nw)
    u: np.ndarray                  # (nu,) physical offsets along e_u
    v: np.ndarray                  # (nv,) physical offsets along e_v
    w: np.ndarray                  # (nw,) physical offsets along e_w == n0
    e_u: np.ndarray                # (3,)
    e_v: np.ndarray
    e_w: np.ndarray                # == n0
    origin: np.ndarray            # (3,) physical position of (u,v,w)=(0,0,0)
    du: float
    dv: float
    dw: float
    L0: float
    n_periods: int
    W: float                       # sampled normal extent (== n_periods * L0)
    w_periodic: bool               # True only when W*e_w is a box-lattice vector
    diagnostics: Dict[str, Any]
    warnings: List[str]

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.phi_uvw.shape

    def column_positions(self, iu: int, iv: int) -> np.ndarray:
        """Physical (N, 3) positions of the sample column at lateral index (iu, iv)."""
        base = self.origin + self.u[iu] * self.e_u + self.v[iv] * self.e_v
        return base[None, :] + self.w[:, None] * self.e_w[None, :]


def sample_on_slab(
    field: np.ndarray,
    grid: GridSpec,
    slab: "SlabField",
    order: int = 1,
) -> np.ndarray:
    """Sample an arbitrary grid field (e.g. rho_A, rho_B) on an existing slab's
    (u, v, w) sample points, with exact periodic wrapping.

    Guarantees geometric consistency: penetration profiles use the SAME
    columns and w positions as the interface detection that produced h_j.
    """
    U, V, Wm = np.meshgrid(slab.u, slab.v, slab.w, indexing="ij")
    pts = (slab.origin[None, :]
           + U.ravel()[:, None] * slab.e_u[None, :]
           + V.ravel()[:, None] * slab.e_v[None, :]
           + Wm.ravel()[:, None] * slab.e_w[None, :])
    idx_coords = grid.points_to_index_coords(pts)
    vals = map_coordinates(field, idx_coords, order=order, mode="grid-wrap")
    return vals.reshape(slab.shape)


def build_frame(n0: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Right-handed orthonormal frame (e_u, e_v, e_w) with e_w = n0.

    e_u is chosen deterministically as the normalized rejection of the world
    axis least parallel to n0, so the frame is reproducible across frames.
    """
    e_w = np.asarray(n0, dtype=float)
    e_w = e_w / np.linalg.norm(e_w)
    world = np.eye(3)
    seed = world[int(np.argmin(np.abs(e_w)))]     # axis most perpendicular to n0
    e_u = seed - np.dot(seed, e_w) * e_w
    e_u /= np.linalg.norm(e_u)
    e_v = np.cross(e_w, e_u)
    e_v /= np.linalg.norm(e_v)
    return e_u, e_v, e_w


def _resolve_extent(value: Any, default: float) -> float:
    return float(default) if is_auto(value) or value is None else float(value)


def resample_slab(
    comp: CompositionField,
    n0: np.ndarray,
    L0: float,
    cfg: Dict[str, Any],
    log: Optional[LogFn] = None,
    lattice_mode: Optional[Tuple[int, int, int]] = None,
) -> SlabField:
    """Sample phi on an oblique grid aligned with n0.

    Extents/steps come from cfg (all support 'auto'):
      lateral_voxel_size (du, dv), normal_sample_size (dw),
      lateral_extent (u, v window), n_periods (w window = n_periods * L0).
    """
    grid: GridSpec = comp.grid
    L = grid.L
    n0 = np.asarray(n0, float)
    n0 = n0 / np.linalg.norm(n0)
    e_u, e_v, e_w = build_frame(n0)

    base_vox = float(np.min(grid.voxel))
    du = _resolve_extent(cfg.get("lateral_voxel_size"), cfg["target_voxel_size"])
    dv = du
    dw = _resolve_extent(cfg.get("normal_sample_size"), cfg["target_voxel_size"] / 2.0)

    lateral_extent = _resolve_extent(cfg.get("lateral_extent"), float(np.median(L)))

    # The raw FFT lattice mode has an exact box-compatible wavelength.  The
    # centroid-refined FFT magnitude is useful as a diagnostic, but using it
    # to close an oblique sampling line introduces a false seam.
    L0_input = float(L0)
    L0_lattice = None
    closure_periods = None
    if lattice_mode is not None:
        m = np.asarray(lattice_mode, dtype=float)
        q_lat = 2.0 * np.pi * m / L
        q_lat_mag = float(np.linalg.norm(q_lat))
        if q_lat_mag > 0:
            L0_lattice = 2.0 * np.pi / q_lat_mag
            L0 = L0_lattice
    # Find the smallest number of periods whose displacement is an integer
    # multiple of every box edge.  This also supports direct callers that do
    # not supply lattice_mode. A closure need not exist for a general
    # orthorhombic box, so the search is bounded.
    if L0 > 0:
        one_period_box = L0 * n0 / L
        # Longer closures create unnecessarily large slabs; bounded sampling
        # is safer beyond 32 periods.
        for p in range(1, 33):
            if np.max(np.abs(p * one_period_box - np.rint(p * one_period_box))) < 1e-8:
                closure_periods = p
                break

    if is_auto(cfg.get("n_periods")) or cfg.get("n_periods") is None:
        if closure_periods is not None:
            n_periods = max(2, int(closure_periods))
            if n_periods % closure_periods:
                n_periods = 2 * closure_periods
        else:
            # No exact short closure: cover roughly the box projection and
            # use a non-periodic column downstream.
            span_along_n0 = float(np.sum(np.abs(n0) * L))
            n_periods = max(3, int(round(span_along_n0 / L0)) + 1) if L0 > 0 else 3
    else:
        n_periods = max(2, int(cfg["n_periods"]))

    W = n_periods * L0
    nu = max(4, int(round(lateral_extent / du)))
    nv = max(4, int(round(lateral_extent / dv)))
    nw = max(8, int(round(W / dw)))
    # Uniform w grid.  It is circular only if its total displacement closes
    # on the original periodic box, checked below rather than assumed.
    dw = W / nw
    u = (np.arange(nu) - (nu - 1) / 2.0) * du
    v = (np.arange(nv) - (nv - 1) / 2.0) * dv
    w = (np.arange(nw) - nw / 2.0) * dw          # unsampled endpoint is w[0] + W
    origin = grid.box_lo + 0.5 * L

    # Build all sample positions (nu*nv*nw, 3) and interpolate periodically.
    U, V, Wm = np.meshgrid(u, v, w, indexing="ij")
    pts = (origin[None, :]
           + U.ravel()[:, None] * e_u[None, :]
           + V.ravel()[:, None] * e_v[None, :]
           + Wm.ravel()[:, None] * e_w[None, :])
    idx_coords = grid.points_to_index_coords(pts)
    order = int(cfg.get("interp_order", 3))
    phi_flat = map_coordinates(comp.phi, idx_coords, order=order, mode="grid-wrap")
    phi_uvw = phi_flat.reshape(nu, nv, nw)

    warnings: List[str] = []
    box_shift = W * n0 / L
    box_shift_integer = np.rint(box_shift)
    closure_error = float(np.max(np.abs(box_shift - box_shift_integer)))
    w_periodic = closure_error < 1e-8
    if not w_periodic:
        warnings.append(
            "the sampled normal window does not close on a periodic box translation; "
            "w is treated as non-periodic and edge-crossing measurements are excluded."
        )
    if dw > 0.5 * L0 / 2.0:
        warnings.append(
            f"normal_sample_size dw={dw:.3f} is coarse vs interface region "
            f"(L0/4={L0 / 4:.3f}); crossing positions and widths will be under-resolved."
        )
    # Nyquist check: sampling finer than the source grid does not add
    # information; sampling coarser than the grid can alias.
    if dw > base_vox:
        warnings.append(
            f"normal_sample_size dw={dw:.3f} exceeds source voxel {base_vox:.3f}: "
            "the oblique sampling is coarser than the density grid (aliasing risk)."
        )

    diagnostics: Dict[str, Any] = {
        "e_u": e_u.tolist(), "e_v": e_v.tolist(), "e_w": e_w.tolist(),
        "origin": origin.tolist(),
        "du": du, "dv": dv, "dw": dw,
        "nu": nu, "nv": nv, "nw": nw,
        "lateral_extent": lateral_extent,
        "n_periods": n_periods,
        "W_normal_extent": W,
        "w_period": W if w_periodic else None,
        "L0_used": L0,
        "L0_input_centroid": L0_input,
        "L0_lattice": L0_lattice,
        "closure_periods": closure_periods,
        "normal_window_box_shift": box_shift.tolist(),
        "normal_window_box_shift_integer": box_shift_integer.astype(int).tolist(),
        "normal_closure_error": closure_error,
        "interp_order": order,
        "interp_mode": "grid-wrap (exact periodic)",
        "source_voxel": grid.voxel.tolist(),
        "preserves_source_voxel_spacing": False,
        "w_periodic": w_periodic,
        "uv_periodic": False,
        "uv_periodic_note": ("the oblique lattice plane is not tiled by the square (u,v) "
                             "window in general; spectra/correlations window+detrend the plane"),
    }
    if log:
        log(f"slab: shape=({nu},{nv},{nw}), du={du:.3f}, dw={dw:.3f}, "
            f"n_periods={n_periods}, W={W:.2f}, order={order}")
        for wmsg in warnings:
            log(f"WARN: {wmsg}")

    return SlabField(
        phi_uvw=phi_uvw, u=u, v=v, w=w, e_u=e_u, e_v=e_v, e_w=e_w, origin=origin,
        du=du, dv=dv, dw=dw, L0=float(L0), n_periods=n_periods, W=float(W),
        w_periodic=w_periodic, diagnostics=diagnostics, warnings=warnings,
    )
