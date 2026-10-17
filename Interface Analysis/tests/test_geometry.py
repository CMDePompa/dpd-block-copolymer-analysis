"""Waviness, spacing, and surface-geometry tests against analytic ground truth."""

from __future__ import annotations

import numpy as np
import pytest

from interface_analysis_v2.composition import CompositionField
from interface_analysis_v2.config import load_config
from interface_analysis_v2.height_fields import prepare_height_map
from interface_analysis_v2.interface_detection import detect_interfaces
from interface_analysis_v2.interface_detection import COL_GOOD, InterfaceSet
from interface_analysis_v2.orientation import resolve_orientation
from interface_analysis_v2.resample import resample_slab
from interface_analysis_v2.surface_geometry import (
    compute_area,
    compute_orientational_order,
    height_field_area,
)
from interface_analysis_v2.spacing import compute_spacing
from interface_analysis_v2.tests.synthetic import make_phi_grid
from interface_analysis_v2.waviness import compute_waviness


def _comp(phi, grid):
    return CompositionField(phi=phi, invalid_mask=np.zeros(phi.shape, bool), grid=grid,
                            stats={}, warnings=[])


def test_waviness_rms_of_known_sinusoid():
    """A pure sinusoidal height field: RMS = amp/sqrt(2), one spectral peak."""
    nu, nv = 64, 64
    du = dv = 0.5
    u = (np.arange(nu) - nu / 2) * du
    amp, kx = 1.5, 3
    Lu = nu * du
    h = amp * np.cos(2 * np.pi * kx * u / Lu)[:, None] * np.ones((1, nv))
    hmap = prepare_height_map(h, du, dv)
    wav = compute_waviness(hmap, detrend=True, window=False)
    assert wav.rms == pytest.approx(amp / np.sqrt(2), rel=0.05)
    assert wav.peak_to_peak == pytest.approx(2 * amp, rel=0.05)
    # dominant wavelength = Lu / kx
    assert wav.dominant_lambda == pytest.approx(Lu / kx, rel=0.15)
    # strongly anisotropic (varies only along u)
    assert wav.anisotropy_ratio > 3.0


def test_waviness_flat_is_zero():
    h = np.full((40, 40), 5.0)
    wav = compute_waviness(prepare_height_map(h, 0.5, 0.5))
    assert wav.rms == pytest.approx(0.0, abs=1e-9)
    assert wav.slope_rms == pytest.approx(0.0, abs=1e-9)


def test_slope_of_known_tilt():
    """A constant-slope ramp has |grad h| equal to the analytic slope."""
    nu, nv = 50, 50
    du = dv = 0.4
    slope = 0.3
    u = np.arange(nu) * du
    h = slope * u[:, None] * np.ones((1, nv))
    wav = compute_waviness(prepare_height_map(h, du, dv), detrend=True)
    assert wav.slope_mean == pytest.approx(slope, rel=0.02)
    assert wav.slope_max == pytest.approx(slope, rel=0.05)


def test_excess_area_of_known_surface():
    """For h = s*u (constant slope s), dA = sqrt(1+s^2), so excess = sqrt(1+s^2)-1."""
    nu, nv = 60, 60
    du = dv = 0.5
    s = 0.5
    u = np.arange(nu) * du
    h = s * u[:, None] * np.ones((1, nv))
    hmap = prepare_height_map(h, du, dv)
    area, excess = height_field_area(hmap)
    assert excess == pytest.approx(np.sqrt(1 + s**2) - 1, rel=0.02)


def test_spacing_recovers_L0_flat_lamellae():
    phi, grid, truth = make_phi_grid(shape=(48, 48, 48), L=(24., 24., 24.),
                                     m=(0, 0, 2), lam=0.7, amp=0.95)
    cfg = load_config(None)
    comp = _comp(phi, grid)
    ori = resolve_orientation(comp, cfg, sq_q_star=truth["q_mag"])
    slab = resample_slab(comp, ori.n0, ori.L0, cfg)
    iset = detect_interfaces(slab, cfg)
    sp = compute_spacing(iset, L0_sq=truth["L0"])
    assert sp.mean == pytest.approx(truth["L0"], rel=0.05)
    assert sp.cv < 0.05                       # flat lamellae: near-uniform spacing
    assert sp.rel_diff_vs_sq < 0.05


def test_orientational_order_flat_lamellae_is_high():
    """Perfectly stacked flat lamellae: all normals || n0, so S_n ~ 1, angle ~ 0."""
    phi, grid, truth = make_phi_grid(shape=(48, 48, 48), L=(24., 24., 24.),
                                     m=(0, 0, 2), lam=0.7, amp=0.95)
    comp = _comp(phi, grid)
    res = compute_orientational_order(comp, truth["n0"], band=0.5)
    assert res.S_n > 0.9
    assert res.mean_angle_deg < 12.0


def test_orientational_order_oblique_normal():
    """S_n must be near 1 when measured against the CORRECT oblique n0,
    and much lower against a wrong axis — proving it is n0-relative."""
    phi, grid, truth = make_phi_grid(shape=(44, 44, 44), L=(22., 22., 22.),
                                     m=(2, 0, 2), lam=0.7, amp=0.95)
    comp = _comp(phi, grid)
    right = compute_orientational_order(comp, truth["n0"], band=0.5)
    wrong = compute_orientational_order(comp, np.array([1.0, 0.0, 0.0]), band=0.5)
    assert right.S_n > 0.9
    assert wrong.S_n < right.S_n - 0.3


def test_marching_cubes_area_crosscheck_runs():
    phi, grid, truth = make_phi_grid(shape=(40, 40, 40), L=(20., 20., 20.),
                                     m=(0, 0, 2), lam=0.7, amp=0.95)
    comp = _comp(phi, grid)
    cfg = load_config(None)
    ori = resolve_orientation(comp, cfg, sq_q_star=truth["q_mag"])
    slab = resample_slab(comp, ori.n0, ori.L0, cfg)
    iset = detect_interfaces(slab, cfg)
    hmaps = [prepare_height_map(iset.h[j], slab.du, slab.dv)
             for j in range(iset.n_interfaces) if iset.valid_fraction[j] > 0.9]
    res = compute_area(hmaps, comp=comp, box_volume=float(np.prod(grid.L)))
    assert res.marching_cubes_area is not None and res.marching_cubes_area > 0
    # 4 flat interfaces of area ~ L^2 each => total ~ 4 * 400 = 1600; both
    # estimates should be within a factor of ~1.5 of that geometry.
    expected = 2 * 2 * (20.0 ** 2)   # 2 interfaces/period * 2 periods * area
    assert 0.6 * expected < res.marching_cubes_area < 1.6 * expected
    assert np.isfinite(res.total_height_field_area)


def test_height_area_rejects_heavily_infilled_surface():
    h = np.full((20, 20), np.nan)
    h[:10] = 1.0
    area, excess = height_field_area(prepare_height_map(h, 0.5, 0.5),
                                     max_filled_fraction=0.2)
    assert np.isnan(area) and np.isnan(excess)


def test_spacing_includes_periodic_wrap_pairs_and_uses_clean_columns():
    h = np.empty((4, 3, 3), dtype=float)
    for j, value in enumerate((0.0, 2.0, 5.0, 8.0)):
        h[j] = value
    classes = np.full((3, 3), COL_GOOD, dtype=np.int32)
    iset = InterfaceSet(
        h=h, plane_type=np.array([1, -1, 1, -1]),
        plane_ideal=np.array([0., 2., 5., 8.]),
        plane_mean=np.array([0., 2., 5., 8.]),
        valid_fraction=np.ones(4), column_class=classes,
        expected_crossings=4, crossing_count=np.full((3, 3), 4),
        bad_column_fraction=0.0, class_fractions={"good": 1.0},
        n_interfaces=4, n_valid_interfaces=4, W=10.0, w_periodic=True,
        diagnostics={}, warnings=[],
    )
    sp = compute_spacing(iset)
    assert len(sp.local_maps) == 4
    assert sp.mean == pytest.approx(5.0)
    assert sp.clean_coverage == pytest.approx(1.0)
