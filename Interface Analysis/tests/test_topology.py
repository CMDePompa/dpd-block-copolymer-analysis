"""Periodic labeling, wrap detection, and Euler characteristic on known solids."""

from __future__ import annotations

import numpy as np
import pytest

from interface_analysis_v2.composition import CompositionField
from interface_analysis_v2.tests.synthetic import make_phi_grid
from interface_analysis_v2.topology import (
    compute_topology,
    euler_characteristic_periodic,
    expected_lamellae_gcd,
    periodic_label,
)


def _mask(shape=(16, 16, 16)):
    return np.zeros(shape, dtype=bool)


# ---------------------------------------------------------------------------
# Euler characteristic on known solids
# ---------------------------------------------------------------------------
def test_chi_single_voxel_is_1():
    m = _mask()
    m[4, 5, 6] = True
    assert euler_characteristic_periodic(m) == 1


def test_chi_two_disjoint_voxels_is_2():
    m = _mask()
    m[2, 2, 2] = True
    m[9, 9, 9] = True
    assert euler_characteristic_periodic(m) == 2


def test_chi_full_torus_is_0():
    assert euler_characteristic_periodic(np.ones((8, 8, 8), bool)) == 0


def test_chi_wrapping_slab_is_0():
    m = _mask()
    m[:, :, 4:8] = True          # T^2 x I
    assert euler_characteristic_periodic(m) == 0


def test_chi_solid_ball_is_1():
    m = _mask()
    m[4:12, 4:12, 4:12] = True
    assert euler_characteristic_periodic(m) == 1


def test_chi_hollow_shell_is_2():
    m = _mask()
    m[3:13, 3:13, 3:13] = True
    m[5:11, 5:11, 5:11] = False   # enclosed cavity
    assert euler_characteristic_periodic(m) == 2


def test_chi_solid_wrapping_tube_is_0():
    m = _mask()
    m[:, 6:10, 6:10] = True       # solid torus wrapping x
    assert euler_characteristic_periodic(m) == 0


def test_chi_slab_with_tunnel_is_minus_1():
    m = _mask()
    m[:, :, 4:8] = True
    m[6:10, 6:10, 4:8] = False    # punch a through-hole
    assert euler_characteristic_periodic(m) == -1


# ---------------------------------------------------------------------------
# Periodic labeling and wrap flags
# ---------------------------------------------------------------------------
def test_periodic_label_merges_across_boundary():
    from scipy import ndimage
    m = _mask()
    m[:, :, 14:] = True
    m[:, :, :2] = True            # one slab crossing the z boundary
    n_nonperiodic = ndimage.label(m)[1]
    assert n_nonperiodic == 2     # plain scipy splits it at the boundary...
    pc = periodic_label(m)
    assert pc.n_components == 1   # ...periodic labeling does not


def test_wrap_flags_slab_tube_blob():
    # slab normal to z: wraps x and y, not z
    m = _mask()
    m[:, :, 4:8] = True
    pc = periodic_label(m)
    assert pc.n_components == 1
    assert pc.wraps[0].tolist() == [True, True, False]

    # tube along x: wraps x only
    m = _mask()
    m[:, 6:10, 6:10] = True
    pc = periodic_label(m)
    assert pc.wraps[0].tolist() == [True, False, False]

    # blob: wraps nothing
    m = _mask()
    m[4:8, 4:8, 4:8] = True
    pc = periodic_label(m)
    assert pc.wraps[0].tolist() == [False, False, False]


def test_boundary_crossing_slab_wrap_flags():
    """A slab whose body crosses the z boundary still wraps only x and y."""
    m = _mask()
    m[:, :, 14:] = True
    m[:, :, :2] = True
    pc = periodic_label(m)
    assert pc.n_components == 1
    assert pc.wraps[0].tolist() == [True, True, False]


def test_roll_invariance_of_topology():
    """Rolling the field must not change any topology numbers."""
    phi, grid, truth = make_phi_grid(shape=(32, 32, 32), L=(16., 16., 16.),
                                     m=(0, 0, 2), lam=0.6, amp=0.9)
    m = phi > 0.0
    pc0 = periodic_label(m)
    m_roll = np.roll(m, (5, 11, 7), axis=(0, 1, 2))
    pc1 = periodic_label(m_roll)
    assert pc0.n_components == pc1.n_components
    np.testing.assert_array_equal(np.sort(pc0.volumes), np.sort(pc1.volumes))
    assert euler_characteristic_periodic(m) == euler_characteristic_periodic(m_roll)


def test_gcd_expected_components():
    assert expected_lamellae_gcd((0, 0, 2)) == 2
    assert expected_lamellae_gcd((2, -2, 0)) == 2
    assert expected_lamellae_gcd((2, -1, 1)) == 1
    assert expected_lamellae_gcd((2, 0, 2)) == 2
    assert expected_lamellae_gcd((3, 3, 0)) == 3
    assert expected_lamellae_gcd(None) is None


def test_topology_sweep_lamellar_synthetic():
    """Axis lamellae m=(0,0,2): at phi_c=0 the A set must be 2 slabs, each
    wrapping laterally, chi=0, no islands; bridge detection must fire when a
    bridge is added."""
    phi, grid, truth = make_phi_grid(shape=(32, 32, 32), L=(16., 16., 16.),
                                     m=(0, 0, 2), lam=0.5, amp=0.9)
    comp = CompositionField(phi=phi, invalid_mask=np.zeros(phi.shape, bool),
                            grid=grid, stats={}, warnings=[])
    res = compute_topology(comp, thresholds=[-0.4, 0.0, 0.4],
                           lattice_mode=truth["m"])
    rowsA0 = [r for r in res.rows if r["phase"] == "A" and r["phi_c"] == 0.0]
    assert len(rowsA0) == 1
    r = rowsA0[0]
    assert r["beta0"] == 2 == r["expected_components_perfect"]
    assert r["n_wrapping"] == 2 and r["n_islands"] == 0
    assert r["chi_total"] == 0 and r["n_excess_connections"] == 0

    # now weld a bridge between the two A slabs through one B layer
    phi_b = phi.copy()
    zc = np.argmin(phi_b[0, 0, :])          # center of a B layer
    phi_b[14:18, 14:18, :] = 0.95           # column bridging everything
    comp_b = CompositionField(phi=phi_b, invalid_mask=np.zeros(phi.shape, bool),
                              grid=grid, stats={}, warnings=[])
    res_b = compute_topology(comp_b, thresholds=[0.0], lattice_mode=truth["m"])
    rb = [r for r in res_b.rows if r["phase"] == "A"][0]
    assert rb["beta0"] == 1                  # slabs fused
    assert rb["n_excess_connections"] >= 1   # negative chi flags the bridges


def test_normalized_topology_handles_asymmetric_phase_plateaus():
    phi, grid, truth = make_phi_grid(shape=(32, 32, 32), L=(16., 16., 16.),
                                     m=(0, 0, 2), lam=0.5, amp=0.60,
                                     offset=0.25)
    comp = CompositionField(phi=phi, invalid_mask=np.zeros(phi.shape, bool),
                            grid=grid, stats={}, warnings=[])
    res = compute_topology(
        comp, thresholds=[0.0], lattice_mode=truth["m"],
        threshold_mode="normalized_plateaus",
    )
    row = [r for r in res.rows if r["phase"] == "A"][0]
    assert row["beta0"] == 2
    assert row["threshold_mode"] == "normalized_plateaus"
    assert row["phi_threshold_actual"] == pytest.approx(0.25, abs=0.08)
    assert row["connection_score"] == row["n_excess_connections"]


def test_topology_marks_large_invalid_fraction_unreliable():
    phi, grid, truth = make_phi_grid(shape=(24, 24, 24), L=(12., 12., 12.),
                                     m=(0, 0, 2), lam=0.5, amp=0.9)
    invalid = np.zeros(phi.shape, bool)
    invalid[:2] = True
    comp = CompositionField(phi=phi, invalid_mask=invalid,
                            grid=grid, stats={}, warnings=[])
    res = compute_topology(comp, thresholds=[0.0], lattice_mode=truth["m"])
    assert not all(r["topology_reliable"] for r in res.rows)
    assert any("invalid-density" in w for w in res.warnings)
