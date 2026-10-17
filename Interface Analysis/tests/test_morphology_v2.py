"""Analytic regression tests for the V2 nonlamellar geometry."""

from __future__ import annotations

import numpy as np
import pandas as pd

from interface_analysis_v2.batch import run_batch
from interface_analysis_v2.composition import CompositionField
from interface_analysis_v2.config import load_config, validate_config
from interface_analysis_v2.density_field import DensityFields, GridSpec
from interface_analysis_v2.domain_geometry import analyze_domain_geometry
from interface_analysis_v2.normal_profiles import compute_normal_profiles
from interface_analysis_v2.surface_mesh import extract_surface_geometry
from interface_analysis_v2.trajectory import FrameData
from interface_analysis_v2.tests.synthetic import write_lammpstrj


def _grid_and_points(shape=(48, 48, 48), length=(24.0, 24.0, 24.0)):
    shape = tuple(shape)
    L = np.asarray(length, dtype=float)
    grid = GridSpec(
        box_lo=np.zeros(3), box_hi=L, shape=shape,
        voxel=L / np.asarray(shape, dtype=float),
    )
    axes = [grid.axis_centers(i) for i in range(3)]
    return grid, np.meshgrid(*axes, indexing="ij")


def _composition(phi, grid):
    return CompositionField(
        phi=np.asarray(phi, dtype=float), invalid_mask=np.zeros(grid.shape, dtype=bool),
        grid=grid,
        stats={
            "phi_mean": float(np.mean(phi)), "phi_rms": float(np.sqrt(np.mean(phi ** 2))),
            "phi_min": float(np.min(phi)), "phi_max": float(np.max(phi)),
            "fraction_invalid_voxels": 0.0,
        },
        warnings=[],
    )


def _sphere(radius=5.0, width=0.8):
    grid, (x, y, z) = _grid_and_points()
    center = 0.5 * grid.L
    r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2)
    return _composition(np.tanh((radius - r) / width), grid)


def _cylinder(radius=4.0, width=0.8):
    grid, (x, y, _z) = _grid_and_points()
    center = 0.5 * grid.L
    r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    return _composition(np.tanh((radius - r) / width), grid)


def test_config_morphology_is_explicit_and_validated():
    cfg = load_config(None)
    assert cfg["morphology"] == "lamella"
    cfg["morphology"] = "gyroid"
    validate_config(cfg)
    cfg["morphology"] = "not-a-morphology"
    try:
        validate_config(cfg)
    except ValueError as exc:
        assert "morphology" in str(exc)
    else:
        raise AssertionError("invalid morphology should fail validation")


def test_periodic_sphere_area_curvature_and_domain_geometry():
    radius = 5.0
    comp = _sphere(radius=radius)
    surface = extract_surface_geometry(comp)
    assert np.isclose(surface.area, 4.0 * np.pi * radius ** 2, rtol=0.09)
    assert np.isclose(surface.H_abs_mean, 1.0 / radius, rtol=0.14)
    assert np.isclose(surface.K_mean, 1.0 / radius ** 2, rtol=0.22)
    assert np.isclose(surface.gauss_bonnet_solid_chi_estimate, 1.0, atol=0.18)
    assert np.allclose(surface.normal_tensor_eigenvalues, np.full(3, 1.0 / 3.0), atol=0.035)

    cfg = load_config(None)
    cfg.update({"morphology": "sphere", "minority_phase": "A"})
    domains = analyze_domain_geometry(comp, surface, cfg)
    assert domains.metrics["domain_component_count"] == 1
    assert domains.metrics["domain_wrapping_component_count"] == 0
    assert domains.metrics["domain_interface_level_chi"] == 1
    assert np.isclose(domains.metrics["sphere_radius_volume_area"], radius, rtol=0.12)


def test_periodic_cylinder_area_curvature_and_axis():
    radius = 4.0
    comp = _cylinder(radius=radius)
    surface = extract_surface_geometry(comp)
    expected_area = 2.0 * np.pi * radius * comp.grid.L[2]
    assert np.isclose(surface.area, expected_area, rtol=0.08)
    assert np.isclose(surface.H_abs_mean, 1.0 / (2.0 * radius), rtol=0.16)
    assert abs(surface.K_mean) < 0.01
    assert surface.normal_tensor_eigenvalues[0] < 0.02
    axis = surface.normal_tensor_eigenvectors[:, 0]
    assert abs(axis[2]) > 0.98

    cfg = load_config(None)
    cfg.update({"morphology": "cylinder", "minority_phase": "A"})
    domains = analyze_domain_geometry(comp, surface, cfg)
    assert domains.metrics["domain_wrapping_component_count"] == 1
    assert domains.metrics["domain_interface_level_chi"] == 0
    assert np.isclose(domains.metrics["cylinder_radius_volume_area"], radius, rtol=0.10)


def test_gyroid_level_surface_is_saddle_dominated_and_triply_connected():
    n = 48
    repeats = 2
    L = np.full(3, 2.0 * np.pi * repeats)
    grid, (x, y, z) = _grid_and_points((n, n, n), tuple(L))
    g = (np.sin(x) * np.cos(y) + np.sin(y) * np.cos(z)
         + np.sin(z) * np.cos(x))
    comp = _composition(np.tanh(g / 0.45), grid)
    surface = extract_surface_geometry(comp)
    assert surface.K_negative_area_fraction > 0.80

    cfg = load_config(None)
    cfg.update({"morphology": "gyroid", "minority_phase": "A"})
    domains = analyze_domain_geometry(comp, surface, cfg)
    assert domains.metrics["domain_fully_wrapping_component_count"] >= 1
    assert domains.metrics["domain_interface_level_chi"] < 0


def test_local_normal_sphere_profile_recovers_intrinsic_width():
    true_lambda = 0.8
    comp = _sphere(radius=5.0, width=true_lambda)
    surface = extract_surface_geometry(comp)
    rho_total = 3.0
    fields = DensityFields(
        rho_A=0.5 * rho_total * (1.0 + comp.phi),
        rho_B=0.5 * rho_total * (1.0 - comp.phi),
        grid=comp.grid, diagnostics={}, warnings=[],
    )
    frame = FrameData(
        timestep=0, box_lo=comp.grid.box_lo, box_hi=comp.grid.box_hi,
        types=np.zeros(0, dtype=np.int32), coords=np.zeros((0, 3)),
    )
    cfg = load_config(None)
    cfg.update({
        "morphology": "sphere", "surface_normal_samples": 384,
        "normal_profile_points": 65, "normal_profile_halfwidth": 3.0,
        "smoothing_sigma": 0.0,
    })
    result = compute_normal_profiles(frame, fields, comp, surface, 12.0, cfg)
    assert result.metrics["normal_width_failed_fraction"] < 0.1
    assert np.isclose(result.metrics["normal_width_lam_mean"], true_lambda, rtol=0.10)
    expected_erf_sigma = np.pi * true_lambda / np.sqrt(12.0)
    assert np.isclose(
        result.metrics["normal_width_derivative_moment_sigma_pooled"],
        expected_erf_sigma, rtol=0.15,
    )


def test_sphere_route_runs_end_to_end(tmp_path):
    """The batch router must produce the nonlamellar V2 output contract."""
    rng = np.random.default_rng(2409)
    length = 16.0
    n_atoms = int(3.0 * length ** 3)
    coords = rng.uniform(0.0, length, size=(n_atoms, 3))
    delta = coords - 0.5 * length
    radius = np.linalg.norm(delta, axis=1)
    probability_A = 0.03 + 0.94 / (1.0 + np.exp((radius - 3.5) / 0.45))
    types = np.where(rng.random(n_atoms) < probability_A, 1, 2).astype(np.int32)
    frame = FrameData(
        timestep=5000, box_lo=np.zeros(3), box_hi=np.full(3, length),
        types=types, coords=coords,
    )
    traj = tmp_path / "synthetic_sphere.prod.lammpstrj"
    write_lammpstrj(traj, [frame], include_mol=False)
    cfg = load_config(None)
    cfg.update({
        "morphology": "sphere", "minority_phase": "A",
        "input_dir": str(tmp_path), "output_dir": str(tmp_path / "out"),
        "analysis_start_step": None, "max_frames": 1, "grid_shape": [32, 32, 32],
        "surface_normal_samples": 128, "normal_profile_points": 33,
        "plots": False, "save_maps": False,
    })
    validate_config(cfg)
    combined = run_batch(cfg, echo=None)
    summary = pd.read_csv(combined).iloc[0]
    assert summary["status"] == "ok"
    assert summary["morphology"] == "sphere"
    assert summary["n_frames_analyzed"] == 1
    sim_out = tmp_path / "out" / "synthetic_sphere"
    for name in (
        "normal_profiles.csv", "normal_profiles_per_frame.csv",
        "domain_components_per_frame.csv", "topology_by_threshold.csv",
        "interface_surface_last.npz",
    ):
        # save_maps=false intentionally suppresses only maps/surface archives.
        if name == "interface_surface_last.npz":
            assert not (sim_out / name).exists()
        else:
            assert (sim_out / name).is_file()
