"""Density deposition, smoothing, and composition-field tests.

Key invariants: exact mass conservation, exact periodicity (roll invariance),
convergence of CIC+Gaussian toward the true periodic Gaussian KDE, and the
composition field reproducing a known lamellar law.
"""

from __future__ import annotations

import numpy as np
import pytest

from interface_analysis_v2.composition import composition_field, sample_field_at_points
from interface_analysis_v2.density_field import (
    GridSpec,
    build_density_fields,
    deposit_counts,
    resolve_grid,
    smooth_field,
)
from interface_analysis_v2.tests.synthetic import make_lamellar_frame


def _uniform_coords(n, L, seed=0):
    return np.random.default_rng(seed).uniform(0, L, size=(n, 3))


def test_mass_conservation_and_mean_density():
    L, n = 16.0, 5000
    grid, _ = resolve_grid(np.zeros(3), np.full(3, L), 0.5)
    for method in ("cic", "ngp"):
        counts = deposit_counts(_uniform_coords(n, L), grid, method=method)
        assert abs(counts.sum() - n) < 1e-8 * n
        sm = smooth_field(counts, 0.9, grid)
        assert abs(sm.sum() - n) < 1e-8 * n           # periodic smoothing preserves mass
        rho = sm / grid.voxel_volume
        assert np.isclose(rho.mean(), n / L**3, rtol=1e-10)


def test_periodicity_roll_invariance():
    """Shifting all particles by an integer number of voxels rolls the field."""
    L = 12.0
    grid, _ = resolve_grid(np.zeros(3), np.full(3, L), 0.5)
    coords = _uniform_coords(2000, L, seed=3)
    shift_vox = np.array([5, 0, 11])
    shifted = np.mod(coords + shift_vox * grid.voxel, L)

    for method in ("cic", "ngp"):
        c0 = deposit_counts(coords, grid, method=method)
        c1 = deposit_counts(shifted, grid, method=method)
        np.testing.assert_allclose(c1, np.roll(c0, shift_vox, axis=(0, 1, 2)), atol=1e-9)

    s0 = smooth_field(deposit_counts(coords, grid), 0.8, grid)
    s1 = smooth_field(deposit_counts(shifted, grid), 0.8, grid)
    np.testing.assert_allclose(s1, np.roll(s0, shift_vox, axis=(0, 1, 2)), atol=1e-9)


def test_cic_gaussian_converges_to_true_kde():
    """CIC + gaussian_filter must approach the exact periodic Gaussian KDE
    as the grid is refined (validates the chosen field-construction method)."""
    L, sigma, n = 10.0, 1.0, 300
    coords = _uniform_coords(n, L, seed=5)

    def exact_kde(grid: GridSpec) -> np.ndarray:
        axes = [grid.axis_centers(i) for i in range(3)]
        X, Y, Z = np.meshgrid(*axes, indexing="ij")
        rho = np.zeros(grid.shape)
        norm = (2 * np.pi * sigma**2) ** 1.5
        for p in coords:
            acc = np.zeros(grid.shape)
            for img in (-1, 0, 1):          # periodic images (L/sigma = 10 -> plenty)
                dx = X - p[0] + img * L
                for imgy in (-1, 0, 1):
                    dy = Y - p[1] + imgy * L
                    for imgz in (-1, 0, 1):
                        dz = Z - p[2] + imgz * L
                        acc += np.exp(-(dx**2 + dy**2 + dz**2) / (2 * sigma**2))
            rho += acc
        return rho / norm

    errs = []
    for nvox in (10, 20):
        grid, _ = resolve_grid(np.zeros(3), np.full(3, L), L / nvox)
        approx = smooth_field(deposit_counts(coords, grid), sigma, grid) / grid.voxel_volume
        exact = exact_kde(grid)
        errs.append(np.sqrt(np.mean((approx - exact) ** 2)) / exact.std())
    assert errs[1] < errs[0]          # refining the grid reduces the error
    assert errs[1] < 0.10             # and the fine-grid field is close to exact


def test_lamellar_frame_reproduces_composition_law():
    """End-to-end fields: particle synthetic with known phi law."""
    frame, truth = make_lamellar_frame(L=20.0, rho=3.0, m=(0, 0, 2),
                                       lam=1.0, fA_B_block=0.2, seed=11)
    fields = build_density_fields(frame, 1, 2, target_voxel_size=0.5,
                                  smoothing_sigma=0.75)
    assert fields.diagnostics["mass_conservation_rel_err_A"] < 1e-10
    comp = composition_field(fields, 0.05)
    assert comp.stats["fraction_invalid_voxels"] == 0.0

    # Dominant non-DC FFT mode must be the imposed (0, 0, +/-2)
    F = np.abs(np.fft.fftn(comp.phi - comp.phi.mean())) ** 2
    F[0, 0, 0] = 0.0
    kx, ky, kz = np.unravel_index(np.argmax(F), F.shape)
    n = comp.phi.shape
    mode = tuple(int(k if k <= nn // 2 else k - nn) for k, nn in zip((kx, ky, kz), n))
    assert mode in ((0, 0, 2), (0, 0, -2))

    # phi averaged deep inside each lamellar phase reproduces the known
    # offset +/- amp plateaus (means over many voxels suppress counting noise;
    # pointwise extremes would not).
    g = fields.grid
    axes = [g.axis_centers(i) for i in range(3)]
    X, Y, Z = np.meshgrid(*axes, indexing="ij")
    s = np.sin(truth["q_vec"][0] * X + truth["q_vec"][1] * Y
               + truth["q_vec"][2] * Z + truth["phase"])
    plateau_A = float(comp.phi[s > 0.95].mean())
    plateau_B = float(comp.phi[s < -0.95].mean())
    # exact analytic expectation over the same masks (the tanh profile does
    # not fully saturate for lam ~ 1/q, so comparing to +/-amp would be wrong)
    phi_true = (truth["expected_offset"]
                + truth["expected_amp"] * np.tanh(s / (truth["q_mag"] * truth["lam"])))
    assert plateau_A == pytest.approx(float(phi_true[s > 0.95].mean()), abs=0.06)
    assert plateau_B == pytest.approx(float(phi_true[s < -0.95].mean()), abs=0.06)
    assert comp.stats["phi_mean"] == pytest.approx(truth["expected_offset"], abs=0.05)


def test_density_cutoff_masks_sparse_regions():
    """A nearly empty system with tiny smoothing must trigger the invalid-voxel path."""
    rng = np.random.default_rng(2)
    from interface_analysis_v2.trajectory import FrameData
    frame = FrameData(timestep=0, box_lo=np.zeros(3), box_hi=np.full(3, 20.0),
                      types=np.array([1, 2] * 50, dtype=np.int32),
                      coords=rng.uniform(0, 20, size=(100, 3)))
    fields = build_density_fields(frame, 1, 2, target_voxel_size=0.5, smoothing_sigma=0.5)
    assert fields.warnings                      # bead-per-volume warning fires
    comp = composition_field(fields, 0.5)
    assert comp.stats["fraction_invalid_voxels"] > 0.01
    assert comp.warnings                        # masking warning fires
    assert np.all(comp.phi[comp.invalid_mask] == 0.0)


def test_sample_field_at_points_periodic():
    grid, _ = resolve_grid(np.zeros(3), np.full(3, 8.0), 0.5)
    x = grid.axis_centers(0)
    field = np.broadcast_to(np.cos(2 * np.pi * x / 8.0)[:, None, None], grid.shape).copy()
    pts = np.array([[1.25, 4.0, 4.0], [1.25 + 8.0, 4.0, 4.0], [1.25 - 16.0, 4.0, 4.0]])
    vals = sample_field_at_points(field, grid, pts, order=1)
    assert vals[0] == pytest.approx(vals[1], abs=1e-12)   # +L wraps
    assert vals[0] == pytest.approx(vals[2], abs=1e-12)   # -2L wraps
    # voxel-center convention: sampling exactly at a center returns the value
    center_pt = np.array([[x[3], grid.axis_centers(1)[2], grid.axis_centers(2)[5]]])
    v = sample_field_at_points(field, grid, center_pt, order=1)
    assert v[0] == pytest.approx(field[3, 2, 5], abs=1e-12)


def test_freud_gaussian_density_crosscheck():
    """Optional consistency check against freud.density.GaussianDensity."""
    freud = pytest.importorskip("freud")
    L, sigma = 12.0, 0.8
    frame, _ = make_lamellar_frame(L=L, rho=3.0, m=(0, 0, 2), lam=0.8, seed=4)
    grid, _ = resolve_grid(frame.box_lo, frame.box_hi, 0.5)

    mine = smooth_field(deposit_counts(frame.coords, grid), sigma, grid) / grid.voxel_volume

    box = freud.box.Box.cube(L)
    gd = freud.density.GaussianDensity(grid.shape, r_max=min(4 * sigma, L / 2 - 1e-3), sigma=sigma)
    pts = (frame.coords - L / 2.0).astype(np.float32)     # freud boxes are centered
    gd.compute(system=(box, pts))
    theirs = np.asarray(gd.density, dtype=float)

    # Align via circular cross-correlation (grid origin conventions may differ),
    # then require strong agreement.
    a = mine - mine.mean()
    b = theirs - theirs.mean()
    xc = np.fft.ifftn(np.fft.fftn(a) * np.conj(np.fft.fftn(b))).real
    b_aligned = np.roll(b, np.unravel_index(np.argmax(xc), xc.shape), axis=(0, 1, 2))
    r = float(np.corrcoef(a.ravel(), b_aligned.ravel())[0, 1])
    assert r > 0.97
