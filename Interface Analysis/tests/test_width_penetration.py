"""Width fitting and penetration metrics against analytic ground truth."""

from __future__ import annotations

import numpy as np
import pytest

from interface_analysis_v2.composition import CompositionField, composition_field
from interface_analysis_v2.config import load_config
from interface_analysis_v2.density_field import build_density_fields
from interface_analysis_v2.interface_detection import detect_interfaces
from interface_analysis_v2.orientation import resolve_orientation
from interface_analysis_v2.penetration import compute_penetration
from interface_analysis_v2.resample import resample_slab
from interface_analysis_v2.tests.synthetic import make_lamellar_frame, make_phi_grid
from interface_analysis_v2.width import W1090_PER_LAM, compute_width


def _comp(phi, grid):
    return CompositionField(phi=phi, invalid_mask=np.zeros(phi.shape, bool), grid=grid,
                            stats={}, warnings=[])


def _detect(phi, grid, truth, cfg):
    comp = _comp(phi, grid)
    ori = resolve_orientation(comp, cfg, sq_q_star=truth["q_mag"])
    slab = resample_slab(comp, ori.n0, ori.L0, cfg)
    iset = detect_interfaces(slab, cfg)
    return comp, ori, slab, iset


@pytest.mark.parametrize("lam_true", [0.6, 1.0, 1.5])
def test_width_recovers_known_lambda(lam_true):
    """Analytic tanh lamellae: fitted lam must equal the imposed tanh length.

    Near the crossing phi = amp*tanh(sin(qs)/(q*lam)) ~ amp*tanh((s-s0)/lam),
    so the ground truth is exact at the crossing; tolerance covers the sin
    curvature over the fit window and interpolation.
    """
    phi, grid, truth = make_phi_grid(shape=(48, 48, 48), L=(24.0, 24.0, 24.0),
                                     m=(0, 0, 2), lam=lam_true, amp=0.9)
    cfg = load_config(None)
    comp, ori, slab, iset = _detect(phi, grid, truth, cfg)
    res = compute_width(slab, iset, cfg, smoothing_sigma=0.0)
    assert res.failed_fraction < 0.2
    assert res.mean_lam == pytest.approx(lam_true, rel=0.12)
    assert res.mean_w1090 == pytest.approx(W1090_PER_LAM * lam_true, rel=0.15)
    # flat lamellae: obliquity correction is a no-op
    assert res.mean_lam_corrected == pytest.approx(res.mean_lam, rel=0.02)
    # aligned mean profile gives the same answer as the local-fit mean
    assert res.aligned_fit["lam"] == pytest.approx(lam_true, rel=0.12)


def test_width_asymmetric_offset_still_fits():
    """Offset (A dissolved in B block) must be absorbed by the fit's 'a' term."""
    phi, grid, truth = make_phi_grid(shape=(48, 48, 48), L=(24.0, 24.0, 24.0),
                                     m=(0, 0, 2), lam=0.8, amp=0.7, offset=0.25)
    cfg = load_config(None)
    comp, ori, slab, iset = _detect(phi, grid, truth, cfg)
    res = compute_width(slab, iset, cfg, smoothing_sigma=0.0)
    assert res.failed_fraction < 0.2
    assert res.mean_lam == pytest.approx(0.8, rel=0.15)
    assert res.aligned_fit["a"] == pytest.approx(0.25, abs=0.08)


def test_waviness_broadens_naive_but_not_aligned_width():
    """THE key distinction: with strong waviness the naive laterally averaged
    profile must be broader than the locally aligned width."""
    phi, grid, truth = make_phi_grid(
        shape=(56, 56, 56), L=(28.0, 28.0, 28.0), m=(0, 0, 2),
        lam=0.7, amp=0.9, waviness_amp=1.5, waviness_kint=(1, 0, 0),
    )
    cfg = load_config(None)
    comp, ori, slab, iset = _detect(phi, grid, truth, cfg)
    res = compute_width(slab, iset, cfg, smoothing_sigma=0.0)
    # aligned width stays close to the intrinsic value
    assert res.mean_lam == pytest.approx(0.7, rel=0.25)
    # naive profile is significantly broadened by the +-1.5 undulation
    assert res.waviness_broadening_ratio > 1.5


def test_penetration_metrics_particle_synthetic():
    """Synthetic mixed-domain case (20% A in the B-rich region): wrong-side A fraction
    reflects the dissolved A; the pure-side wrong-side B stays near zero;
    P_B->A < P_A->B mirrors the asymmetry; overlap is between 0 and 1."""
    frame, truth = make_lamellar_frame(L=24.0, rho=3.0, m=(0, 0, 2),
                                       lam=0.8, fA_B_block=0.2, seed=3)
    cfg = load_config(None)
    fields = build_density_fields(frame, 1, 2, cfg["target_voxel_size"], cfg["smoothing_sigma"])
    comp = composition_field(fields, cfg["density_cutoff_fraction"])
    ori = resolve_orientation(comp, cfg, sq_q_star=truth["q_mag"])
    slab = resample_slab(comp, ori.n0, ori.L0, cfg)
    iset = detect_interfaces(slab, cfg)
    pen = compute_penetration(fields, comp, slab, iset, frame, cfg)

    # In the B-rich phase phi_plateau = 2*0.2-1 = -0.6, so A beads there sit at
    # phi ~ -0.6: with threshold 0.25 they count as wrong-side.
    # A beads: half the beads are in the A block (phi>0), the other "half" of A
    # beads are the dissolved ones. Expected wrong-side_A(0.25) is roughly the
    # dissolved fraction of all A beads that sits at phi < -0.25.
    ws = pen.wrong_side_A["0.25"]
    assert 0.05 < ws < 0.35
    # essentially no B beads DEEP inside A (P(A)=1 there -> no B present).
    # At low thresholds the tanh tail legitimately contains interfacial B, so
    # the deep-domain statement is tested at the strongest threshold.
    assert pen.wrong_side_B["0.75"] < 0.02
    # asymmetry: dissolved A exceeds interfacial B at the same threshold
    # (with an offset composition the |phi|<t band is NOT symmetric about the
    # crossing, so only the direction of the inequality is asserted)
    assert pen.wrong_side_A["0.25"] > pen.wrong_side_B["0.25"]
    # Directional penetration. NOTE: with s_half = L0/2 the window reaches the
    # neighbouring interface, giving a clean-interface baseline P ~ 0.13 here
    # (verified with the pure synthetic below); the dissolved A adds ~0.11 on
    # top of it for P_A->B only.
    assert pen.P_B_to_A < 0.20
    assert pen.P_A_to_B > pen.P_B_to_A + 0.05
    assert 0.0 < pen.overlap_O < 1.0
    # signed-distance profile: rho_A rises toward s>0 (A side), rho_B falls
    mid = len(pen.s_bins) // 2
    assert np.nanmean(pen.rhoA_of_s[mid + 2:]) > np.nanmean(pen.rhoA_of_s[:mid - 2])
    assert np.nanmean(pen.rhoB_of_s[mid + 2:]) < np.nanmean(pen.rhoB_of_s[:mid - 2])
    # midplane maps exist and are in [0, 1]
    m1 = pen.map_fB_in_A_core
    assert np.isfinite(m1).any()
    assert np.nanmin(m1) >= 0.0 and np.nanmax(m1) <= 1.0
    # B core contains dissolved A (~30% of local density: 0.2 A vs 0.8 B block
    # composition -> f_A ~ 0.2/(0.2+0.8) with equal densities)
    fA_B = float(np.nanmean(pen.map_fA_in_B_core))
    assert fA_B == pytest.approx(0.2, abs=0.08)
    fB_A = float(np.nanmean(pen.map_fB_in_A_core))
    assert fB_A < 0.08


def test_penetration_pure_diblock_symmetric():
    """Pure diblock (no dissolved A): both penetrations small and comparable."""
    frame, truth = make_lamellar_frame(L=24.0, rho=3.0, m=(0, 0, 2),
                                       lam=0.6, fA_B_block=0.0, seed=5)
    cfg = load_config(None)
    fields = build_density_fields(frame, 1, 2, cfg["target_voxel_size"], cfg["smoothing_sigma"])
    comp = composition_field(fields, cfg["density_cutoff_fraction"])
    ori = resolve_orientation(comp, cfg, sq_q_star=truth["q_mag"])
    slab = resample_slab(comp, ori.n0, ori.L0, cfg)
    iset = detect_interfaces(slab, cfg)
    pen = compute_penetration(fields, comp, slab, iset, frame, cfg)
    assert pen.P_B_to_A < 0.20
    assert pen.P_A_to_B < 0.20
    assert abs(pen.P_B_to_A - pen.P_A_to_B) < 0.05
    # midplane cores are clean in both directions for the pure diblock
    assert float(np.nanmean(pen.map_fA_in_B_core)) < 0.05
    assert float(np.nanmean(pen.map_fB_in_A_core)) < 0.05
