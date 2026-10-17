"""Integration tests: full pipeline on synthetic and (when present) real data."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from interface_analysis_v2.batch import run_batch
from interface_analysis_v2.pipeline import _stack_profiles_on_common_grid
from interface_analysis_v2.config import load_config
from interface_analysis_v2.tests.synthetic import make_lamellar_frame, write_lammpstrj

REAL_TRAJ_DIR = Path(__file__).resolve().parents[2] / "traj files"
REAL_TRAJ = REAL_TRAJ_DIR / "N32_n3000_pA3_a12_32_5.prod.lammpstrj"

EXPECTED_FILES = [
    "interface_summary.json", "interface_summary.csv", "interface_per_frame.csv",
    "interface_per_surface.csv", "interface_parameters.json",
    "interface_resolution.json", "interface_warnings.json",
    "interface_profiles.npz", "interface_maps.npz", "topology_by_threshold.csv",
    "frame_read_report.json",
]


def test_profile_stacking_interpolates_physical_coordinates():
    g1 = np.arange(-1.0, 1.01, 0.5)
    g2 = np.arange(-1.0, 1.01, 0.4)
    common, stack = _stack_profiles_on_common_grid(
        [g1, g2], [1.0 + 2.0 * g1, 1.0 + 2.0 * g2]
    )
    i0 = int(np.argmin(np.abs(common)))
    assert common[i0] == pytest.approx(0.0)
    assert stack[:, i0] == pytest.approx([1.0, 1.0])


@pytest.fixture(scope="module")
def synthetic_batch(tmp_path_factory):
    """3-frame synthetic A/B trajectory analyzed end to end."""
    root = tmp_path_factory.mktemp("synth")
    frames = []
    for k in range(3):
        fr, truth = make_lamellar_frame(L=20.0, rho=3.0, m=(0, 0, 2), lam=0.8,
                                        fA_B_block=0.2, seed=10 + k,
                                        timestep=1000 * (k + 1))
        frames.append(fr)
    write_lammpstrj(root / "synth_pA2_a12_30_1.prod.lammpstrj", frames)
    cfg = load_config(None)
    cfg.update({
        "input_dir": str(root),
        "output_dir": str(root / "out"),
        "analysis_start_step": None,
        "sq_results_dir": None,
        "n_periods": 2,
        "plots": True,
    })
    combined = run_batch(cfg, echo=None)
    return root, combined, truth


def test_all_output_files_exist(synthetic_batch):
    root, combined, truth = synthetic_batch
    sim_out = root / "out" / "synth_pA2_a12_30_1"
    for name in EXPECTED_FILES:
        assert (sim_out / name).is_file(), f"missing {name}"
    assert (sim_out / "logs" / "interface_analysis_v2.log").is_file()
    assert combined.is_file()
    assert (root / "out" / "interface_comparison_headline.csv").is_file()
    plots = list((sim_out / "plots").glob("*.png"))
    assert len(plots) >= 10, f"only {len(plots)} plots produced"


def test_summary_values_recover_ground_truth(synthetic_batch):
    root, combined, truth = synthetic_batch
    df = pd.read_csv(combined)
    assert len(df) == 1
    r = df.iloc[0]
    assert r["status"] == "ok"
    assert r["n_frames_analyzed"] == 3
    # spacing must match the imposed L0 = 10
    assert r["spacing_mean_mean"] == pytest.approx(truth["L0"], rel=0.05)
    # irregular lamellae: high S_n, low bad-column fraction
    assert r["S_n_mean"] > 0.7
    assert r["bad_column_fraction_mean"] < 0.35
    # asymmetric penetration present by construction
    assert r["P_A_to_B_mean"] > r["P_B_to_A_mean"]
    # width in a physically sensible band (imposed 0.8 + smoothing broadening)
    assert 0.6 < r["width_lam_corrected_mean_mean"] < 1.6


def test_per_frame_and_per_surface_tables(synthetic_batch):
    root, _, _ = synthetic_batch
    sim_out = root / "out" / "synth_pA2_a12_30_1"
    pf = pd.read_csv(sim_out / "interface_per_frame.csv")
    assert len(pf) == 3
    assert {"timestep", "waviness_rms", "spacing_mean", "bad_column_fraction",
            "P_A_to_B", "S_n"} <= set(pf.columns)
    ps = pd.read_csv(sim_out / "interface_per_surface.csv")
    assert len(ps) >= 3 * 3  # >= 3 interfaces x 3 frames
    topo = pd.read_csv(sim_out / "topology_by_threshold.csv")
    assert {"phi_c", "phase", "beta0", "chi_total"} <= set(topo.columns)


def test_warnings_and_parameters_recorded(synthetic_batch):
    root, _, _ = synthetic_batch
    sim_out = root / "out" / "synth_pA2_a12_30_1"
    with (sim_out / "interface_parameters.json").open() as fh:
        params = json.load(fh)
    assert "config" in params and "config_hash" in params
    assert params["config"]["smoothing_sigma"] == 0.75
    with (sim_out / "interface_summary.json").open() as fh:
        summ = json.load(fh)
    assert "provenance" in summ
    assert "package_versions" in summ["provenance"]


def test_maps_and_profiles_npz(synthetic_batch):
    root, _, _ = synthetic_batch
    sim_out = root / "out" / "synth_pA2_a12_30_1"
    maps = np.load(sim_out / "interface_maps.npz")
    assert "heights" in maps and "column_class" in maps
    assert maps["heights"].ndim == 3
    profs = np.load(sim_out / "interface_profiles.npz")
    assert "aligned_profile_mean" in profs and "s_grid" in profs
    assert profs["aligned_profiles"].shape[0] == 3   # one per frame


@pytest.mark.skipif(not REAL_TRAJ.is_file(), reason="real trajectory not present")
def test_real_trajectory_one_frame(tmp_path):
    """Real pA3 file, one frame end to end (integration against real data)."""
    cfg = load_config(None)
    cfg.update({
        "input_dir": str(REAL_TRAJ_DIR),
        "output_dir": str(tmp_path / "out"),
        "sims": ["N32_n3000_pA3_a12_32_5"],
        "analysis_start_step": None,
        "sq_results_dir": None,
        "max_frames": 1,
        "plots": False,
    })
    combined = run_batch(cfg, echo=None)
    df = pd.read_csv(combined)
    r = df.iloc[0]
    assert r["status"] == "ok"
    assert r["n_frames_analyzed"] == 1
    assert 9.0 < r["spacing_mean_mean"] < 13.0      # around L0 ~ 11
    assert r["bad_column_fraction_mean"] < 0.5
