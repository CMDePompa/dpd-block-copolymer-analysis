"""Architecture-resolved lamellar density-profile tests."""

from __future__ import annotations

import numpy as np
import pytest

from interface_analysis_v2.config import load_config
from interface_analysis_v2.density_profiles import (
    aggregate_density_profiles,
    compute_density_profile,
    plot_density_profile_summary,
)
from interface_analysis_v2.trajectory import FrameData


def _profile_frame(*, explicit_bead_indices: bool = False) -> FrameData:
    """Two shuffled A2-(AB)2 chains in an oblique (1,1,0) lamella."""
    # Atom-ID order within each molecule defines bead positions 1..4.
    atom_ids = np.array([3, 7, 1, 8, 4, 5, 2, 6], dtype=np.int64)
    mol_ids = np.array([1, 2, 1, 2, 1, 2, 1, 2], dtype=np.int32)
    bead = (atom_ids - 1) % 4 + 1
    # Pure positions 1,2 are A. The remaining positions contain one A and one B.
    types_by_bead = np.array([1, 1, 1, 2], dtype=np.int32)
    types = types_by_bead[bead - 1]

    # For mode (1,1,0), x/L + y/L fixes the reciprocal phase. A beads lie
    # near phase 0.55, B beads near 0.05; z and the orthogonal coordinate vary.
    phase = np.where(types == 1, 0.55, 0.05)
    xfrac = np.array([0.10, 0.22, 0.31, 0.37, 0.44, 0.49, 0.62, 0.71])
    yfrac = np.mod(phase - xfrac, 1.0)
    zfrac = np.linspace(0.1, 0.8, len(types))
    coords = 10.0 * np.column_stack([xfrac, yfrac, zfrac])
    return FrameData(
        timestep=1000,
        box_lo=np.zeros(3), box_hi=np.full(3, 10.0),
        types=types, coords=coords, mol_ids=mol_ids, atom_ids=atom_ids,
        bead_indices=bead.astype(np.int32) if explicit_bead_indices else None,
    )


def _cfg():
    cfg = load_config(None)
    cfg.update({
        "density_profile_enabled": True,
        "density_profile_chain_length": 4,
        "density_profile_pure_a_length": 2,
        "density_profile_bins": 20,
        "density_profile_alignment_sigma_bins": 0.0,
        "density_profile_interface_window_fraction": 0.05,
        "density_profile_allow_inferred_bead_order": True,
    })
    return cfg


def test_oblique_profile_decomposes_and_conserves_particles():
    result = compute_density_profile(_profile_frame(), (1, 1, 0), _cfg())
    assert result.diagnostics["topology_source"] == "atom_id_order_within_molecule"
    assert result.diagnostics["lattice_mode_canonical"] == [1, 1, 0]
    assert result.profiles["count_A_head"].sum() == 4
    assert result.profiles["count_A_tail"].sum() == 2
    assert result.profiles["count_B_tail"].sum() == 2
    assert result.profiles["count_total"].sum() == 8
    # Mean of a phase-binned number density recovers N / box volume.
    assert result.profiles["rho_total"].mean() == pytest.approx(8 / 1000)
    assert int(np.nanargmax(result.profiles["phi_A_total"])) == 10


def test_explicit_bead_index_avoids_order_assumption_warning():
    result = compute_density_profile(_profile_frame(explicit_bead_indices=True),
                                     (-1, -1, 0), _cfg())
    assert result.diagnostics["topology_source"] == "dump_bead_index_column"
    assert not result.warnings
    # Friedel-equivalent mode signs are canonicalized before frame averaging.
    assert result.diagnostics["lattice_mode_canonical"] == [1, 1, 0]


def test_non_a_bead_in_pure_block_is_rejected():
    frame = _profile_frame(explicit_bead_indices=True)
    frame.types[np.flatnonzero(frame.bead_indices == 1)[0]] = 2
    with pytest.raises(ValueError, match="pure-block positions"):
        compute_density_profile(frame, (1, 1, 0), _cfg())


def test_aggregate_and_plot(tmp_path):
    result = compute_density_profile(_profile_frame(), (1, 1, 0), _cfg())
    table, arrays = aggregate_density_profiles([result, result], [1000, 2000])
    assert len(table) == 20
    assert arrays["rho_A_head"].shape == (2, 20)
    assert np.all(table["n_frames"] == 2)
    plot_density_profile_summary("test", tmp_path, table)
    assert (tmp_path / "test_block_composition_profile.png").is_file()
    assert (tmp_path / "test_block_density_profile.png").is_file()
