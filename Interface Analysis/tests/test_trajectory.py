"""Trajectory reading: correctness, reference equivalence, selection, failure reporting."""

from __future__ import annotations

import numpy as np
import pytest

from interface_analysis_v2.sq_bridge import load_reference_reader
from interface_analysis_v2.trajectory import (
    FrameData,
    FrameReadReport,
    TrajectoryIndex,
    get_index,
    iter_selected_frames,
    select_records,
)
from interface_analysis_v2.tests.synthetic import write_lammpstrj


def _random_frames(n_frames=3, n_atoms=200, L=12.0, seed=0):
    rng = np.random.default_rng(seed)
    frames = []
    for k in range(n_frames):
        coords = rng.uniform(0, L, size=(n_atoms, 3))
        types = rng.integers(1, 3, size=n_atoms).astype(np.int32)
        mols = (np.arange(n_atoms) // 10 + 1).astype(np.int32)
        frames.append(FrameData(
            timestep=1000 * (k + 1),
            box_lo=np.zeros(3), box_hi=np.full(3, L),
            types=types, coords=coords, mol_ids=mols,
        ))
    return frames


def _read_all(path, start_step=None, stride=1, max_frames=None,
              last_step_window=None):
    report = FrameReadReport(traj_path=str(path))
    index = TrajectoryIndex.build(path)
    records = select_records(index, start_step, stride, max_frames, report,
                             last_step_window=last_step_window)
    frames = list(iter_selected_frames(path, records, report, A_type_id=1, B_type_id=2))
    return frames, report, index


@pytest.mark.parametrize("coord_style", ["x", "xu", "xs"])
def test_roundtrip_and_folding(tmp_path, coord_style):
    frames_in = _random_frames()
    path = write_lammpstrj(tmp_path / "t.lammpstrj", frames_in,
                           coord_style=coord_style, unwrap_shift=2.0)
    frames_out, report, index = _read_all(path)

    assert index.n_frames == 3
    assert report.n_parsed_ok == 3 and not report.failed_frames
    for fin, fout in zip(frames_in, frames_out):
        assert fout.timestep == fin.timestep
        assert fout.n_atoms == fin.n_atoms
        np.testing.assert_array_equal(fout.types, fin.types)
        np.testing.assert_array_equal(fout.mol_ids, fin.mol_ids)
        np.testing.assert_array_equal(fout.atom_ids, np.arange(1, fin.n_atoms + 1))
        # coords must come back folded into the box regardless of style
        L = fin.L
        assert np.all(fout.coords >= fin.box_lo - 1e-9)
        assert np.all(fout.coords < fin.box_hi + 1e-9)
        # and equal the original wrapped positions (limited by text precision)
        expected = fin.box_lo + np.mod(fin.coords - fin.box_lo, L)
        wrapped_diff = np.abs(fout.coords - expected)
        wrapped_diff = np.minimum(wrapped_diff, L - wrapped_diff)  # positions near 0/L may fold either way
        assert np.max(wrapped_diff) < 1e-6


def test_equivalence_with_reference_reader(tmp_path):
    ref = load_reference_reader()
    if ref is None:
        pytest.skip("S(q) pipeline reference reader not found")
    frames_in = _random_frames(n_frames=4, n_atoms=150)
    path = write_lammpstrj(tmp_path / "t.lammpstrj", frames_in,
                           coord_style="xu", unwrap_shift=1.0)

    ref_frames = list(ref.iter_frames(path, stride=2, A_type_id=1, B_type_id=2,
                                      start_step=2000))
    mine, report, _ = _read_all(path, start_step=2000, stride=2)

    assert len(ref_frames) == len(mine) > 0
    for rf, mf in zip(ref_frames, mine):
        assert rf.timestep == mf.timestep
        np.testing.assert_array_equal(rf.types, mf.types)
        np.testing.assert_array_equal(rf.mol_ids, mf.mol_ids)
        np.testing.assert_allclose(rf.coords, mf.coords, atol=1e-12, rtol=0)
        np.testing.assert_allclose(rf.box_lo, mf.box_lo)
        np.testing.assert_allclose(rf.box_hi, mf.box_hi)


def test_explicit_bead_index_column_roundtrip(tmp_path):
    frames_in = _random_frames(n_frames=1, n_atoms=40)
    frames_in[0].bead_indices = np.tile(np.arange(1, 11), 4).astype(np.int32)
    path = write_lammpstrj(tmp_path / "with_beads.lammpstrj", frames_in)
    frames_out, report, _ = _read_all(path)
    assert report.n_parsed_ok == 1
    np.testing.assert_array_equal(frames_out[0].bead_indices,
                                  frames_in[0].bead_indices)


def test_selection_start_stride_maxframes(tmp_path):
    frames_in = _random_frames(n_frames=7, n_atoms=50)
    path = write_lammpstrj(tmp_path / "t.lammpstrj", frames_in)

    frames, report, _ = _read_all(path, start_step=3000, stride=2)
    assert [f.timestep for f in frames] == [3000, 5000, 7000]
    assert report.n_after_start_step == 5

    frames, report, _ = _read_all(path, start_step=None, stride=1, max_frames=2)
    assert [f.timestep for f in frames] == [6000, 7000]  # keeps the LAST frames
    assert any("max_frames" in n for n in report.notes)


def test_selection_last_step_window(tmp_path):
    frames_in = _random_frames(n_frames=8, n_atoms=50)
    path = write_lammpstrj(tmp_path / "t.lammpstrj", frames_in)

    frames, report, _ = _read_all(
        path, start_step=3000, stride=2, last_step_window=3000
    )
    assert [f.timestep for f in frames] == [5000, 7000]
    assert report.n_after_start_step == 6
    assert report.n_after_last_step_window == 4
    assert report.last_step_window_used == 3000
    assert any("last_step_window=3000" in n for n in report.notes)


def test_truncated_last_frame_reported_not_silent(tmp_path):
    frames_in = _random_frames(n_frames=3, n_atoms=100)
    path = write_lammpstrj(tmp_path / "t.lammpstrj", frames_in, truncate_last_atoms=30)

    logged = []
    report = FrameReadReport(traj_path=str(path))
    index = TrajectoryIndex.build(path)
    records = select_records(index, None, 1, None, report)
    frames = list(iter_selected_frames(path, records, report,
                                       A_type_id=1, B_type_id=2, log=logged.append))
    assert len(frames) == 2                     # good frames still delivered
    assert len(report.failed_frames) == 1       # bad frame recorded ...
    assert report.failed_frames[0]["timestep"] == 3000
    assert "truncated" in report.failed_frames[0]["reason"]
    assert any("ERROR" in m for m in logged)    # ... and loudly logged


def test_missing_type_id_raises(tmp_path):
    frames_in = _random_frames(n_frames=1)
    path = write_lammpstrj(tmp_path / "t.lammpstrj", frames_in)
    report = FrameReadReport()
    index = TrajectoryIndex.build(path)
    records = select_records(index, None, 1, None, report)
    with pytest.raises(RuntimeError, match="type IDs not found"):
        list(iter_selected_frames(path, records, report, A_type_id=1, B_type_id=9))


def test_later_frame_missing_type_is_recorded_and_skipped(tmp_path):
    frames_in = _random_frames(n_frames=3, n_atoms=80)
    frames_in[0].types[:2] = [1, 2]
    frames_in[1].types[:] = 1
    path = write_lammpstrj(tmp_path / "t.lammpstrj", frames_in)
    report = FrameReadReport()
    index = TrajectoryIndex.build(path)
    records = select_records(index, None, 1, None, report)
    frames = list(iter_selected_frames(path, records, report, 1, 2))
    assert [f.timestep for f in frames] == [1000, 3000]
    assert report.failed_frames[0]["timestep"] == 2000
    assert "missing" in report.failed_frames[0]["reason"]


def test_variable_atom_count_requires_opt_in(tmp_path):
    frames_in = _random_frames(n_frames=2, n_atoms=80)
    frames_in[0].types[:2] = [1, 2]
    frames_in[1].types[:2] = [1, 2]
    frames_in[1].coords = frames_in[1].coords[:-5]
    frames_in[1].types = frames_in[1].types[:-5]
    frames_in[1].mol_ids = frames_in[1].mol_ids[:-5]
    path = write_lammpstrj(tmp_path / "t.lammpstrj", frames_in)
    report = FrameReadReport()
    index = TrajectoryIndex.build(path)
    records = select_records(index, None, 1, None, report)
    frames = list(iter_selected_frames(path, records, report, 1, 2,
                                       allow_variable_natoms=False))
    assert [f.timestep for f in frames] == [1000]
    assert "N_atoms changed" in report.failed_frames[0]["reason"]


def test_index_cache_roundtrip(tmp_path):
    frames_in = _random_frames(n_frames=3, n_atoms=40)
    path = write_lammpstrj(tmp_path / "t.lammpstrj", frames_in)
    idx1 = get_index(path, cache_dir=tmp_path / "cache", use_cache=True)
    idx2 = get_index(path, cache_dir=tmp_path / "cache", use_cache=True)
    assert [r.offset for r in idx1.records] == [r.offset for r in idx2.records]
    assert [r.timestep for r in idx1.records] == [1000, 2000, 3000]
    assert (tmp_path / "cache" / "t.lammpstrj.index.json").is_file()
