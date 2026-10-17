"""LAMMPS dump reading for multi-gigabyte trajectories: index + vectorized parse.

This module mirrors the SEMANTICS of the S(q) pipeline's
``pipeline/trajectory_reader.py`` (the reference implementation):

* identical coordinate-column priority: (x,y,z) then (xu,yu,zu) then (xs,ys,zs);
* scaled coordinates are unscaled with the frame's box; unwrapped coordinates
  are accepted as absolute;
* coordinates are folded into the primary periodic cell [lo, lo+L);
* atom file order is preserved (no id sort);
* A/B type presence is validated on the first parsed frame;
* box-dimension changes between frames produce a warning, not an error.

Equivalence against the reference reader is enforced by a unit test
(``tests/test_trajectory.py``) whenever the S(q) pipeline is present.

Differences (and why): the test trajectories are 1.7-14 GB, so this module
adds (1) a cached byte-offset frame index so selected frames are reached by
``seek`` instead of line-by-line streaming, and (2) ``numpy.loadtxt``-based
block parsing (~20x faster than per-line float()). Failed or malformed frames
are recorded per frame in a :class:`FrameReadReport` and logged — never
silently dropped.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

from .io_utils import LogFn

_MARKER = b"ITEM: TIMESTEP"


class TrajectoryFormatError(ValueError):
    """Raised when a frame block violates the expected LAMMPS dump format."""


class IncompleteFrameError(TrajectoryFormatError):
    """Raised when a frame block ends before all atom lines are present."""


@dataclass
class FrameData:
    """One snapshot from a LAMMPS trajectory (same fields as the S(q) reader)."""

    timestep: int
    box_lo: np.ndarray                     # (3,)
    box_hi: np.ndarray                     # (3,)
    types: np.ndarray                      # (N,) int32
    coords: np.ndarray                     # (N, 3) folded into [lo, hi)
    mol_ids: Optional[np.ndarray] = None   # (N,) int32 if present
    atom_ids: Optional[np.ndarray] = None  # (N,) int64 if an id column is present
    bead_indices: Optional[np.ndarray] = None  # (N,) int32 if explicitly dumped

    @property
    def L(self) -> np.ndarray:
        return self.box_hi - self.box_lo

    @property
    def n_atoms(self) -> int:
        return len(self.types)

    def positions_of_type(self, type_id: int) -> np.ndarray:
        return self.coords[self.types == type_id]


@dataclass
class FrameRecord:
    """Index entry: where a frame starts and its cheap header metadata."""

    offset: int
    timestep: int
    natoms: int


@dataclass
class FrameReadReport:
    """Full accounting of what was read, selected, parsed, and skipped."""

    traj_path: str = ""
    n_frames_in_file: int = 0
    n_after_start_step: int = 0
    n_after_last_step_window: int = 0
    n_selected: int = 0
    n_parsed_ok: int = 0
    failed_frames: List[Dict[str, Any]] = field(default_factory=list)
    selected_timesteps: List[int] = field(default_factory=list)
    start_step_used: Optional[int] = None
    last_step_window_used: Optional[int] = None
    stride_used: int = 1
    max_frames_used: Optional[int] = None
    notes: List[str] = field(default_factory=list)

    def record_failure(self, timestep: Optional[int], offset: int, reason: str) -> None:
        self.failed_frames.append(
            {"timestep": timestep, "byte_offset": offset, "reason": reason}
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "traj_path": self.traj_path,
            "n_frames_in_file": self.n_frames_in_file,
            "n_after_start_step": self.n_after_start_step,
            "n_after_last_step_window": self.n_after_last_step_window,
            "n_selected": self.n_selected,
            "n_parsed_ok": self.n_parsed_ok,
            "n_failed": len(self.failed_frames),
            "failed_frames": self.failed_frames,
            "selected_timesteps": self.selected_timesteps,
            "start_step_used": self.start_step_used,
            "last_step_window_used": self.last_step_window_used,
            "stride_used": self.stride_used,
            "max_frames_used": self.max_frames_used,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Index construction and caching
# ---------------------------------------------------------------------------
@dataclass
class TrajectoryIndex:
    path: str
    file_size: int
    file_mtime_ns: int
    records: List[FrameRecord]

    @property
    def n_frames(self) -> int:
        return len(self.records)

    @classmethod
    def build(cls, path: str | Path, log: Optional[LogFn] = None,
              chunk_size: int = 64 * 1024 * 1024) -> "TrajectoryIndex":
        """Scan the file once (buffered binary) for frame start offsets."""
        path = Path(path)
        stat = path.stat()
        offsets: List[int] = []
        last = -1
        with path.open("rb") as fh:
            base = 0            # absolute offset of buf[0]
            carry = b""
            while True:
                chunk = fh.read(chunk_size)
                if not chunk:
                    break
                buf = carry + chunk
                start = 0
                while True:
                    i = buf.find(_MARKER, start)
                    if i == -1:
                        break
                    abs_off = base + i
                    at_line_start = (abs_off == 0) or (i >= 1 and buf[i - 1:i] == b"\n")
                    if at_line_start and abs_off > last:
                        offsets.append(abs_off)
                        last = abs_off
                    start = i + 1
                keep = len(_MARKER)  # enough to catch a marker straddling the boundary
                carry = buf[-keep:] if len(buf) >= keep else buf
                base += len(buf) - len(carry)

        # Cheap header read (timestep, natoms) per frame via seek.
        records: List[FrameRecord] = []
        with path.open("rb") as fh:
            for off in offsets:
                fh.seek(off)
                try:
                    hdr = fh.readline()          # ITEM: TIMESTEP
                    ts = int(fh.readline())
                    line = fh.readline()
                    if not line.startswith(b"ITEM: NUMBER OF ATOMS"):
                        raise TrajectoryFormatError(
                            f"offset {off}: expected 'ITEM: NUMBER OF ATOMS', got {line[:50]!r}"
                        )
                    natoms = int(fh.readline())
                    if not hdr.startswith(_MARKER):
                        raise TrajectoryFormatError(f"offset {off}: bad frame marker")
                except (ValueError, TrajectoryFormatError) as exc:
                    # An unreadable header at a detected marker is a real
                    # format problem; report it as a frame with natoms = -1 so
                    # selection/parse surfaces it rather than hiding it.
                    if log:
                        log(f"WARN: unreadable frame header at byte {off}: {exc}")
                    records.append(FrameRecord(offset=off, timestep=-1, natoms=-1))
                    continue
                records.append(FrameRecord(offset=off, timestep=ts, natoms=natoms))

        if log:
            log(f"Indexed {len(records)} frames in {path.name} "
                f"({stat.st_size / 1e9:.2f} GB)")
        return cls(path=str(path), file_size=stat.st_size,
                   file_mtime_ns=stat.st_mtime_ns, records=records)

    # -- caching ------------------------------------------------------------
    def save(self, cache_path: str | Path) -> None:
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        doc = {
            "path": self.path,
            "file_size": self.file_size,
            "file_mtime_ns": self.file_mtime_ns,
            "offsets": [r.offset for r in self.records],
            "timesteps": [r.timestep for r in self.records],
            "natoms": [r.natoms for r in self.records],
        }
        with cache_path.open("w") as fh:
            json.dump(doc, fh)

    @classmethod
    def load(cls, cache_path: str | Path, traj_path: str | Path) -> Optional["TrajectoryIndex"]:
        """Load a cached index; returns None if stale or unreadable."""
        cache_path = Path(cache_path)
        traj_path = Path(traj_path)
        if not cache_path.is_file():
            return None
        try:
            with cache_path.open("r") as fh:
                doc = json.load(fh)
            stat = traj_path.stat()
            if doc["file_size"] != stat.st_size or doc["file_mtime_ns"] != stat.st_mtime_ns:
                return None
            records = [
                FrameRecord(offset=o, timestep=t, natoms=n)
                for o, t, n in zip(doc["offsets"], doc["timesteps"], doc["natoms"])
            ]
            return cls(path=str(traj_path), file_size=doc["file_size"],
                       file_mtime_ns=doc["file_mtime_ns"], records=records)
        except (OSError, KeyError, ValueError, TypeError):
            return None


def get_index(traj_path: str | Path, cache_dir: Optional[str | Path],
              use_cache: bool = True, log: Optional[LogFn] = None) -> TrajectoryIndex:
    """Return a (possibly cached) TrajectoryIndex for the file."""
    traj_path = Path(traj_path)
    cache_path = None
    if cache_dir is not None:
        cache_path = Path(cache_dir) / (traj_path.name + ".index.json")
    if use_cache and cache_path is not None:
        cached = TrajectoryIndex.load(cache_path, traj_path)
        if cached is not None:
            if log:
                log(f"Loaded cached frame index ({cached.n_frames} frames).")
            return cached
    index = TrajectoryIndex.build(traj_path, log=log)
    if use_cache and cache_path is not None:
        index.save(cache_path)
    return index


# ---------------------------------------------------------------------------
# Frame parsing
# ---------------------------------------------------------------------------
def _resolve_columns(
    cols: List[str], path_label: str
) -> Tuple[int, Optional[int], Optional[int], Optional[int], List[int], str]:
    """Column indices for type/topology/xyz with the reference reader's priority."""
    if "type" not in cols:
        raise TrajectoryFormatError(f"{path_label}: ATOMS header must contain 'type'. Found: {cols}")
    i_type = cols.index("type")
    i_mol = cols.index("mol") if "mol" in cols else None
    i_id = cols.index("id") if "id" in cols else None
    # These are deliberately specific names. A generic ``index`` column is
    # too ambiguous to use as molecular topology without user verification.
    bead_names = ("bead_index", "bead", "monomer_index", "monomer", "chain_index")
    i_bead = next((cols.index(name) for name in bead_names if name in cols), None)
    if all(c in cols for c in ("x", "y", "z")):
        ixyz = [cols.index("x"), cols.index("y"), cols.index("z")]
        mode = "abs"
    elif all(c in cols for c in ("xu", "yu", "zu")):
        ixyz = [cols.index("xu"), cols.index("yu"), cols.index("zu")]
        mode = "abs"
    elif all(c in cols for c in ("xs", "ys", "zs")):
        ixyz = [cols.index("xs"), cols.index("ys"), cols.index("zs")]
        mode = "scaled"
    else:
        raise TrajectoryFormatError(
            f"{path_label}: ATOMS header must contain (x,y,z), (xu,yu,zu) or (xs,ys,zs). Found: {cols}"
        )
    return i_type, i_mol, i_id, i_bead, ixyz, mode


def parse_frame_at(fh: io.BufferedReader, offset: int, path_label: str = "",
                   fold_coords: bool = True) -> FrameData:
    """Parse one full frame from a binary handle positioned via seek(offset).

    Raises :class:`TrajectoryFormatError` / :class:`IncompleteFrameError` on
    malformed or truncated frames — the caller decides skip-vs-abort policy.
    """
    fh.seek(offset)

    def _line() -> bytes:
        ln = fh.readline()
        if not ln:
            raise IncompleteFrameError(f"{path_label}: unexpected end of file in frame at byte {offset}")
        return ln

    hdr = _line()
    if not hdr.startswith(_MARKER):
        raise TrajectoryFormatError(f"{path_label}: no TIMESTEP marker at byte {offset}")
    try:
        timestep = int(_line())
    except ValueError as exc:
        raise TrajectoryFormatError(f"{path_label}: unparseable timestep at byte {offset}: {exc}") from exc

    hdr = _line()
    if not hdr.startswith(b"ITEM: NUMBER OF ATOMS"):
        raise TrajectoryFormatError(f"{path_label}: expected 'ITEM: NUMBER OF ATOMS', got {hdr[:60]!r}")
    natoms = int(_line())
    if natoms <= 0:
        raise TrajectoryFormatError(f"{path_label}: frame at step {timestep} has natoms={natoms}")

    hdr = _line()
    if not hdr.startswith(b"ITEM: BOX BOUNDS"):
        raise TrajectoryFormatError(f"{path_label}: expected 'ITEM: BOX BOUNDS', got {hdr[:60]!r}")
    box_lo = np.zeros(3)
    box_hi = np.zeros(3)
    for i in range(3):
        parts = _line().split()
        if len(parts) >= 3:
            raise TrajectoryFormatError(
                f"{path_label}: triclinic box detected at step {timestep} "
                "(3 values per bounds line). Only orthorhombic boxes are supported."
            )
        if len(parts) != 2:
            raise TrajectoryFormatError(
                f"{path_label}: malformed BOX BOUNDS line at step {timestep}: {parts!r}"
            )
        box_lo[i] = float(parts[0])
        box_hi[i] = float(parts[1])
    L = box_hi - box_lo
    if np.any(L <= 0):
        raise TrajectoryFormatError(f"{path_label}: non-positive box lengths {L} at step {timestep}")

    hdr = _line()
    if not hdr.startswith(b"ITEM: ATOMS"):
        raise TrajectoryFormatError(f"{path_label}: expected 'ITEM: ATOMS ...', got {hdr[:60]!r}")
    cols = hdr.decode("ascii", errors="replace").strip().split()[2:]
    i_type, i_mol, i_id, i_bead, ixyz, mode = _resolve_columns(cols, path_label)

    lines: List[bytes] = []
    for _ in range(natoms):
        ln = fh.readline()
        if not ln or ln.startswith(b"ITEM:"):
            raise IncompleteFrameError(
                f"{path_label}: frame at step {timestep} truncated: "
                f"{len(lines)}/{natoms} atom lines present"
            )
        lines.append(ln)

    try:
        arr = np.loadtxt(io.BytesIO(b"".join(lines)), ndmin=2, dtype=np.float64)
    except ValueError as exc:
        raise TrajectoryFormatError(
            f"{path_label}: unparseable atom block at step {timestep}: {exc}"
        ) from exc
    if arr.shape != (natoms, len(cols)):
        raise TrajectoryFormatError(
            f"{path_label}: atom block at step {timestep} has shape {arr.shape}, "
            f"expected ({natoms}, {len(cols)})"
        )

    types = arr[:, i_type].astype(np.int32)
    mol_ids = arr[:, i_mol].astype(np.int32) if i_mol is not None else None
    atom_ids = arr[:, i_id].astype(np.int64) if i_id is not None else None
    bead_indices = arr[:, i_bead].astype(np.int32) if i_bead is not None else None
    coords = arr[:, ixyz].copy()
    if mode == "scaled":
        coords = box_lo[None, :] + coords * L[None, :]
    if fold_coords:
        coords = box_lo[None, :] + np.mod(coords - box_lo[None, :], L[None, :])

    return FrameData(timestep=timestep, box_lo=box_lo, box_hi=box_hi,
                     types=types, coords=coords, mol_ids=mol_ids,
                     atom_ids=atom_ids, bead_indices=bead_indices)


# ---------------------------------------------------------------------------
# Selection and iteration
# ---------------------------------------------------------------------------
def select_records(
    index: TrajectoryIndex,
    start_step: Optional[int],
    stride: int,
    max_frames: Optional[int],
    report: FrameReadReport,
    last_step_window: Optional[int] = None,
) -> List[FrameRecord]:
    """Apply start step, trailing timestep window, stride, and frame cap.

    ``last_step_window`` is measured in simulation timesteps, not frames. For
    example, 100000 keeps frames at or after ``final_timestep - 100000``.
    Filters are composable: start step, trailing window, stride, then cap.
    """
    records = index.records
    report.n_frames_in_file = len(records)
    after = [r for r in records if start_step is None or r.timestep >= start_step]
    report.n_after_start_step = len(after)
    if last_step_window is not None and after:
        final_step = max(r.timestep for r in records)
        cutoff = final_step - int(last_step_window)
        before_window = len(after)
        after = [r for r in after if r.timestep >= cutoff]
        report.notes.append(
            f"last_step_window={last_step_window} kept {len(after)} of "
            f"{before_window} frames at timestep >= {cutoff} "
            f"(final timestep {final_step})."
        )
    report.n_after_last_step_window = len(after)
    sel = after[::max(1, int(stride))]
    if max_frames is not None and len(sel) > int(max_frames):
        sel = sel[-int(max_frames):]
        report.notes.append(
            f"max_frames={max_frames} kept the LAST {max_frames} of {len(after[::stride])} "
            "selected frames (bias toward late, most-equilibrated configurations)."
        )
    report.n_selected = len(sel)
    report.start_step_used = start_step
    report.last_step_window_used = last_step_window
    report.stride_used = int(stride)
    report.max_frames_used = max_frames
    return sel


def iter_selected_frames(
    traj_path: str | Path,
    records: List[FrameRecord],
    report: FrameReadReport,
    A_type_id: int,
    B_type_id: int,
    log: Optional[LogFn] = None,
    allow_variable_natoms: bool = False,
) -> Iterator[FrameData]:
    """Yield parsed FrameData for each selected record.

    Malformed frames are logged as ERROR, recorded in the report, and skipped
    (loudly). Raises RuntimeError if the first frame fails A/B type validation;
    later frames missing either configured type are recorded and skipped.
    Atom-count changes are likewise skipped unless explicitly allowed.
    """
    traj_path = Path(traj_path)
    first_natoms: Optional[int] = None
    first_L: Optional[np.ndarray] = None
    L_warned = False
    with traj_path.open("rb") as fh:
        for rec in records:
            try:
                frame = parse_frame_at(fh, rec.offset, path_label=traj_path.name)
            except TrajectoryFormatError as exc:
                msg = f"ERROR: frame at step {rec.timestep} (byte {rec.offset}) failed to parse: {exc}"
                if log:
                    log(msg)
                report.record_failure(rec.timestep, rec.offset, str(exc))
                continue

            if first_natoms is None:
                first_natoms = frame.n_atoms
                first_L = frame.L.copy()
                present = set(np.unique(frame.types).tolist())
                missing = [f"{name}_type_id={tid}"
                           for name, tid in (("A", A_type_id), ("B", B_type_id))
                           if tid not in present]
                if missing:
                    raise RuntimeError(
                        f"{traj_path.name}: type IDs not found in first frame: "
                        f"{', '.join(missing)}. Present types: {sorted(present)}"
                    )
            else:
                present = set(np.unique(frame.types).tolist())
                missing = [f"{name}_type_id={tid}"
                           for name, tid in (("A", A_type_id), ("B", B_type_id))
                           if tid not in present]
                if missing:
                    msg = (f"configured bead type(s) missing from frame: {', '.join(missing)}; "
                           f"present types: {sorted(present)}")
                    if log:
                        log(f"ERROR: step {frame.timestep}: {msg}")
                    report.record_failure(frame.timestep, rec.offset, msg)
                    continue
                if frame.n_atoms != first_natoms and not allow_variable_natoms:
                    msg = (f"N_atoms changed from {first_natoms} to {frame.n_atoms}; "
                           "set allow_variable_natoms=true to analyze such frames")
                    if log:
                        log(f"ERROR: step {frame.timestep}: {msg}")
                    report.record_failure(frame.timestep, rec.offset, msg)
                    continue
                if first_L is not None and not np.allclose(frame.L, first_L, rtol=1e-4) and not L_warned:
                    if log:
                        log(f"WARN: box dimensions changed at step {frame.timestep} "
                            f"(initial L={first_L}, now L={frame.L}). "
                            "Analysis assumes a fixed periodic box per frame; "
                            "fields are rebuilt per frame so this is handled, but "
                            "cross-frame comparisons mix box sizes.")
                    L_warned = True

            report.n_parsed_ok += 1
            report.selected_timesteps.append(frame.timestep)
            yield frame
