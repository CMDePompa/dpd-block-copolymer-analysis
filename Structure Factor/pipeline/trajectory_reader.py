"""LAMMPS lammpstrj trajectory reader for DPD simulations.

Supports absolute (x, y, z), unwrapped (xu, yu, zu), and scaled (xs, ys, zs)
coordinate columns. Molecule IDs are read if present. Periodic boxes are assumed
throughout; warns if box dimensions change between frames.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import numpy as np


@dataclass
class FrameData:
    """One snapshot from a LAMMPS trajectory."""

    timestep: int
    box_lo: np.ndarray       # shape (3,)  lower bound per dimension
    box_hi: np.ndarray       # shape (3,)  upper bound per dimension
    types: np.ndarray        # shape (N,)  integer bead types
    coords: np.ndarray       # shape (N, 3)  positions folded into [lo, hi)
    mol_ids: Optional[np.ndarray] = None   # shape (N,) if present in file

    @property
    def L(self) -> np.ndarray:
        return self.box_hi - self.box_lo

    @property
    def n_atoms(self) -> int:
        return len(self.types)

    def positions_of_type(self, type_id: int) -> np.ndarray:
        """Return (M, 3) array of coordinates for beads with the given type ID."""
        mask = self.types == type_id
        return self.coords[mask]


def _fold_into_box(coords: np.ndarray, lo: np.ndarray, L: np.ndarray) -> np.ndarray:
    """Wrap coordinates into the primary periodic cell [lo, lo+L)."""
    return lo + np.mod(coords - lo, L)


def iter_frames(
    path: str | Path,
    stride: int = 1,
    A_type_id: int = 1,
    B_type_id: int = 2,
    allow_variable_natoms: bool = False,
    fold_coords: bool = True,
    start_step: Optional[int] = None,
) -> Iterator[FrameData]:
    """Yield :class:`FrameData` objects from a LAMMPS dump file.

    Parameters
    ----------
    path:
        Path to a .lammpstrj (or any LAMMPS dump) file.
    stride:
        Yield every *stride*-th frame (stride=1 yields all frames).
    A_type_id, B_type_id:
        Expected bead type IDs; raises ValueError if absent in the first frame.
    allow_variable_natoms:
        If False (default) a warning is emitted when N_atoms changes mid-trajectory.
    fold_coords:
        Wrap coordinates into the primary cell. Default True; set False only for
        visualisation of unwrapped chains.
    start_step:
        Skip all frames with timestep < start_step. The skip happens at the
        text level — the atom block of a skipped frame is consumed without
        numeric parsing, so skipping is cheap even for very large frames.
        Frames before start_step do not count toward the stride phase; the
        first kept frame is always yielded and stride counts from there.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Trajectory not found: {path}")

    frame_index = 0
    first_natoms: Optional[int] = None
    first_L: Optional[np.ndarray] = None
    L_warned = False

    with path.open("r") as fh:
        while True:
            # --- locate TIMESTEP header ---
            line = fh.readline()
            if not line:
                break
            if not line.startswith("ITEM: TIMESTEP"):
                continue

            timestep = int(fh.readline().strip())

            # NUMBER OF ATOMS
            hdr = fh.readline()
            if not hdr.startswith("ITEM: NUMBER OF ATOMS"):
                raise ValueError(f"{path}: expected 'ITEM: NUMBER OF ATOMS', got: {hdr.strip()!r}")
            natoms = int(fh.readline().strip())

            # BOX BOUNDS
            hdr = fh.readline()
            if not hdr.startswith("ITEM: BOX BOUNDS"):
                raise ValueError(f"{path}: expected 'ITEM: BOX BOUNDS', got: {hdr.strip()!r}")
            box_lo = np.zeros(3)
            box_hi = np.zeros(3)
            for i in range(3):
                parts = fh.readline().split()
                box_lo[i] = float(parts[0])
                box_hi[i] = float(parts[1])
            L = box_hi - box_lo

            # ATOMS header — parse column names
            hdr = fh.readline()
            if not hdr.startswith("ITEM: ATOMS"):
                raise ValueError(f"{path}: expected 'ITEM: ATOMS ...', got: {hdr.strip()!r}")
            cols = hdr.strip().split()[2:]

            # --- cheap skip BEFORE any numeric parsing of the atom block ---
            # start_step cut: frames before it are consumed line-by-line
            # without parsing and do not count toward the stride phase.
            if start_step is not None and timestep < start_step:
                for _ in range(natoms):
                    fh.readline()
                continue

            # Stride filter: also skipped without parsing the atom block.
            if frame_index % stride != 0:
                frame_index += 1
                for _ in range(natoms):
                    fh.readline()
                continue

            if "type" not in cols:
                raise ValueError(f"{path}: ATOMS header must contain 'type'. Found: {cols}")

            i_type = cols.index("type")
            i_mol = cols.index("mol") if "mol" in cols else None

            if all(c in cols for c in ("x", "y", "z")):
                i_x, i_y, i_z = cols.index("x"), cols.index("y"), cols.index("z")
                coord_mode = "abs"
            elif all(c in cols for c in ("xu", "yu", "zu")):
                i_x, i_y, i_z = cols.index("xu"), cols.index("yu"), cols.index("zu")
                coord_mode = "abs"
            elif all(c in cols for c in ("xs", "ys", "zs")):
                i_x, i_y, i_z = cols.index("xs"), cols.index("ys"), cols.index("zs")
                coord_mode = "scaled"
            else:
                raise ValueError(
                    f"{path}: ATOMS header must contain (x,y,z) or (xu,yu,zu) or (xs,ys,zs). Found: {cols}"
                )

            # Read atom data
            types = np.empty(natoms, dtype=np.int32)
            coords = np.empty((natoms, 3), dtype=np.float64)
            mol_ids = np.empty(natoms, dtype=np.int32) if i_mol is not None else None

            for i in range(natoms):
                parts = fh.readline().split()
                types[i] = int(parts[i_type])
                if i_mol is not None:
                    mol_ids[i] = int(parts[i_mol])
                x, y, z = float(parts[i_x]), float(parts[i_y]), float(parts[i_z])
                if coord_mode == "scaled":
                    coords[i] = box_lo + np.array([x, y, z]) * L
                else:
                    coords[i, 0] = x
                    coords[i, 1] = y
                    coords[i, 2] = z

            if fold_coords:
                coords = _fold_into_box(coords, box_lo, L)

            # --- one-time validations on first yielded frame ---
            if first_natoms is None:
                first_natoms = natoms
                first_L = L.copy()
                # Validate bead types
                present = set(np.unique(types))
                missing = []
                if A_type_id not in present:
                    missing.append(f"A_type_id={A_type_id}")
                if B_type_id not in present:
                    missing.append(f"B_type_id={B_type_id}")
                if missing:
                    raise ValueError(
                        f"{path}: type IDs not found in first frame: {', '.join(missing)}. "
                        f"Present types: {sorted(present)}"
                    )
            else:
                if natoms != first_natoms and not allow_variable_natoms:
                    warnings.warn(
                        f"{path}: N_atoms changed from {first_natoms} to {natoms} at step {timestep}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                if first_L is not None and not np.allclose(L, first_L, rtol=1e-4) and not L_warned:
                    warnings.warn(
                        f"{path}: box dimensions changed at step {timestep}. "
                        f"Initial L={first_L}, current L={L}. "
                        "Analysis assumes a fixed periodic box.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    L_warned = True

            frame_index += 1
            yield FrameData(
                timestep=int(timestep),
                box_lo=box_lo.copy(),
                box_hi=box_hi.copy(),
                types=types,
                coords=coords,
                mol_ids=mol_ids,
            )


def final_timestep(path: str | Path, tail_bytes: int = 1 << 23) -> int:
    """Return the timestep of the last complete frame in a dump file.

    Only the tail of the file is read: *tail_bytes* is doubled until an
    ``ITEM: TIMESTEP`` header with a complete record after it is found, so the
    cost does not scale with file size. The default (8 MiB) covers a single
    frame of a few million beads.

    A trailing header truncated by an interrupted write is skipped in favour of
    the previous complete one.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Trajectory not found: {path}")
    size = path.stat().st_size
    if size == 0:
        raise ValueError(f"{path}: file is empty")

    marker = b"ITEM: TIMESTEP"
    chunk = min(max(int(tail_bytes), 4096), size)

    with path.open("rb") as fh:
        while True:
            fh.seek(size - chunk)
            buf = fh.read(chunk)

            # Walk markers from the end backwards; the first complete frame
            # wins. Merely finding a timestep plus the following ITEM header is
            # insufficient because interrupted jobs often leave a truncated
            # final atom block.
            search_end = len(buf)
            while True:
                idx = buf.rfind(marker, 0, search_end)
                if idx == -1:
                    break
                text = buf[idx:].decode("utf-8", errors="replace")
                lines = text.splitlines()
                try:
                    if (
                        len(lines) >= 9
                        and lines[0].startswith("ITEM: TIMESTEP")
                        and lines[1].strip().isdigit()
                        and lines[2].startswith("ITEM: NUMBER OF ATOMS")
                        and lines[3].strip().isdigit()
                        and lines[4].startswith("ITEM: BOX BOUNDS")
                        and lines[8].startswith("ITEM: ATOMS")
                    ):
                        natoms = int(lines[3].strip())
                        if len(lines) >= 9 + natoms:
                            return int(lines[1].strip())
                except (ValueError, IndexError):
                    pass
                search_end = idx

            if chunk >= size:
                raise ValueError(
                    f"{path}: no complete 'ITEM: TIMESTEP' record found in the file"
                )
            chunk = min(chunk * 2, size)


def count_frames(path: str | Path, stride: int = 1) -> int:
    """Return the number of frames that would be yielded by iter_frames (without reading atoms)."""
    path = Path(path)
    n = 0
    frame_index = 0
    with path.open("r") as fh:
        for line in fh:
            if line.startswith("ITEM: TIMESTEP"):
                if frame_index % stride == 0:
                    n += 1
                frame_index += 1
    return n
