"""
Fast I/O for MD trajectory files.

Provides public helpers used by :mod:`fishmol.trj`:

- :func:`read_extxyz` — memory-mapped reader for extended XYZ files that
  parses per-frame ``Lattice=`` fields, enabling NPT trajectories.
- :func:`write_extxyz` — writer that embeds ``Lattice=`` in every comment
  line so files are round-trippable.
- :func:`read_lammpstrj` — reader for LAMMPS dump files (``.lammpstrj``),
  supporting both orthogonal and triclinic boxes, all coordinate styles
  (``x y z``, ``xs ys zs``, ``xu yu zu``), and both ``element`` and integer
  ``type`` columns.

Extended XYZ comment-line conventions understood:

* ASE / OVITO standard: ``Lattice="a1x a1y a1z a2x … a3z" Properties=…``
* Alternative key:       ``cell="…"`` (same 9-float format)
* CP2K / custom format:  ``i = N, time = T, cell = "…"``  (comma-separated)
"""

import mmap
import os
import re
import numpy as np
from typing import Any, List, Optional, Tuple, Union
from recordclass import make_dataclass, dataobject


# ── Comment-line parsing ───────────────────────────────────────────────────────

# Matches  key="quoted value"  OR  key=bare_value  (no commas/spaces in value)
_KV_RE = re.compile(
    r'(\w+)\s*=\s*(?:"([^"]*)"|(.+?))(?=\s*\w+\s*=\s*|$)',
    re.IGNORECASE,
)


def _parse_comment(line: str) -> dict:
    """Return a lower-cased ``{key: value_string}`` dict from a comment line."""
    return {
        m.group(1).lower(): (m.group(2) if m.group(2) is not None else m.group(3))
        for m in _KV_RE.finditer(line)
    }


def _extract_cell(d: dict) -> Optional[np.ndarray]:
    """Return a ``(3, 3)`` float64 cell matrix from a parsed comment dict.

    Tries both ``lattice`` and ``cell`` keys.  Accepts 9 floats (general
    triclinic) or 3 floats (orthogonal diagonal).  Returns ``None`` when
    neither key is present or the value cannot be parsed.
    """
    for key in ('lattice', 'Lattice', 'cell', 'Cell'):
        raw = d.get(key)
        if raw is None:
            continue
        vals = raw.split()
        if len(vals) == 9:
            return np.array(vals, dtype=float).reshape(3, 3)
        if len(vals) == 3:
            return np.diag(np.array(vals, dtype=float))

    # Try extracting from 'properties' value (e.g. "Properties = ..., Cell: [[...]]")
    props = d.get('properties')
    if props:
        m = re.search(r'(?:Cell|Lattice):\s*([^,]+)', props)
        if m:
            val_str = m.group(1).strip()
            vals = re.findall(r"[-+]?\d*\.\d+|\d+", val_str)
            if len(vals) == 9:
                return np.array(vals, dtype=float).reshape(3, 3)
            if len(vals) == 3:
                return np.diag(np.array(vals, dtype=float))
    return None


def _col_indices(comment: str) -> Tuple[int, slice]:
    """Parse ``Properties=`` to find symbol and position column offsets.

    Returns ``(sym_col, pos_slice)`` where *sym_col* is the integer column
    index of the element symbol and *pos_slice* is a slice for the three
    Cartesian coordinates.

    Falls back to ``(0, slice(1, 4))`` for non-standard or absent
    ``Properties`` fields (symbol in column 0, xyz in columns 1–3).
    """
    m = re.search(r'[Pp]roperties\s*=\s*(?:"([^"]*)"|([^\s"]+))', comment)
    if not m:
        return 0, slice(1, 4)

    spec = m.group(1) or m.group(2)
    parts = spec.split(':')
    col = 0
    sym_col = 0
    pos_slice = slice(1, 4)

    i = 0
    while i + 2 < len(parts):
        name = parts[i].lower()
        kind = parts[i + 1].upper()
        try:
            n = int(parts[i + 2])
        except ValueError:
            break
        if name == 'species' and kind == 'S':
            sym_col = col
        elif name in ('pos', 'positions') and kind == 'R' and n == 3:
            pos_slice = slice(col, col + n)
        col += n
        i += 3

    return sym_col, pos_slice


# ── Public I/O functions ───────────────────────────────────────────────────────

def read_extxyz(
    path: str,
    index: Union[str, slice, int] = ':',
    timestep: float = 1.0,
    cell: Any = None,
) -> Tuple[int, int, list, float]:
    """Read an extended XYZ trajectory file.

    Per-frame ``Lattice=`` (or ``cell=``) fields in the comment line are
    parsed and attached to each returned :class:`~fishmol.atoms.Atoms` frame,
    enabling NPT trajectories where the simulation cell evolves over time.

    When a frame carries no cell information in its comment line the *cell*
    fallback argument is used.  If *cell* is also ``None`` the frame's
    ``.cell.lattice`` attribute will be ``None`` — geometry methods that rely
    on periodic boundary conditions (MIC distances, wrapping) will raise in
    that case.

    Parameters
    ----------
    path : str
        Path to the ``.xyz`` / ``.extxyz`` file.
    index : str, int, or slice
        Frame selection:

        * ``':'`` or ``'all'`` — every frame in the file.
        * Integer *n* — single frame (negative indices count from the end).
        * ``slice(start, stop, step)`` — a strided sub-range.

    timestep : float
        Nominal time step between consecutive frames in **fs**.  When a slice
        step *k* is used the returned effective timestep is ``timestep * k``.
    cell : array-like of shape (3, 3) or None
        Fallback cell (lattice vectors as rows, in Å) used for frames that
        carry no ``Lattice=`` field.  Pass an explicit value for NVT
        trajectories produced by codes that omit cell data from each frame
        (e.g. CP2K ``FORMAT XYZ``).  Ignored when the file already contains
        per-frame cell information.

    Returns
    -------
    natoms : int
        Number of atoms per frame (must be constant throughout the file).
    nframes : int
        Total number of frames present in the file, **before** slicing.
    frames : list of :class:`~fishmol.atoms.Atoms`
        Selected frames with per-frame cells attached.
    effective_timestep : float
        Adjusted time step ``timestep * slice_step`` in fs.

    Raises
    ------
    ValueError
        If the file is empty or the first line cannot be parsed as an integer.
    IndexError
        If a requested frame index is out of range.

    Notes
    -----
    File reading uses :mod:`mmap` with ``ACCESS_READ`` (cross-platform; works
    on both Windows and Unix).  The entire file is memory-mapped but only the
    selected frame blocks are decoded, keeping peak memory low for large files.
    """
    from fishmol.atoms import Atoms

    dc_cell = make_dataclass("Cell", "lattice")

    # ── 1. Memory-map the file ────────────────────────────────────────────────
    with open(path, 'rb') as fh:
        mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        raw = [
            line.decode('utf-8', errors='replace').rstrip('\r\n')
            for line in iter(mm.readline, b'')
        ]
        mm.close()

    if not raw:
        raise ValueError(f"File {path!r} is empty.")

    # ── 2. Detect natoms and total frame count ────────────────────────────────
    try:
        natoms = int(raw[0].strip())
    except ValueError:
        raise ValueError(
            f"Expected atom count on line 1 of {path!r}, got: {raw[0]!r}"
        )
    block = natoms + 2

    # Count only lines that look like a valid atom-count header to be robust
    # against trailing blank lines.
    nframes = 0
    for i in range(0, len(raw), block):
        if i < len(raw) and raw[i].strip().lstrip('-').isdigit():
            nframes += 1

    # ── 3. Resolve index → ordered list of frame numbers ─────────────────────
    step = 1
    if index in (':', 'all'):
        frame_ids = list(range(nframes))
    elif isinstance(index, int):
        n = index if index >= 0 else nframes + index
        if n < 0 or n >= nframes:
            raise IndexError(
                f"Frame index {index} is out of range for a trajectory with "
                f"{nframes} frames."
            )
        frame_ids = [n]
    elif isinstance(index, slice):
        start = index.start or 0
        stop = index.stop if index.stop is not None else nframes
        step = index.step or 1
        frame_ids = list(range(start, stop, step))
    else:
        frame_ids = list(index)

    effective_timestep = timestep * step

    # ── 4. Fallback cell ──────────────────────────────────────────────────────
    if cell is not None:
        _fallback = dc_cell(np.asarray(cell, dtype=float))
    else:
        _fallback = None

    # ── 5. Detect column layout from the first frame comment ─────────────────
    first_comment = raw[1] if len(raw) > 1 else ''
    sym_col, pos_slice = _col_indices(first_comment)

    # ── 6. Parse each selected frame ─────────────────────────────────────────
    frames: List[Atoms] = []
    for fi in frame_ids:
        base = fi * block
        comment = raw[base + 1]
        atom_lines = raw[base + 2: base + 2 + natoms]

        # Cell: prefer per-frame Lattice=, then fallback, then None
        cell_dict = _parse_comment(comment)
        cell_arr = _extract_cell(cell_dict)
        if cell_arr is not None:
            frame_cell = dc_cell(cell_arr)
        elif _fallback is not None:
            frame_cell = _fallback
        else:
            frame_cell = dc_cell(None)

        # Atom symbols and Cartesian positions
        symbs: List[str] = []
        positions: List[List[float]] = []
        for ln in atom_lines:
            cols = ln.split()
            symbs.append(cols[sym_col])
            positions.append([float(v) for v in cols[pos_slice]])

        frames.append(
            Atoms(
                symbs=np.array(symbs, dtype='<U2'),
                pos=np.array(positions, dtype=float),
                cell=frame_cell,
            )
        )

    return natoms, nframes, frames, effective_timestep


def write_extxyz(
    frames: list,
    filename: str,
    natoms: int,
    timestep: float,
) -> None:
    """Write a list of :class:`~fishmol.atoms.Atoms` frames to extended XYZ.

    Each frame's cell is written as a ``Lattice=`` field in the comment line
    so that the file is fully round-trippable by :func:`read_extxyz`.  Frames
    whose cell is ``None`` or has no lattice data omit the ``Lattice=`` field.

    Parameters
    ----------
    frames : list of :class:`~fishmol.atoms.Atoms`
        Frames to write (in order).
    filename : str
        Output file path.  If the file already exists a ``-1`` (then ``-2``,
        etc.) counter is appended before the extension until a free name is
        found.
    natoms : int
        Atoms per frame — written as the first line of each XYZ block.
    timestep : float
        Effective timestep in **fs** (already accounting for any trajectory
        stride).  Used to compute per-frame timestamps in the comment line.

    Notes
    -----
    Column format: ``%-3s %18.8f %18.8f %18.8f`` (symbol, x, y, z).
    The comment line follows the ASE / OVITO extxyz convention:
    ``Lattice="…" Properties=species:S:1:pos:R:3 frame=N time=T fs``
    """
    # Avoid silently overwriting existing files
    if os.path.exists(filename):
        base, ext = os.path.splitext(filename)
        counter = 1
        candidate = f"{base}-{counter}{ext}"
        while os.path.exists(candidate):
            counter += 1
            candidate = f"{base}-{counter}{ext}"
        filename = candidate
        print(f"File already exists — output written to {filename!r}")

    with open(filename, 'w', encoding='utf-8') as fh:
        for i, frame in enumerate(frames):
            # Build Lattice= string if cell data is available
            lattice_str: Optional[str] = None
            fc = frame.cell
            if fc is not None:
                lat = getattr(fc, 'lattice', None)
                if lat is not None:
                    lat = np.asarray(lat)
                    if lat.shape == (3, 3):
                        lattice_str = ' '.join(f'{v:.8f}' for v in lat.flatten())

            fh.write(f"{natoms}\n")
            comment = (
                f'Properties=species:S:1:pos:R:3 '
                f'frame={i} time={i * timestep:.3f} fs'
            )
            if lattice_str is not None:
                comment = f'Lattice="{lattice_str}" ' + comment
            fh.write(comment + '\n')

            for sym, (x, y, z) in zip(frame.symbs, frame.pos):
                fh.write(f"{sym:<3s} {x:>18.8f} {y:>18.8f} {z:>18.8f}\n")


# ── LAMMPS dump reader ─────────────────────────────────────────────────────────

def _lammps_cell(
    box_header: str,
    bound_lines: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Parse an ``ITEM: BOX BOUNDS`` block into a cell matrix and origin.

    Handles both orthogonal (6 values) and triclinic (9 values, ``xy xz yz``
    tilt factors) LAMMPS boxes.

    Parameters
    ----------
    box_header : str
        The ``ITEM: BOX BOUNDS …`` line (used to detect triclinic).
    bound_lines : list of str
        The three data lines that follow (xlo/xhi[/xy], ylo/yhi[/xz],
        zlo/zhi[/yz]).

    Returns
    -------
    cell : (3, 3) ndarray
        Lattice vectors as rows: ``a = cell[0]``, ``b = cell[1]``,
        ``c = cell[2]``.
    origin : (3,) ndarray
        Cartesian position of the box corner ``(xlo, ylo, zlo)``.
    """
    triclinic = 'xy' in box_header

    vals = [list(map(float, ln.split())) for ln in bound_lines]

    if triclinic:
        xlo_b, xhi_b, xy = vals[0][0], vals[0][1], vals[0][2]
        ylo_b, yhi_b, xz = vals[1][0], vals[1][1], vals[1][2]
        zlo_b, zhi_b, yz = vals[2][0], vals[2][1], vals[2][2]

        # Recover true lo/hi from the reported boundary-extended values
        xlo = xlo_b - min(0.0, xy, xz, xy + xz)
        xhi = xhi_b - max(0.0, xy, xz, xy + xz)
        ylo = ylo_b - min(0.0, yz)
        yhi = yhi_b - max(0.0, yz)
        zlo, zhi = zlo_b, zhi_b

        lx, ly, lz = xhi - xlo, yhi - ylo, zhi - zlo
        cell = np.array([
            [lx,  0.0, 0.0],
            [xy,  ly,  0.0],
            [xz,  yz,  lz],
        ])
    else:
        xlo, xhi = vals[0][0], vals[0][1]
        ylo, yhi = vals[1][0], vals[1][1]
        zlo, zhi = vals[2][0], vals[2][1]
        cell = np.diag([xhi - xlo, yhi - ylo, zhi - zlo])

    return cell.astype(float), np.array([xlo, ylo, zlo], dtype=float)


def read_lammpstrj(
    path: str,
    index: Union[str, slice, int] = ':',
    timestep: float = 1.0,
    cell: Any = None,
    type_map: Optional[dict] = None,
    sort_by_id: bool = True,
) -> Tuple[int, int, list, float]:
    """Read a LAMMPS dump trajectory file.

    Supports both orthogonal and triclinic simulation boxes, and all three
    LAMMPS coordinate styles:

    * ``x y z``   — wrapped Cartesian (metal/real units: Å)
    * ``xu yu zu`` — unwrapped Cartesian
    * ``xs ys zs`` — scaled (fractional) coordinates

    Atom symbols are read from an ``element`` column when present.  When only
    an integer ``type`` column is available, *type_map* must be supplied.

    Positions are stored relative to the box origin so that they are
    consistent with the cell matrix (fractional coords in ``[0, 1)`` after
    wrapping, MIC operations work correctly).

    Parameters
    ----------
    path : str
        Path to the LAMMPS dump / ``.lammpstrj`` file.
    index : str, int, or slice
        Frame selection — same semantics as :func:`read_extxyz`.
    timestep : float
        Nominal time step between consecutive stored frames in **fs**.
    cell : array-like of shape (3, 3) or None
        Optional cell override applied to every frame (rarely needed; each
        frame's box is read from the file header).
    type_map : dict or None
        Mapping from integer LAMMPS atom type to element symbol, e.g.
        ``{1: 'O', 2: 'H'}``.  Required when the dump uses a ``type``
        column.  Ignored when an ``element`` column is present.
    sort_by_id : bool
        If True (default), atom rows are reordered by LAMMPS atom ``id``
        so that the same physical atom occupies the same row in every frame.

    Returns
    -------
    natoms : int
    nframes : int
    frames : list of :class:`~fishmol.atoms.Atoms`
    effective_timestep : float

    Raises
    ------
    ValueError
        If the file is empty, contains no ``ITEM: TIMESTEP`` blocks, or an
        integer ``type`` column is found without a *type_map*.
    IndexError
        If the requested frame index is out of range.

    Examples
    --------
    Orthogonal NVT dump with ``element`` column:

    >>> natoms, nframes, frames, dt = read_lammpstrj("dump.lammpstrj")

    Triclinic NPT dump with integer type column:

    >>> natoms, nframes, frames, dt = read_lammpstrj(
    ...     "dump.lammpstrj",
    ...     type_map={1: 'O', 2: 'H'},
    ... )
    """
    from fishmol.atoms import Atoms

    dc_cell = make_dataclass("Cell", "lattice")

    with open(path, 'rb') as fh:
        mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        raw = [
            line.decode('utf-8', errors='replace').rstrip('\r\n')
            for line in iter(mm.readline, b'')
        ]
        mm.close()

    if not raw:
        raise ValueError(f"File {path!r} is empty.")

    # Each frame starts with "ITEM: TIMESTEP"
    frame_starts = [
        i for i, ln in enumerate(raw)
        if ln.strip().startswith("ITEM: TIMESTEP")
    ]
    if not frame_starts:
        raise ValueError(
            f"No LAMMPS frames (ITEM: TIMESTEP) found in {path!r}."
        )

    nframes = len(frame_starts)
    natoms = int(raw[frame_starts[0] + 3].strip())

    # ── Resolve index ─────────────────────────────────────────────────────────
    step = 1
    if index in (':', 'all'):
        frame_ids = list(range(nframes))
    elif isinstance(index, int):
        n = index if index >= 0 else nframes + index
        if n < 0 or n >= nframes:
            raise IndexError(
                f"Frame index {index} is out of range for a trajectory with "
                f"{nframes} frames."
            )
        frame_ids = [n]
    elif isinstance(index, slice):
        start = index.start or 0
        stop = index.stop if index.stop is not None else nframes
        step = index.step or 1
        frame_ids = list(range(start, stop, step))
    else:
        frame_ids = list(index)

    effective_timestep = timestep * step

    _fallback_cell = dc_cell(np.asarray(cell, dtype=float)) if cell is not None else None

    # ── Parse frames ──────────────────────────────────────────────────────────
    frames: List = []
    for fi in frame_ids:
        s = frame_starts[fi]
        # Frame layout (line offsets from s):
        #  0: ITEM: TIMESTEP
        #  1: <timestep>
        #  2: ITEM: NUMBER OF ATOMS
        #  3: <natoms>
        #  4: ITEM: BOX BOUNDS [xy xz yz] <px> <py> <pz>
        #  5-7: box bound data (3 lines)
        #  8: ITEM: ATOMS <col1> <col2> …
        #  9…9+natoms-1: atom data

        natoms_frame = int(raw[s + 3].strip())
        cell_arr, origin = _lammps_cell(raw[s + 4], raw[s + 5: s + 8])

        col_names = raw[s + 8].replace("ITEM: ATOMS", "").split()
        col_idx = {name: j for j, name in enumerate(col_names)}

        # Symbol column
        if 'element' in col_idx:
            sym_c = col_idx['element']
            use_type_map = False
        elif 'type' in col_idx:
            sym_c = col_idx['type']
            use_type_map = True
        else:
            raise ValueError(
                f"Frame {fi} in {path!r} has neither 'element' nor 'type' "
                f"column in: {raw[s + 8]!r}"
            )

        if use_type_map and type_map is None:
            raise ValueError(
                f"LAMMPS dump {path!r} uses integer atom types but no "
                f"type_map was provided.  Pass type_map={{1: 'O', 2: 'H', …}}."
            )

        # Coordinate columns — prefer scaled, then unwrapped, then wrapped
        if 'xs' in col_idx:
            cx, cy, cz = col_idx['xs'], col_idx['ys'], col_idx['zs']
            coord_style = 'scaled'
        elif 'xsu' in col_idx:
            cx, cy, cz = col_idx['xsu'], col_idx['ysu'], col_idx['zsu']
            coord_style = 'scaled'
        elif 'xu' in col_idx:
            cx, cy, cz = col_idx['xu'], col_idx['yu'], col_idx['zu']
            coord_style = 'cartesian'
        else:
            cx, cy, cz = col_idx['x'], col_idx['y'], col_idx['z']
            coord_style = 'cartesian'

        id_c = col_idx.get('id', None)

        ids: List[int] = []
        symbs: List[str] = []
        coords: List[List[float]] = []

        for ln in raw[s + 9: s + 9 + natoms_frame]:
            parts = ln.split()
            if id_c is not None:
                ids.append(int(parts[id_c]))
            symbs.append(
                parts[sym_c] if not use_type_map
                else type_map[int(parts[sym_c])]
            )
            coords.append([float(parts[cx]), float(parts[cy]), float(parts[cz])])

        arr = np.array(coords, dtype=float)

        # Sort by atom ID for consistent row order across frames
        if sort_by_id and ids:
            order = np.argsort(ids)
            arr = arr[order]
            symbs = [symbs[j] for j in order]

        # Convert to Cartesian, origin-relative
        if coord_style == 'scaled':
            # r = xs*a + ys*b + zs*c  ↔  positions @ cell_arr
            # (cell_arr rows are a, b, c)
            positions = arr @ cell_arr
        else:
            # Absolute Cartesian → shift to box-origin frame
            positions = arr - origin

        frame_cell = _fallback_cell if _fallback_cell is not None else dc_cell(cell_arr)

        frames.append(
            Atoms(
                symbs=np.array(symbs, dtype='<U2'),
                pos=positions,
                cell=frame_cell,
            )
        )

    return natoms, nframes, frames, effective_timestep
