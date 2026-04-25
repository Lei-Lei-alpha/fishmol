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
    d = {
        m.group(1).lower(): (m.group(2) if m.group(2) is not None else m.group(3))
        for m in _KV_RE.finditer(line)
    }
    # Minimal addition: extract CP2K comma-separated 'key: value' pairs safely
    for m in re.finditer(r'(\w+)\s*[:=]\s*([^,]+)', line):
        k = m.group(1).lower()
        if k not in d:
            d[k] = m.group(2).strip()
    return d


def _extract_cell(d: dict, line: str) -> Optional[np.ndarray]:
    """Return a ``(3, 3)`` float64 cell matrix from a parsed comment dict.

    Tries both ``lattice`` and ``cell`` keys.  Accepts 9 floats (general
    triclinic) or 3 floats (orthogonal diagonal).  Returns ``None`` when
    neither key is present or the value cannot be parsed.
    """
    float_re = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
    
    for key in ('lattice', 'Lattice', 'cell', 'Cell'):
        raw = d.get(key)
        if raw is None:
            continue
        vals = re.findall(float_re, raw)
        if len(vals) == 9:
            return np.array(vals, dtype=float).reshape(3, 3)
        if len(vals) == 3:
            return np.diag(np.array(vals, dtype=float))

    # Minimal addition: CP2K nested brackets fallback using the raw line
    m = re.search(r'(?:Cell|Lattice)\s*[:=]\s*([\[\s\d.,eE+-\]]+)', line, re.IGNORECASE)
    if m:
        vals = re.findall(float_re, m.group(1))
        if len(vals) >= 9:
            return np.array(vals[:9], dtype=float).reshape(3, 3)
        if len(vals) >= 3:
            return np.diag(np.array(vals[:3], dtype=float))
            
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
    timestep: Optional[float] = None,
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

    timestep : float, optional
        Nominal time step between consecutive frames in **fs**.  When a slice
        step *k* is used the returned effective timestep is ``timestep * k``.
        If None, attempts to infer from frame time stamps.
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
    times: List[float] = []

    for fi in frame_ids:
        base = fi * block
        comment = raw[base + 1]
        atom_lines = raw[base + 2: base + 2 + natoms]

        # Cell: prefer per-frame Lattice=, then fallback, then None
        cell_dict = _parse_comment(comment)
        cell_arr = _extract_cell(cell_dict, comment)
        if cell_arr is not None:
            frame_cell = dc_cell(cell_arr)
        elif _fallback is not None:
            frame_cell = _fallback
        else:
            frame_cell = dc_cell(None)
            
        # Minimal addition: Metadata auto-extraction (time and frame index)
        t_raw = cell_dict.get('t', cell_dict.get('time'))
        t_val = 0.0
        if t_raw:
            t_match = re.search(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", str(t_raw))
            if t_match:
                t_val = float(t_match.group(1))
                if 'ps' in str(t_raw).lower(): t_val *= 1000.0
        times.append(t_val)
        
        idx_raw = cell_dict.get('frame', cell_dict.get('step', fi))
        idx_match = re.search(r'\d+', str(idx_raw))
        frame_idx = int(idx_match.group()) if idx_match else fi

        # Atom symbols and Cartesian positions
        symbs: List[str] = []
        positions: List[List[float]] = []
        for ln in atom_lines:
            cols = ln.split()
            symbs.append(cols[sym_col])
            positions.append([float(v) for v in cols[pos_slice]])

        frame_obj = Atoms(
            symbs=np.array(symbs, dtype='<U2'),
            pos=np.array(positions, dtype=float),
            cell=frame_cell,
        )
        frame_obj.time = t_val
        frame_obj.step = frame_idx
        frames.append(frame_obj)

    # Infer timestep if missing
    if timestep is None and len(times) > 1:
        base_ts = times[1] - times[0]
    else:
        base_ts = timestep or 1.0

    effective_timestep = base_ts * step

    return natoms, nframes, frames, effective_timestep


def write_extxyz(
    frames: list,
    filename: str,
    natoms: int,
    timestep: float,
) -> None:
    """Write a list of :class:`~fishmol.atoms.Atoms` frames to extended XYZ."""
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
            lattice_str: Optional[str] = None
            fc = frame.cell
            if fc is not None:
                lat = getattr(fc, 'lattice', None)
                if lat is not None:
                    lat = np.asarray(lat)
                    if lat.shape == (3, 3):
                        lattice_str = ' '.join(f'{v:.8f}' for v in lat.flatten())

            fh.write(f"{natoms}\n")
            # Pull step/time from frame if available
            f_step = getattr(frame, "step", i)
            f_time = getattr(frame, "time", i * timestep)
            comment = (
                f'Properties=species:S:1:pos:R:3 '
                f'frame={f_step} time={f_time:.3f} fs'
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
    """Parse an ``ITEM: BOX BOUNDS`` block into a cell matrix and origin."""
    triclinic = 'xy' in box_header
    vals = [list(map(float, ln.split())) for ln in bound_lines]

    if triclinic:
        xlo_b, xhi_b, xy = vals[0][0], vals[0][1], vals[0][2]
        ylo_b, yhi_b, xz = vals[1][0], vals[1][1], vals[1][2]
        zlo_b, zhi_b, yz = vals[2][0], vals[2][1], vals[2][2]
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
    timestep: Optional[float] = None,
    cell: Any = None,
    type_map: Optional[dict] = None,
    sort_by_id: bool = True,
) -> Tuple[int, int, list, float]:
    """Read a LAMMPS dump trajectory file."""
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

    base_ts = timestep or 1.0
    effective_timestep = base_ts * step

    _fallback_cell = dc_cell(np.asarray(cell, dtype=float)) if cell is not None else None

    frames: List = []
    for fi in frame_ids:
        s = frame_starts[fi]
        natoms_frame = int(raw[s + 3].strip())
        cell_arr, origin = _lammps_cell(raw[s + 4], raw[s + 5: s + 8])

        col_names = raw[s + 8].replace("ITEM: ATOMS", "").split()
        col_idx = {name: j for j, name in enumerate(col_names)}

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

        if sort_by_id and ids:
            order = np.argsort(ids)
            arr = arr[order]
            symbs = [symbs[j] for j in order]

        if coord_style == 'scaled':
            positions = arr @ cell_arr
        else:
            positions = arr - origin

        frame_cell = _fallback_cell if _fallback_cell is not None else dc_cell(cell_arr)
        
        # Read LAMMPS step
        step_val = int(raw[s + 1].strip())
        
        frame_obj = Atoms(
            symbs=np.array(symbs, dtype='<U2'),
            pos=positions,
            cell=frame_cell,
        )
        frame_obj.step = step_val
        frame_obj.time = step_val * base_ts
        frames.append(frame_obj)

    return natoms, nframes, frames, effective_timestep