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
from recordclass import make_dataclass


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


def _extract_cell(d: dict, line: str) -> Optional[np.ndarray]:
    """Return a ``(3, 3)`` float64 cell matrix from a parsed comment dict.

    Tries both ``lattice`` and ``cell`` keys.  Accepts 9 floats (general
    triclinic) or 3 floats (orthogonal diagonal).  Returns ``None`` when
    neither key is present or the value cannot be parsed.

    Includes a fallback search on the raw comment line to handle CP2K-style
    nested brackets and comma-separated values that break standard KV parsing.
    """
    # Regex for floats including scientific notation
    float_re = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"

    # 1. Dictionary Lookup (ASE / OVITO style)
    # We check the dict first for standard Lattice=" " entries.
    for key in ('lattice', 'cell'):
        raw = d.get(key)
        if raw:
            vals = re.findall(float_re, raw)
            if len(vals) == 9:
                return np.array(vals, dtype=float).reshape(3, 3)
            if len(vals) == 3:
                return np.diag(np.array(vals, dtype=float))

    # 2. Raw Line Fallback (CP2K style: "Cell: [[...]]")
    # We look for the keyword and capture everything until the next potential 
    # key-value pair or the end of the line, including brackets.
    m = re.search(r'(?:Cell|Lattice)[:=]\s*([\[\s\d.,eE+-\]]+)', line, re.IGNORECASE)
    if m:
        vals = re.findall(float_re, m.group(1))
        if len(vals) >= 9:
            # For CP2K, we take the first 9 floats found after the 'Cell' keyword.
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
    col, sym_col, pos_slice = 0, 0, slice(1, 4)

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
    """
    from fishmol.atoms import Atoms
    dc_cell = make_dataclass("Cell", "lattice")

    # ── 1. Memory-map the file ────────────────────────────────────────────────
    with open(path, 'rb') as fh:
        mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        raw = [line.decode('utf-8', errors='replace').rstrip('\r\n') for line in iter(mm.readline, b'')]
        mm.close()

    if not raw: raise ValueError(f"File {path!r} is empty.")

    # ── 2. Detect natoms and total frame count ────────────────────────────────
    try:
        natoms = int(raw[0].strip())
    except ValueError:
        raise ValueError(f"Expected atom count on line 1, got: {raw[0]!r}")

    block = natoms + 2
    nframes = sum(1 for i in range(0, len(raw), block) if raw[i].strip().lstrip('-').isdigit())

    # ── 3. Resolve index → ordered list of frame numbers ─────────────────────
    if index in (':', 'all'): frame_ids = list(range(nframes))
    elif isinstance(index, int):
        n = index if index >= 0 else nframes + index
        frame_ids = [n]
    elif isinstance(index, slice):
        frame_ids = list(range(index.start or 0, index.stop or nframes, index.step or 1))
    else: frame_ids = list(index)

    step = index.step if isinstance(index, slice) and index.step else 1
    effective_timestep = timestep * step

    # ── 4. Fallback cell ──────────────────────────────────────────────────────
    _fallback = dc_cell(np.asarray(cell, dtype=float)) if cell is not None else None

    # ── 5. Detect column layout from the first frame comment ─────────────────
    sym_col, pos_slice = _col_indices(raw[1] if len(raw) > 1 else '')

    # ── 6. Parse each selected frame ─────────────────────────────────────────
    frames: List[Atoms] = []
    for fi in frame_ids:
        base = fi * block
        if base + 1 >= len(raw): break
        comment = raw[base + 1]
        
        # Cell: prefer per-frame Lattice=, then fallback, then None
        cell_dict = _parse_comment(comment)
        cell_arr = _extract_cell(cell_dict, comment)
        
        if cell_arr is not None:
            frame_cell = dc_cell(cell_arr)
        elif _fallback is not None:
            frame_cell = _fallback
        else:
            frame_cell = dc_cell(None)

        # Atom symbols and Cartesian positions
        atom_lines = raw[base + 2: base + 2 + natoms]
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


def write_extxyz(frames: list, filename: str, natoms: int, timestep: float) -> None:
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
            lattice_str = None
            fc = frame.cell
            if fc is not None:
                lat = getattr(fc, 'lattice', None)
                if lat is not None:
                    lat = np.asarray(lat)
                    if lat.shape == (3, 3):
                        lattice_str = ' '.join(f'{v:.8f}' for v in lat.flatten())

            fh.write(f"{natoms}\n")
            comment = f'Properties=species:S:1:pos:R:3 frame={i} time={i * timestep:.3f} fs'
            if lattice_str is not None:
                comment = f'Lattice="{lattice_str}" ' + comment
            fh.write(comment + '\n')

            for sym, (x, y, z) in zip(frame.symbs, frame.pos):
                fh.write(f"{sym:<3s} {x:>18.8f} {y:>18.8f} {z:>18.8f}\n")