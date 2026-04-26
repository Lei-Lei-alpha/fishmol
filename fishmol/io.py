"""
Fast I/O for extended XYZ trajectory files and LAMMPS trajectories.

Provides public helpers used by :mod:`fishmol.trj`:

- :func:`read_extxyz` — memory-mapped reader that parses per-frame
  ``Lattice=`` fields, enabling NPT trajectories where the cell changes
  between frames.
- :func:`write_extxyz` — writer that embeds ``Lattice=`` in every comment
  line so files are round-trippable.
- :func:`read_trajectory_files` — flexible loader for folders (CP2K, LAMMPS)
  or specific sets of files.

Extended XYZ comment-line conventions understood:

* ASE / OVITO standard: ``Lattice="a1x a1y a1z a2x … a3z" Properties=…``
* Alternative key:       ``cell="…"`` (same 9-float format)
* CP2K / custom format:  ``i = N, time = T, cell = "…"``  (comma-separated)
"""

import mmap
import os
import re
import glob
import numpy as np
from typing import Any, List, Optional, Tuple, Union, Dict
from recordclass import make_dataclass, dataobject


# ── Comment-line parsing ───────────────────────────────────────────────────────

# Matches  key="quoted value"  OR  key=bare_value  (no commas/spaces in value)
_KV_RE = re.compile(
    r'(\w+)\s*=\s*(?:"([^"]*)"|([^\s,"]+))',
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
    for key in ('lattice', 'cell'):
        raw = d.get(key)
        if raw is None:
            continue
        vals = raw.split()
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


# ── Flexible Trajectory Reading ────────────────────────────────────────────────

def read_trajectory_files(
    path: str,
    index: Union[str, slice, int] = ':',
    timestep: Optional[float] = None,
    cell_path: Optional[str] = None,
    frc_path: Optional[str] = None,
    vel_path: Optional[str] = None,
) -> Tuple[int, int, list]:
    """Read a trajectory from multiple files or a simulation folder (.xyz, .cell, .frc, .vel, .lammpstrj, .data).
    
    If .extxyz is used, all info is read from one file. If a folder is provided, it attempts
    to discover CP2K or LAMMPS output files automatically.
    
    Parameters
    ----------
    path : str
        Path to the .xyz, .extxyz, .lammpstrj file, or a directory.
    index : str, int, or slice
        Frame selection.
    timestep : float, optional
        Nominal time step between consecutive frames in **fs**.
    cell_path : str, optional
        Path to the .cell file (overrides discovery).
    frc_path : str, optional
        Path to the .frc file (usually frc-1.xyz, overrides discovery).
    vel_path : str, optional
        Path to the .vel file (usually vel-1.xyz, overrides discovery).
        
    Returns
    -------
    natoms : int
        Number of atoms per frame.
    nframes : int
        Number of selected frames.
    frames : list of Atoms
        The loaded frames.
    """
    from fishmol.atoms import Atoms

    if os.path.isdir(path):
        # 1. Directory discovery logic
        
        # Check for LAMMPS
        lammpstrj_files = glob.glob(os.path.join(path, "*.lammpstrj"))
        data_files = glob.glob(os.path.join(path, "*.data"))
        if lammpstrj_files and data_files:
            return read_lammps(
                lammpstrj_path=lammpstrj_files[0],
                data_path=data_files[0],
                index=index,
                timestep=timestep
            )
            
        # Check for CP2K
        xyz_files = glob.glob(os.path.join(path, "*.xyz"))
        # Exclude force and velocity files from being the primary trajectory
        primary_xyz = [f for f in xyz_files if not (f.endswith('frc-1.xyz') or f.endswith('vel-1.xyz'))]
        
        if primary_xyz:
            xyz_path = primary_xyz[0]
            if cell_path is None:
                cells = glob.glob(os.path.join(path, "*.cell"))
                if cells: cell_path = cells[0]
            if frc_path is None:
                frcs = glob.glob(os.path.join(path, "*frc-1.xyz"))
                if frcs: frc_path = frcs[0]
            if vel_path is None:
                vels = glob.glob(os.path.join(path, "*vel-1.xyz"))
                if vels: vel_path = vels[0]
        else:
            raise FileNotFoundError(f"No suitable trajectory files found in {path}")
    else:
        # It's a file
        xyz_path = path
        if xyz_path.endswith('.lammpstrj'):
            # If it's a lammpstrj file, we might need a .data file in the same dir
            base_dir = os.path.dirname(xyz_path)
            data_files = glob.glob(os.path.join(base_dir, "*.data"))
            return read_lammps(
                lammpstrj_path=xyz_path,
                data_path=data_files[0] if data_files else None,
                index=index,
                timestep=timestep
            )

    # 2. Standard XYZ / CP2K loading
    natoms, nframes, frames = read_extxyz(
        xyz_path, index=index, timestep=timestep
    )
    
    # Read cell if provided
    if cell_path:
        with open(cell_path, 'r') as f:
            lines = f.read().splitlines()
            # Filter out header lines (usually start with #)
            cell_lines = [ln for ln in lines if ln.strip() and not ln.strip().startswith('#')]
            
            # Assume one cell per frame or one cell for all
            from recordclass import make_dataclass
            Cell = make_dataclass("Cell", "lattice")
            
            for i, frame in enumerate(frames):
                if i < len(cell_lines):
                    parts = cell_lines[i].split()
                    # The .cell file has: Step, Time, Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz, Volume
                    # We need indices 2 to 10 (the 9 cell parameters)
                    if len(parts) >= 11:
                        cell_vals = parts[2:11]
                        cell_arr = np.array(cell_vals, dtype=float).reshape(3, 3)
                        frame.cell = Cell(cell_arr)
                    elif len(parts) == 3: # Fallback for simple 3-float orthogonal cell
                        frame.cell = Cell(np.diag(np.array(parts, dtype=float)))
                elif len(cell_lines) == 1: # One cell for all frames
                    parts = cell_lines[0].split()
                    if len(parts) >= 11:
                        cell_vals = parts[2:11]
                        frame.cell = Cell(np.array(cell_vals, dtype=float).reshape(3, 3))

    # Read forces if provided
    if frc_path:
        # Forces are usually in another XYZ-like file
        _, _, f_frames = read_extxyz(frc_path, index=index)
        for i, frame in enumerate(frames):
            if i < len(f_frames):
                frame.forces = f_frames[i].pos

    # Read velocities if provided
    if vel_path:
        _, _, v_frames = read_extxyz(vel_path, index=index)
        for i, frame in enumerate(frames):
            if i < len(v_frames):
                frame.velocities = v_frames[i].pos

    return natoms, nframes, frames


# ── LAMMPS Specific Loading ────────────────────────────────────────────────────

def read_lammps(
    lammpstrj_path: str,
    data_path: Optional[str] = None,
    index: Union[str, slice, int] = ':',
    timestep: Optional[float] = None
) -> Tuple[int, int, list]:
    """Read a LAMMPS trajectory from a .lammpstrj file.
    
    Element information is extracted from a corresponding .data file if provided.
    
    Parameters
    ----------
    lammpstrj_path : str
        Path to the LAMMPS trajectory file.
    data_path : str, optional
        Path to the LAMMPS data file for element mapping.
    index : str, int, or slice
        Frame selection.
    timestep : float, optional
        Nominal time step between consecutive frames in **fs**.
        
    Returns
    -------
    natoms : int
    nframes : int
    frames : list of Atoms
    """
    from fishmol.atoms import Atoms
    from fishmol.data import elements
    
    # 1. Parse .data file for element mapping
    type_to_symb = {}
    if data_path:
        type_to_symb = _parse_lammps_data(data_path)
    
    # 2. Parse .lammpstrj
    frames = []
    with open(lammpstrj_path, 'r') as f:
        # Split by ITEM: TIMESTEP to get frame blocks
        content = f.read().split('ITEM: TIMESTEP\n')[1:]
    
    all_indices = list(range(len(content)))
    if isinstance(index, int):
        indices = [all_indices[index]]
    elif isinstance(index, slice):
        indices = all_indices[index]
    else: # ':' or 'all'
        indices = all_indices

    from recordclass import make_dataclass
    Cell = make_dataclass("Cell", "lattice")

    natoms = 0
    for idx in indices:
        block = content[idx].splitlines()
        ts = float(block[0])
        num_atoms = int(block[2])
        natoms = num_atoms
        
        # Box bounds
        # ITEM: BOX BOUNDS xy xz yz pp pp pp
        box_header = block[3]
        is_triclinic = 'xy xz yz' in box_header
        
        bounds = [line.split() for line in block[4:7]]
        xlo, xhi = float(bounds[0][0]), float(bounds[0][1])
        ylo, yhi = float(bounds[1][0]), float(bounds[1][1])
        zlo, zhi = float(bounds[2][0]), float(bounds[2][1])
        
        if is_triclinic:
            xy = float(bounds[0][2])
            xz = float(bounds[1][2])
            yz = float(bounds[2][2])
            
            # LAMMPS triclinic cell vectors:
            lattice = np.array([
                [xhi - xlo, 0, 0],
                [xy, yhi - ylo, 0],
                [xz, yz, zhi - zlo]
            ])
        else:
            lattice = np.diag([xhi - xlo, yhi - ylo, zhi - zlo])
            
        # Atoms
        # ITEM: ATOMS id type x y z ...
        atom_header = block[7].split()
        id_col = atom_header.index('id') - 2
        type_col = atom_header.index('type') - 2
        x_col = atom_header.index('x') - 2
        y_col = atom_header.index('y') - 2
        z_col = atom_header.index('z') - 2
        
        atom_data = [line.split() for line in block[8:8+num_atoms]]
        # Sort by ID to ensure consistency
        atom_data.sort(key=lambda x: int(x[id_col]))
        
        symbs = []
        pos = []
        for d in atom_data:
            atype = int(d[type_col])
            symbs.append(type_to_symb.get(atype, f"X{atype}"))
            pos.append([float(d[x_col]), float(d[y_col]), float(d[z_col])])
            
        frames.append(Atoms(
            symbs=np.array(symbs, dtype='<U2'),
            pos=np.array(pos, dtype=float),
            cell=Cell(lattice),
            timestamp=ts * (timestep if timestep else 1.0)
        ))
        
    return natoms, len(frames), frames

def _parse_lammps_data(path: str) -> Dict[int, str]:
    """Try to extract atom type to element mapping from a LAMMPS data file.
    
    Uses ASE if available, otherwise falls back to parsing Masses and comments.
    """
    mapping = {}
    from fishmol.data import elements
    
    # Try using ASE if available
    try:
        from ase.io import read
        atoms = read(path, format='lammps-data')
        l_types = atoms.get_array('type')
        l_symbs = atoms.get_chemical_symbols()
        for t, s in zip(l_types, l_symbs):
            if t not in mapping:
                mapping[int(t)] = s
        if mapping: return mapping
    except ImportError:
        pass
    except Exception:
        pass

    # Fallback: manual parsing
    with open(path, 'r') as f:
        lines = f.readlines()
        
    in_masses = False
    for line in lines:
        if 'Masses' in line:
            in_masses = True
            continue
        if in_masses:
            if not line.strip(): 
                if mapping: break
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    tid = int(parts[0])
                    mass = float(parts[1])
                    symbol = "X"
                    min_diff = 1.0
                    for s, m in elements.items():
                        if abs(m - mass) < min_diff:
                            min_diff = abs(m - mass)
                            symbol = s
                    # Check for symbol in comment
                    if '#' in line:
                        comment = line.split('#')[1].strip().split()[0]
                        if comment in elements:
                            symbol = comment
                    mapping[tid] = symbol
                except ValueError:
                    in_masses = False
        if 'Atoms' in line: break

    return mapping


# ── Existing Extended XYZ Support ──────────────────────────────────────────────

def read_extxyz(
    path: str,
    index: Union[str, slice, int] = ':',
    timestep: Optional[float] = None,
    cell: Any = None,
) -> Tuple[int, int, list]:
    """Read an extended XYZ trajectory file.
    
    If the file is a standard .xyz or .extxyz file, it reads positions, symbols, 
    and potentially energy, timestamp, and cell from the comment line.
    
    Parameters
    ----------
    path : str
        Path to the ``.xyz`` / ``.extxyz`` file.
    index : str, int, or slice
        Frame selection.
    timestep : float, optional
        Nominal time step between consecutive frames in **fs**. Used only as a 
        fallback if timestamps are missing from the file.
    cell : array-like of shape (3, 3) or None
        Fallback cell.
        
    Returns
    -------
    natoms : int
        Number of atoms per frame.
    nframes : int
        Total number of frames.
    frames : list of Atoms
        Selected frames.
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

    try:
        natoms = int(raw[0].strip())
    except ValueError:
        raise ValueError(
            f"Expected atom count on line 1 of {path!r}, got: {raw[0]!r}")
    block = natoms + 2

    nframes_total = 0
    for i in range(0, len(raw), block):
        if i < len(raw) and raw[i].strip().lstrip('-').isdigit():
            nframes_total += 1

    if index in (':', 'all'):
        frame_ids = list(range(nframes_total))
    elif isinstance(index, int):
        n = index if index >= 0 else nframes_total + index
        if n < 0 or n >= nframes_total:
            raise IndexError(f"Frame index {index} out of range.")
        frame_ids = [n]
    elif isinstance(index, slice):
        start = index.start or 0
        stop = index.stop if index.stop is not None else nframes_total
        step = index.step or 1
        frame_ids = list(range(start, stop, step))
    else:
        frame_ids = list(index)

    if cell is not None:
        _fallback = dc_cell(np.asarray(cell, dtype=float))
    else:
        _fallback = None

    first_comment = raw[1] if len(raw) > 1 else ''
    sym_col, pos_slice = _col_indices(first_comment)

    frames: List[Atoms] = []
    for fi in frame_ids:
        base = fi * block
        if base >= len(raw): break
        comment = raw[base + 1]
        atom_lines = raw[base + 2: base + 2 + natoms]

        cell_dict = _parse_comment(comment)
        
        # Extract Energy and Timestamp from comment
        energy = cell_dict.get('energy') or cell_dict.get('e')
        try: energy = float(energy) if energy else None
        except ValueError: energy = None
                
        timestamp = cell_dict.get('time')
        try: timestamp = float(timestamp) if timestamp else None
        except ValueError: timestamp = None
        
        if timestamp is None:
            # Fallback if no timestamp in comment
            timestamp = (timestep if timestep is not None else 1.0) * fi

        # Cell: prefer per-frame Lattice=, then fallback, then None
        cell_arr = _extract_cell(cell_dict)
        if cell_arr is not None:
            frame_cell = dc_cell(cell_arr)
        elif _fallback is not None:
            frame_cell = _fallback
        else:
            frame_cell = dc_cell(None)

        symbs: List[str] = []
        positions: List[List[float]] = []
        for ln in atom_lines:
            cols = ln.split()
            if not cols: continue
            symbs.append(cols[sym_col])
            positions.append([float(v) for v in cols[pos_slice]])

        frames.append(
            Atoms(
                symbs=np.array(symbs, dtype='<U2'),
                pos=np.array(positions, dtype=float),
                cell=frame_cell,
                energy=energy,
                timestamp=timestamp,
            )
        )

    return natoms, len(frames), frames


def write_extxyz(
    frames: list,
    filename: str,
    natoms: int,
    timestep: float,
) -> None:
    """Write a list of :class:`~fishmol.atoms.Atoms` frames to extended XYZ.
    
    Writes all available information (cell, energy, timestamp, forces, velocities)
    to the extxyz format.
    
    Parameters
    ----------
    frames : list of :class:`~fishmol.atoms.Atoms`
        Frames to write (in order).
    filename : str
        Output file path.
    natoms : int
        Atoms per frame.
    timestep : float
        Effective timestep in **fs**.
    """
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
            
            # Build comment line with all available info
            comment_parts = []
            if lattice_str:
                comment_parts.append(f'Lattice="{lattice_str}"')
            
            # Properties string
            props = "species:S:1:pos:R:3"
            if frame.forces is not None:
                props += ":forces:R:3"
            if frame.velocities is not None:
                props += ":velocities:R:3"
            
            comment_parts.append(f'Properties={props}')
            
            # Metadata
            comment_parts.append(f'frame={i}')
            ts = frame.timestamp if frame.timestamp is not None else i * (timestep if timestep else 1.0)
            comment_parts.append(f'time={ts:.3f}')
            if frame.energy is not None:
                comment_parts.append(f'energy={frame.energy:.8f}')
                
            comment = " ".join(comment_parts)
            fh.write(comment + '\n')

            # Write atoms: symbol, pos, force, vel
            for j in range(natoms):
                sym = frame.symbs[j]
                pos = frame.pos[j]
                
                line = f"{sym:<3s} {pos[0]:>18.8f} {pos[1]:>18.8f} {pos[2]:>18.8f}"
                
                if frame.forces is not None:
                    f = frame.forces[j]
                    line += f" {f[0]:>18.8f} {f[1]:>18.8f} {f[2]:>18.8f}"
                
                if frame.velocities is not None:
                    v = frame.velocities[j]
                    line += f" {v[0]:>18.8f} {v[1]:>18.8f} {v[2]:>18.8f}"
                
                fh.write(line + '\n')
