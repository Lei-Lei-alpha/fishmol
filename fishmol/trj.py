"""
Trajectory object for MD post-processing.

Reads trajectory files via :mod:`fishmol.io`.  Supported formats:

* Extended XYZ (``.xyz``, ``.extxyz``) — per-frame ``Lattice=`` fields are
  parsed automatically, making both NVT and NPT trajectories transparent.
* LAMMPS dump (``.lammpstrj``, ``.dump``) — orthogonal and triclinic boxes,
  all coordinate styles (``.x y z``, ``.xs ys zs``, ``.xu yu zu``), ``element``
  column or integer ``type`` column (requires *type_map*).

Each :class:`~fishmol.atoms.Atoms` frame carries its own ``.cell``.
"""

import numpy as np
import warnings
from typing import Any, List, Optional, Sequence, Tuple, Union

from fishmol.atoms import Atoms
from fishmol import io as _io


_LAMMPS_EXTS = frozenset({'.lammpstrj', '.dump', '.lammps'})
_XYZ_EXTS    = frozenset({'.xyz', '.extxyz'})


class Trajectory:
    """An ordered sequence of :class:`~fishmol.atoms.Atoms` frames.

    Parameters
    ----------
    timestep : float
        Nominal time step between consecutive stored frames in **fs**.
    data : str, optional
        Path to a trajectory file.  The format is detected automatically
        from the file extension.
    natoms : int, optional
        Number of atoms per frame — required when *data* is not given.
    nframes : int, optional
        Number of frames — required when *data* is not given.
    frames : list of :class:`~fishmol.atoms.Atoms`, optional
        Pre-built frame list — required when *data* is not given.
    index : str or int or slice, optional
        Frame selection passed to the reader.  ``':'`` (default) reads every
        frame.
    cell : array-like of shape (3, 3) or None, optional
        Fallback cell (lattice vectors as rows, in Å).
    type_map : dict or None, optional
        LAMMPS-only. Mapping from integer atom type to element symbol.
    """

    def __init__(
        self,
        timestep: float,
        data: Optional[str] = None,
        natoms: Optional[int] = None,
        nframes: Optional[int] = None,
        frames: Optional[List[Atoms]] = None,
        index: Optional[Union[int, slice, str]] = None,
        cell: Any = None,
        type_map: Optional[dict] = None,
    ) -> None:
        self.timestep = timestep
        self.data = data
        self.index = index if index is not None else ':'
        self._cell_fallback = cell
        self._type_map = type_map

        if self.data is not None:
            import os as _os
            _, ext = _os.path.splitext(self.data.lower())
            reader = _io.read_lammpstrj if ext in _LAMMPS_EXTS else _io.read_extxyz
            
            self.natoms, self.nframes, self.frames, self.timestep = reader(
                path=self.data, index=self.index, timestep=self.timestep, cell=cell,
                **({"type_map": type_map} if reader == _io.read_lammpstrj else {})
            )
        else:
            self.natoms, self.nframes, self.frames = natoms, nframes, (frames or [])

    @property
    def cell(self) -> Optional[np.ndarray]:
        """Cell of the first frame (backward-compatible convenience accessor)."""
        if self.frames:
            fc = self.frames[0].cell
            return getattr(fc, 'lattice', None)
        return None

    @property
    def is_npt(self) -> bool:
        """True when consecutive frames have detectably different cells."""
        if len(self.frames) < 2: return False
        c0, c1 = self.cell, getattr(self.frames[-1].cell, 'lattice', None)
        return c0 is not None and c1 is not None and not np.allclose(c0, c1, atol=1e-5)

    def __len__(self) -> int: return len(self.frames)
    def __iter__(self): return iter(self.frames)

    def __getitem__(self, n: Union[int, slice, Sequence[int]]) -> Union[Atoms, 'Trajectory']:
        """Slice or index frames, returning a new Trajectory or a single Atoms frame."""
        if isinstance(n, int): return self.frames[n]
        
        frames = [self.frames[i] for i in range(*n.indices(self.nframes))] if isinstance(n, slice) else [self.frames[i] for i in n]
        new_ts = self.timestep * (n.step or 1) if isinstance(n, slice) else self.timestep
        return Trajectory(timestep=new_ts, natoms=self.natoms, nframes=len(frames), frames=frames, cell=self._cell_fallback)

    def write(self, filename: str = "trajectory.xyz") -> None:
        """Write the trajectory to an extended XYZ file."""
        _io.write_extxyz(frames=self.frames, filename=filename, natoms=self.natoms, timestep=self.timestep)

    def calib(self, save: bool = False, filename: Optional[str] = None) -> 'Trajectory':
        """Remove centre-of-mass drift, aligning all frames to frame 0."""
        com_0 = self.frames[0].calc_com()
        for frame in self.frames: frame.pos += (com_0 - frame.calc_com())
        if save: self.write(filename or "trajectory_calibrated.xyz")
        return self

    def wrap2box(self, center=(0.5, 0.5, 0.5), pretty_translation=False, eps=1e-7) -> 'Trajectory':
        """Wrap all atomic positions into the primary unit cell."""
        for frame in self.frames: frame.wrap_pos(center=center, pretty_translation=pretty_translation, eps=eps)
        return self