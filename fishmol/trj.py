"""
Trajectory object for MD post-processing.

Reads trajectory files via :mod:`fishmol.io`.  Supported formats:

* Extended XYZ (``.xyz``, ``.extxyz``) — per-frame ``Lattice=`` fields are
  parsed automatically, making both NVT and NPT trajectories transparent.
* LAMMPS dump (``.lammpstrj``, ``.dump``) — orthogonal and triclinic boxes,
  all coordinate styles (``x y z``, ``xs ys zs``, ``xu yu zu``), ``element``
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
    timestep : float, optional
        Nominal time step between consecutive stored frames in **fs**. 
        If omitted, attempts to auto-infer from trajectory frame stamps.
    data : str, optional
        Path to a trajectory file.  The format is detected automatically
        from the file extension:

        * ``.xyz`` / ``.extxyz`` → extended XYZ (via :func:`~fishmol.io.read_extxyz`)
        * ``.lammpstrj`` / ``.dump`` / ``.lammps`` → LAMMPS dump
          (via :func:`~fishmol.io.read_lammpstrj`)

        When supplied, *natoms*, *nframes*, and *frames* are derived from the
        file and must not be passed as well.
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
        Fallback cell (lattice vectors as rows, in Å).  For XYZ files, used
        when the file omits ``Lattice=`` (e.g. CP2K ``FORMAT XYZ``).  For
        LAMMPS files, overrides the per-frame box (rarely needed).
    type_map : dict or None, optional
        LAMMPS-only.  Mapping from integer atom type to element symbol, e.g.
        ``{1: 'O', 2: 'H'}``.  Required when the dump uses a ``type``
        column instead of an ``element`` column.

    Attributes
    ----------
    timestep : float
    natoms : int
    nframes : int
    frames : list of Atoms

    Examples
    --------
    Extended XYZ — NVT, cell supplied by caller:

    >>> cell = [[21.29, 0, 0], [-4.60, 20.79, 0], [-0.97, -1.21, 15.11]]
    >>> traj = Trajectory(timestep=5, data="nvt.xyz", cell=cell)

    Extended XYZ — NPT, cell read from each frame's ``Lattice=``:

    >>> traj = Trajectory(data="npt.xyz")
    >>> traj.frames[0].cell.lattice   # first-frame cell

    LAMMPS dump — orthogonal box with ``element`` column:

    >>> traj = Trajectory(timestep=2, data="dump.lammpstrj")

    LAMMPS dump — triclinic box with integer type column:

    >>> traj = Trajectory(
    ...     timestep=2,
    ...     data="dump.lammpstrj",
    ...     type_map={1: 'O', 2: 'H'},
    ... )
    """

    def __init__(
        self,
        timestep: Optional[float] = None,
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

            if ext in _LAMMPS_EXTS:
                self.natoms, self.nframes, self.frames, self.timestep = (
                    _io.read_lammpstrj(
                        path=self.data,
                        index=self.index,
                        timestep=self.timestep,
                        cell=cell,
                        type_map=type_map,
                    )
                )
            else:
                self.natoms, self.nframes, self.frames, self.timestep = (
                    _io.read_extxyz(
                        path=self.data,
                        index=self.index,
                        timestep=self.timestep,
                        cell=cell,
                    )
                )
        else:
            self.timestep = timestep or 1.0
            self.natoms = natoms
            self.nframes = nframes
            self.frames = frames

    # ── cell convenience property ─────────────────────────────────────────────

    @property
    def cell(self) -> Any:
        """Cell of the first frame (backward-compatible convenience accessor).

        For NVT trajectories this is the same for every frame.  For NPT
        trajectories each frame's ``.cell`` may differ; this property returns
        only the first frame's cell as a summary.

        Returns ``None`` when no frames are loaded or the first frame has no
        cell information.
        """
        if self.frames:
            fc = self.frames[0].cell
            if fc is not None:
                return getattr(fc, 'lattice', None)
        return None

    @property
    def is_npt(self) -> bool:
        """True when consecutive frames have detectably different cells."""
        if not self.frames or len(self.frames) < 2:
            return False
        c0 = getattr(self.frames[0].cell, 'lattice', None)
        c1 = getattr(self.frames[-1].cell, 'lattice', None)
        if c0 is None or c1 is None:
            return False
        return not np.allclose(np.asarray(c0), np.asarray(c1), atol=1e-5)

    # ── sequence protocol ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.frames)

    def __iter__(self):
        return iter(self.frames)

    def __getitem__(self, n: Union[int, slice, Sequence[int]]) -> 'Trajectory':
        """Slice or index frames, returning a new Trajectory.

        The effective timestep is multiplied by the slice step so that
        physical time labels remain correct.
        """
        new_timestep = self.timestep

        if isinstance(n, int):
            if n < 0:
                n += self.nframes
            if n < 0 or n >= self.nframes:
                raise IndexError(
                    f"Frame index {n} is out of range (nframes={self.nframes})."
                )
            frames = [self.frames[n]]

        elif isinstance(n, (tuple, list)):
            frames = [self.frames[x] for x in n]

        elif isinstance(n, slice):
            start = n.start or 0
            stop = n.stop if n.stop is not None else self.nframes
            step = n.step or 1
            frames = [self.frames[x] for x in range(start, stop, step)]
            new_timestep = self.timestep * step

        else:
            raise TypeError(f"Invalid index type: {type(n).__name__!r}.")

        return self.__class__(
            timestep=new_timestep,
            natoms=self.natoms,
            nframes=len(frames),
            frames=frames,
            index=n,
            cell=self._cell_fallback,
        )

    # ── I/O ──────────────────────────────────────────────────────────────────

    def write(self, filename: Optional[str] = None) -> None:
        """Write the trajectory to an extended XYZ file.

        Each frame's cell is encoded as ``Lattice=`` in the comment line so
        that NPT trajectories round-trip correctly through
        :func:`~fishmol.io.read_extxyz`.

        Parameters
        ----------
        filename : str, optional
            Output path.  Defaults to ``"trajectory.xyz"``.  If the file
            already exists a numeric suffix is appended automatically.
        """
        if filename is None:
            filename = "trajectory.xyz"
        _io.write_extxyz(
            frames=self.frames,
            filename=filename,
            natoms=self.natoms,
            timestep=self.timestep,
        )

    # ── trajectory operations ─────────────────────────────────────────────────

    def calib(
        self,
        save: bool = False,
        filename: Optional[str] = None,
    ) -> 'Trajectory':
        """Remove centre-of-mass drift, aligning all frames to frame 0.

        Parameters
        ----------
        save : bool
            If True, also write the calibrated trajectory to *filename*.
        filename : str, optional
            Output path when *save* is True.  Defaults to the source filename
            with ``_calibrated`` appended before the extension.

        Returns
        -------
        self
            Returns ``self`` to allow method chaining.
        """
        com_0 = self.frames[0].calc_com()

        if save:
            if filename is None and self.data is not None:
                base, ext = self.data.rsplit('.', 1)
                filename = f"{base}_calibrated.{ext}"
            elif filename is None:
                filename = "trajectory_calibrated.xyz"

            for frame in self:
                shift = com_0 - frame.calc_com()
                frame.pos = frame.pos + shift

            _io.write_extxyz(
                frames=self.frames,
                filename=filename,
                natoms=self.natoms,
                timestep=self.timestep,
            )
            print(f"Calibrated trajectory written to {filename!r}")
        else:
            for frame in self:
                shift = com_0 - frame.calc_com()
                frame.pos = frame.pos + shift

        return self

    def wrap2box(
        self,
        center: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        pretty_translation: bool = False,
        eps: float = 1e-7,
    ) -> 'Trajectory':
        """Wrap all atomic positions into the primary unit cell.

        For NPT trajectories each frame is wrapped with its own cell.

        Parameters
        ----------
        center : tuple of float
            Fractional centre of the wrapping target box (default: box centre).
        pretty_translation : bool
            If True, translate positions to minimise fractional coordinates
            (useful for visualisation).
        eps : float
            Small shift applied to the wrapping boundary.

        Returns
        -------
        self
        """
        for frame in self.frames:
            frame.wrap_pos(
                center=center,
                pretty_translation=pretty_translation,
                eps=eps,
            )
        return self


# ── Module-level helper ────────────────────────────────────────────────────────

def frame2atoms(
    frame,
    cell=None,
    basis: str = 'Cartesian',
) -> Atoms:
    """Convert a structured-array frame or an existing :class:`~fishmol.atoms.Atoms` to Atoms.

    This helper exists for backward compatibility with analysis code that calls
    ``frame2atoms`` directly.  The *cell* argument is now optional: when omitted
    and *frame* is already an :class:`~fishmol.atoms.Atoms` instance, the
    frame's own per-frame cell is preserved automatically.  This is the correct
    behaviour for NPT trajectories where each frame carries a different cell.

    Parameters
    ----------
    frame : structured ndarray or :class:`~fishmol.atoms.Atoms`
        Either a NumPy structured array with ``'symbol'`` and ``'position'``
        fields (legacy format produced by the old mmap reader) or an
        :class:`~fishmol.atoms.Atoms` object.
    cell : array-like of shape (3, 3) or cell dataobject or None, optional
        Explicit cell override.  When *None* and *frame* is an
        :class:`~fishmol.atoms.Atoms` instance, the frame's existing cell is
        used unchanged.  When *None* and *frame* is a raw structured array,
        the returned :class:`~fishmol.atoms.Atoms` has no cell set.
    basis : str, optional
        Coordinate system of the positions.  Default: ``'Cartesian'``.

    Returns
    -------
    :class:`~fishmol.atoms.Atoms`
        An Atoms object carrying the resolved cell.

    Examples
    --------
    >>> # NPT: each frame has its own cell — no cell argument needed
    >>> atoms = frame2atoms(traj.frames[i])
    >>> atoms.cell.lattice   # correct per-frame cell

    >>> # Legacy: build from a raw structured array
    >>> atoms = frame2atoms(raw_array, cell=my_cell)
    """
    if isinstance(frame, Atoms):
        if cell is not None:
            # Caller explicitly requests a different cell — build a new Atoms
            return Atoms(symbs=frame.symbs, pos=frame.pos, cell=cell, basis=basis)
        # Preserve the frame's own cell (handles NPT correctly)
        return frame

    # Legacy path: numpy structured array with 'symbol' and 'position' fields
    symbs = frame[:]["symbol"]
    pos = frame[:]["position"]
    return Atoms(symbs=symbs, pos=pos, cell=cell, basis=basis)