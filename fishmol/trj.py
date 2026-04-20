import mmap
import numpy as np
import os
import warnings
from typing import List, Tuple, Union, Optional, Any, Sequence
from fishmol.atoms import Atoms

class Trajectory(object):
    """
    Trajectory object.
    -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    Attributes
    timestep : int, the time interval between two frames
    natoms: int, the number of atoms in each frame
    nframes: int, the number of frames in the trajectory
    frames: a list of frames in the trajectory. The frame is an Atoms object.
    
    Methods
    Selecting and slicing: select the specified 
    """
    def __init__(self, timestep: float, data: Optional[str] = None, natoms: Optional[int] = None, nframes: Optional[int] = None, frames: Optional[List[Atoms]] = None, index: Optional[Union[int, slice, str]] = None, cell: Any = None) -> None:
        self.timestep = timestep
        self.data = data
        if index is None:
            self.index = ":"
        else:
            self.index = index
        self.cell = cell
        if self.data is not None:
            self.natoms, self.nframes, self.frames = self.read(data)
        else:
            self.natoms = natoms
            self.nframes = nframes
            self.frames = frames
      
    def __len__(self):
        return len(self.frames)
    
    def __iter__(self):
        return iter(self.frames)
    
    def __getitem__(self, n: Union[int, slice, Sequence[int]]) -> 'Trajectory':
        """
        Enable slicing and selecting frames from a Trajectory, returns a new Trajectory object.
        """
        new_timestep = self.timestep
        if isinstance(n, int):
            if n < 0 : #Handle negative indices
                n += self.nframes
            if n < 0 or n >= self.nframes:
                warnings.warn(f"The index ({n}) is out of bounds (nframes={self.nframes}).")
                raise IndexError("The index (%d) is out of range."%n)
            frames = [self.frames[n],]
        elif isinstance(n, tuple) or isinstance(n, list):
            frames = [self.frames[x] for x in n]
        elif isinstance(n, slice):
            start = n.start
            stop = n.stop
            step = n.step
            if start is None:
                start = 0
            if stop is None:
                stop = self.nframes
            if step is None:
                step = 1
            frames = [self.frames[x] for x in range(start, stop, step)]
            new_timestep = self.timestep * step
        else:
            raise TypeError("Invalid argument type.")
        return self.__class__(timestep=new_timestep, natoms=self.natoms, nframes=len(frames), frames=frames, index=n, cell=self.cell)
    
    def read(self, data: str) -> tuple:
        """
        Fast read trajectory data using mmap.
        """
        with open(data) as f:
            mm = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
            frames = [line.strip().decode('utf8') for line in iter(mm.readline, b"")] # numpy datatype object, one element is string , the other is 3 floats
            mm.close()
        f.close()
        header = frames[0]
        natoms = int(header)
        prop = frames[1].split("=")
        dt = np.dtype([('symbol', np.str_, 2), ('position', np.float64, (3,))]) # numpy datatype object, element symbol is string , the position is an array of 3 floats
        # Store each line as a numpy array
        index = self.index
        nframes = frames.count(frames[0])
        if index == ":" or index == "all":
            step = 1
            frames = [np.array((line.split()[0], line.split()[-3:]), dtype=dt)
                      for line in frames if not (line.startswith(header) or line.startswith(prop[0]))]
        
        if isinstance(index, slice):
            start = index.start
            stop = index.stop
            step = index.step
            if start is None:
                start = 0
            if stop is None:
                stop = nframes
            if step is None:
                step = 1
            self.timestep *= step
            frames = frames[start*(natoms+2):stop*(natoms+2)]
            nframes = frames.count(header)
            frames = [np.array((line.split()[0], line.split()[-3:]), dtype=dt) 
                      for line in frames if not (line.startswith(header) or line.startswith(prop[0]))]
        # split the trajectory into frames
        frames = [frame2atoms(np.array(frames[x:x + natoms]), cell = self.cell) for x in range(0, len(frames), natoms)][::step]
        return natoms, nframes, frames
    
    def write(self, filename: Optional[str] = None) -> None:
        """
        Write trajectory into xyz file. Filtering atoms is supported by passing index, list of indices or slice object. If inverse_select, the atoms not in the select list will be write into xyz file.
        """
        # Filename
        if filename is None:
            filename = "trajectory.xyz"
        
        if os.path.exists(filename):
            while True:
                filename = "".join(filename.split(".")[:-1]) + "-1.xyz"
                if not os.path.exists(filename):
                    print(f"The filename already exists, file saved to {filename}")
                    break
        else:
            pass

        with open(filename, "a") as f:
            for i, frame in enumerate(self.frames):
                f.write(str(self.natoms) + f"\n Properties = frame: {i}, t: {i*self.timestep} fs, Cell: {self.cell}\n")
                np.savetxt(f, np.concatenate(((frame.symbs).reshape((self.natoms,1)), frame.pos), axis=1),
                           fmt="%-2s %s %s %s")
            f.close()
    
    def calib(self, save: bool = False, filename: Optional[str] = None) -> 'Trajectory':
        """
        Calibrate the trajectory by center of mass.
        """
        com_0 = self.frames[0].calc_com()
        if save:
            if filename is None:
                filename = self.data
                filename = "".join(filename.split(".")[:-1]) + "_calibrated.xyz"
            with open(filename, "a") as f:
                for i, frame in enumerate(self):
                    com = frame.calc_com()
                    shift = com_0 - com
                    frame.pos = shift + frame.pos
                    f.write(str(self.natoms) + f"\n Properties = frame: {i}, t: {i*self.timestep} fs, Cell: {self.cell}\n")
                    np.savetxt(f, np.concatenate(((frame.symbs).reshape((self.natoms,1)), frame.pos), axis=1),
                               delimiter=',', fmt = "%-2s %s %s %s")
            f.close()
            print(f"Calibrated trajectory saved to {filename}")
        
        else:
            for frame in self:
                com = frame.calc_com()
                shift = com_0 - com
                frame.pos = shift + frame.pos
        return self
    
    def wrap2box(self, center: Tuple[float, float, float] = (0.5, 0.5, 0.5), pretty_translation: bool = False, eps: float = 1e-7) -> 'Trajectory':
        for frame in self.frames:
            frame = frame.wrap_pos(center = center, pretty_translation = pretty_translation, eps = eps)
        return self
    
def frame2atoms(frame: np.ndarray, cell: Any = None, basis: str = 'Cartesian') -> Atoms:
    symbs = frame[:]["symbol"]
    pos = frame[:]["position"]
    atoms = Atoms(symbs = symbs, pos = pos, cell = cell, basis = basis)
    return atoms
