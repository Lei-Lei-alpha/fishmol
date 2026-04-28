import numpy as np
import itertools
from recordclass import make_dataclass, dataobject
from typing import Dict, List, Tuple, Union, Optional, Any, Sequence
from fishmol.data import elements
from fishmol.utils import make_comb, cart2xys, xys2cart, translate_pretty

Cell = make_dataclass("Cell", 'lattice')

class Atom(np.ndarray):
    """
    Array representing a single atom symbol and its coordinates.

    Parameters
    ----------
    symb : str
        Chemical symbol of the atom.
    pos : array-like
        (3,) coordinates in Å or fractional units.
    cell : Cell or array-like, optional
        Unit cell associated with the coordinates.
    basis : {'Cartesian', 'Crystal'}, optional
        Coordinate system. Default: 'Cartesian'.
    pbc : tuple of bool, optional
        Periodic boundary conditions. Default: (True, True, True).
    """
    def __new__(cls, symb: str, pos: Union[List[float], np.ndarray], cell: Any = None, basis: str = 'Cartesian', pbc: Tuple[int, int, int] = (1,1,1)) -> 'Atom':
        dc_cell = make_dataclass("Cell", 'lattice')
        crys_names = ['Crystal', 'Crys', 'Cr', 'S']
        cart_names = ['Cartesian', 'Cart', 'Ca', 'R']
        coord_names = make_dataclass("Coord_names", 'crys_names cart_names')
        dt = np.dtype([('symb', np.str_, 2), ('pos', np.float64, (3,))])
        obj = np.asarray([(symb, pos)], dtype=dt).view(cls)
        obj._symb = symb
        obj._pos = np.asarray(pos, dtype=float)
        obj._basis = basis
        obj._coord_names = coord_names(crys_names, cart_names)
        if isinstance(cell, dataobject):
            obj._cell = cell
        else:
            obj._cell = dc_cell(np.asarray(cell))
        obj._pbc = pbc
        return obj

    def __array_finalize__(self, obj: Any) -> None:
        if obj is None: return
        self._cell = getattr(obj, '_cell', None)
        self._basis = getattr(obj, '_basis', None)
        self._pbc = getattr(obj, '_pbc', (1,1,1))

    @property
    def symb(self): return self._symb
    @property
    def pos(self): return self._pos
    @property
    def cell(self): return self._cell
    @property
    def basis(self): return self._basis
    @property
    def coord_names(self): return self._coord_names
    @property
    def pbc(self): return self._pbc

    def to_cart(self) -> 'Atom':
        """Ensure the atom coordinates are in Cartesian (Å) basis."""
        if self.basis in self.coord_names.cart_names:
            return self
        pos = xys2cart(self.pos, self.cell)
        return Atom(symb=self.symb, pos=pos, cell=self.cell, basis='Cartesian', pbc=self.pbc)

    def to_crys(self) -> 'Atom':
        """Ensure the atom coordinates are in Crystal (fractional) basis."""
        if self.basis in self.coord_names.crys_names:
            return self
        pos = cart2xys(self.pos, self.cell)
        return Atom(symb=self.symb, pos=pos, cell=self.cell, basis='Crystal', pbc=self.pbc)

    def vec(self, other: 'Atom', mic: bool = False) -> np.ndarray:
        """Calculate the vector connecting this atom to another."""
        if mic:
            a2b = other.to_crys().pos - self.to_crys().pos
            a2b -= np.round(a2b)
            if self.basis in self.coord_names.cart_names:
                return xys2cart(a2b, self.cell)
            return a2b
        else:
            a2b = other.pos - self.pos
            return a2b
            
    def dist(self, other: 'Atom', mic: bool = False) -> float:
        """Calculate the distance to another atom in Å."""
        v = self.vec(other, mic=mic)
        if self.basis in self.coord_names.crys_names:
            v = xys2cart(v, self.cell)
        return np.linalg.norm(v)


class Atoms(np.ndarray):
    """
    Array representing a collection of atoms and their properties.

    The Atoms object stores symbols, positions, forces, and velocities in a 
    structured array for efficient access. It handles coordinate basis 
    conversions and Minimum Image Convention (MIC) calculations automatically.
    """
    def __new__(cls, symbs: Union[List[str], str, np.ndarray], pos: Union[List[List[float]], np.ndarray], cell: Any = None, basis: str ='Cartesian', pbc: Tuple[int, int, int] = (1,1,1), forces: Optional[np.ndarray] = None, velocities: Optional[np.ndarray] = None, energy: Optional[float] = None, timestamp: Optional[float] = None) -> 'Atoms':
        dt = np.dtype([('symbol', np.str_, 2), ('position', np.float64, (3,)), ('force', np.float64, (3,)), ('velocity', np.float64, (3,))])
        dc_cell = make_dataclass("Cell", 'lattice')
        crys_names = ['Crystal', 'Crys', 'Cr', 'S']
        cart_names = ['Cartesian', 'Cart', 'Ca', 'R']
        coord_names = make_dataclass("Coord_names", 'crys_names cart_names')
        
        if isinstance(symbs, str):
            # Parse chemical formula: e.g. "H2O" -> ["H", "H", "O"]
            a = []
            for i, letter in enumerate(symbs):
                if letter.isdigit():
                    a += [a[-1]] * (int(letter) - 1)
                else:
                    a += [letter]
            symbs = a
        
        symbs = np.array(symbs, dtype = "<U2")
        pos =  np.array(pos, dtype = np.float64)
        n = len(symbs)
        
        data_list = np.zeros(n, dtype=dt)
        data_list['symbol'] = symbs
        data_list['position'] = pos
        if forces is not None: data_list['force'] = forces
        if velocities is not None: data_list['velocity'] = velocities
            
        obj = np.asarray(data_list, dtype=dt).view(cls)
        obj._symbs = symbs
        obj._pos = pos
        obj._forces = forces
        obj._velocities = velocities
        obj._energy = energy
        obj._timestamp = timestamp
        obj._basis = basis
        obj._coord_names = coord_names(crys_names, cart_names)
        if isinstance(cell, dataobject):
            obj._cell = cell
        else:
            obj._cell = dc_cell(np.asarray(cell))
        obj._pbc = pbc
        return obj

    def __array_finalize__(self, obj: Any) -> None:
        if obj is None: return
        self._cell = getattr(obj, '_cell', None)
        self._basis = getattr(obj, '_basis', None)
        self._pbc = getattr(obj, '_pbc', (1,1,1))

    @property
    def symbs(self): return self._symbs
    @property
    def pos(self): return self._pos
    @pos.setter
    def pos(self, val): self._pos = np.asarray(val)
    @property
    def forces(self): return self._forces
    @forces.setter
    def forces(self, value): self._forces = value
    @property
    def velocities(self): return self._velocities
    @velocities.setter
    def velocities(self, value): self._velocities = value
    @property
    def energy(self): return self._energy
    @energy.setter
    def energy(self, value): self._energy = value
    @property
    def timestamp(self): return self._timestamp
    @timestamp.setter
    def timestamp(self, value): self._timestamp = value
    @property
    def cell(self): return self._cell
    @cell.setter
    def cell(self, val): self._cell = val
    @property
    def basis(self): return self._basis
    @property
    def coord_names(self): return self._coord_names
    @property
    def pbc(self): return self._pbc
        
    def __len__(self) -> int:
        return len(self.symbs)
    
    def __getitem__(self, n: Union[int, str, Tuple, List, slice]) -> Union['Atom', 'Atoms']:
        if isinstance(n, int):
            if abs(n) >= len(self): raise IndexError(f"Index {n} out of range.")
            idx = n if n >= 0 else len(self) + n
            return Atom(symb=self.symbs[idx], pos=self.pos[idx], cell=self.cell, basis=self.basis, pbc=self.pbc)
            
        elif isinstance(n, str):
            select = [x for x in range(len(self)) if self.symbs[x] == n]
        elif isinstance(n, (tuple, list)):
            if all(isinstance(x, str) for x in n):
                select = [x for x in range(len(self)) if self.symbs[x] in n]
            elif all(isinstance(x, (int, np.integer)) for x in n):
                select = n
            else:
                raise TypeError("Indices must be all int or all str.")
        elif isinstance(n, slice):
            select = list(range(*n.indices(len(self))))
        else:
            raise TypeError("Invalid index type.")
        
        return self.__class__(symbs=self.symbs[select], pos=self.pos[select], cell=self.cell, basis=self.basis, pbc=self.pbc, 
                             forces=self.forces[select] if self.forces is not None else None,
                             velocities=self.velocities[select] if self.velocities is not None else None)

    def to_cart(self) -> 'Atoms':
        """Ensure coordinates are in Cartesian (Å)."""
        if self.basis in self.coord_names.cart_names: return self
        new_pos = xys2cart(self.pos, self.cell)
        return self.__class__(symbs=self.symbs, pos=new_pos, cell=self.cell, basis='Cartesian', pbc=self.pbc, forces=self.forces, velocities=self.velocities, energy=self.energy, timestamp=self.timestamp)

    def to_crys(self) -> 'Atoms':
        """Ensure coordinates are in Crystal (fractional)."""
        if self.basis in self.coord_names.crys_names: return self
        new_pos = cart2xys(self.pos, self.cell)
        return self.__class__(symbs=self.symbs, pos=new_pos, cell=self.cell, basis='Crystal', pbc=self.pbc, forces=self.forces, velocities=self.velocities, energy=self.energy, timestamp=self.timestamp)

    def vec(self, a: int, b: int, mic: bool = False) -> np.ndarray:
        """Vector connecting atom a to atom b."""
        if mic:
            diff = self.pos[b] - self.pos[a]
            if self.basis in self.coord_names.cart_names:
                diff = cart2xys(diff, self.cell)
            diff -= np.round(diff)
            if self.basis in self.coord_names.cart_names:
                return xys2cart(diff, self.cell)
            return diff
        return self.pos[b] - self.pos[a]

    def vecs(self, a: Optional[Union[int, List[int]]] = None, b: Optional[Union[int, List[int]]] = None, combs: Optional[Any] = None, normalise: bool = False, mic: bool = False) -> np.ndarray:
        """Vectorized calculation of multiple inter-atomic vectors."""
        if combs is None:
            combs = make_comb(a, b)
        combs = np.asarray(combs)
        
        idx_a, idx_b = combs[:, 0], combs[:, 1]
        diffs = self.pos[idx_b] - self.pos[idx_a]
        
        if mic:
            if self.basis in self.coord_names.cart_names:
                diffs = cart2xys(diffs, self.cell)
            diffs -= np.round(diffs)
            if self.basis in self.coord_names.cart_names:
                diffs = xys2cart(diffs, self.cell)
        
        if normalise:
            norms = np.linalg.norm(diffs, axis=1, keepdims=True)
            diffs = np.divide(diffs, norms, out=np.zeros_like(diffs), where=norms > 1e-12)
            
        return diffs

    def dist(self, a: int, b: int, mic: bool = False) -> float:
        """Distance between atom a and b in Å."""
        v = self.vec(a, b, mic=mic)
        if self.basis in self.coord_names.crys_names:
            v = xys2cart(v, self.cell)
        return np.linalg.norm(v)
    
    def dists(self, at_g1: Optional[List[int]] = None, at_g2: Optional[List[int]] = None, combs: Optional[Any] = None, cutoff: Optional[float] = None, mic: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorized calculation of distances between two atom groups."""
        if combs is None:
            pairs = np.asarray(make_comb(at_g1, at_g2))
        else:
            pairs = np.asarray(combs)
            
        if len(pairs) == 0: return np.array([]), np.array([])
        
        v = self.vecs(combs=pairs, mic=mic)
        if self.basis in self.coord_names.crys_names:
            v = xys2cart(v, self.cell)
        distances = np.linalg.norm(v, axis=1)

        if cutoff is not None:
            mask = distances <= cutoff
            return pairs[mask], distances[mask]
        return pairs, distances

    def angle(self, a: int, b: int, c: int, mic: bool = False) -> float:
        """Angle formed by atoms a-b-c in degrees (b is vertex)."""
        ba = self.vec(b, a, mic=mic)
        bc = self.vec(b, c, mic=mic)
        if self.basis in self.coord_names.crys_names:
            ba, bc = xys2cart(ba, self.cell), xys2cart(bc, self.cell)
        
        cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        return np.rad2deg(np.arccos(np.clip(cos, -1.0, 1.0)))

    def angles(self, *at_gs: Any, mic: bool = False) -> Tuple[List[Any], np.ndarray]:
        """Vectorized calculation of multiple bond angles."""
        if len(at_gs) == 1 and hasattr(at_gs[0], '__len__') and len(at_gs[0]) > 0 and len(at_gs[0][0]) == 3:
            pairs = np.asarray(at_gs[0])
        elif len(at_gs) == 3:
            pairs = np.asarray(list(itertools.product(at_gs[0], at_gs[1], at_gs[2])))
        else:
            raise ValueError("Invalid input for angles.")

        ba = self.vecs(combs=pairs[:, [1, 0]], mic=mic)
        bc = self.vecs(combs=pairs[:, [1, 2]], mic=mic)
        if self.basis in self.coord_names.crys_names:
            ba, bc = xys2cart(ba, self.cell), xys2cart(bc, self.cell)
        
        dots = np.einsum('ij,ij->i', ba, bc)
        norms = np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1)
        cos = np.divide(dots, norms, out=np.zeros_like(dots), where=norms > 1e-12)
        return pairs.tolist(), np.rad2deg(np.arccos(np.clip(cos, -1.0, 1.0)))

    def dihedral(self, idx: List[int], mic: bool = False) -> float:
        """Dihedral angle in degrees for 4-atom chain or 6-atom plane-plane."""
        if len(idx) == 4:
            v1, v2, v3 = self.vec(idx[0], idx[1], mic=mic), self.vec(idx[1], idx[2], mic=mic), self.vec(idx[2], idx[3], mic=mic)
            if self.basis in self.coord_names.crys_names:
                v1, v2, v3 = xys2cart(v1, self.cell), xys2cart(v2, self.cell), xys2cart(v3, self.cell)
            n1, n2 = np.cross(v1, v2), np.cross(v2, v3)
        elif len(idx) == 6:
            v1, v2 = self.vec(idx[0], idx[1], mic=mic), self.vec(idx[1], idx[2], mic=mic)
            v3, v4 = self.vec(idx[3], idx[4], mic=mic), self.vec(idx[4], idx[5], mic=mic)
            if self.basis in self.coord_names.crys_names:
                v1, v2, v3, v4 = [xys2cart(v, self.cell) for v in [v1, v2, v3, v4]]
            n1, n2 = np.cross(v1, v2), np.cross(v3, v4)
        else:
            raise ValueError("Dihedral requires 4 or 6 indices.")
            
        norm1, norm2 = np.linalg.norm(n1), np.linalg.norm(n2)
        if norm1 < 1e-12 or norm2 < 1e-12: return 0.0
        cos = np.dot(n1, n2) / (norm1 * norm2)
        return np.rad2deg(np.arccos(np.clip(cos, -1.0, 1.0)))

    def dihedrals(self, at_g: List[Any], mic: bool = False) -> Tuple[List[Any], np.ndarray]:
        """Vectorized calculation of multiple dihedral angles."""
        is_specific = len(at_g) > 0 and hasattr(at_g[0], '__len__') and len(at_g[0]) in [4, 6]
        if is_specific:
            pairs = np.asarray(at_g)
        else:
            pairs = np.asarray(list(itertools.product(*at_g)))

        if pairs.shape[1] == 4:
            v1 = self.vecs(combs=pairs[:, [0, 1]], mic=mic)
            v2 = self.vecs(combs=pairs[:, [1, 2]], mic=mic)
            v3 = self.vecs(combs=pairs[:, [2, 3]], mic=mic)
            if self.basis in self.coord_names.crys_names:
                v1, v2, v3 = xys2cart(v1, self.cell), xys2cart(v2, self.cell), xys2cart(v3, self.cell)
            n1, n2 = np.cross(v1, v2), np.cross(v2, v3)
        else:
            v1 = self.vecs(combs=pairs[:, [0, 1]], mic=mic)
            v2 = self.vecs(combs=pairs[:, [1, 2]], mic=mic)
            v3 = self.vecs(combs=pairs[:, [3, 4]], mic=mic)
            v4 = self.vecs(combs=pairs[:, [4, 5]], mic=mic)
            if self.basis in self.coord_names.crys_names:
                v1, v2, v3, v4 = [xys2cart(v, self.cell) for v in [v1, v2, v3, v4]]
            n1, n2 = np.cross(v1, v2), np.cross(v3, v4)

        dots = np.einsum('ij,ij->i', n1, n2)
        norms = np.linalg.norm(n1, axis=1) * np.linalg.norm(n2, axis=1)
        cos = np.divide(dots, norms, out=np.zeros_like(dots), where=norms > 1e-12)
        return pairs.tolist(), np.rad2deg(np.arccos(np.clip(cos, -1.0, 1.0)))

    def calc_com(self) -> np.ndarray:
        """Calculate the mass-weighted center of mass."""
        masses = np.array([elements[s] for s in self.symbs])
        return np.dot(masses, self.pos) / masses.sum()

    def wrap_pos(self, center: Tuple[float, float, float] = (0.5, 0.5, 0.5), pretty_translation: bool = False, eps: float = 1e-7) -> 'Atoms':
        """Wrap all atomic positions into the primary unit cell."""
        pbc = np.asarray(self.pbc, dtype=bool)
        shift = np.asarray(center) - 0.5 - eps
        shift[~pbc] = 0.0

        fractional = self.to_crys().pos - shift
        if pretty_translation:
            fractional = translate_pretty(fractional, pbc)
            fractional += (np.asarray(center) - 0.5)
        else:
            fractional[:, pbc] %= 1.0
            fractional[:, pbc] += shift[pbc]
        
        self.pos = xys2cart(fractional, self.cell)
        return self
