import warnings
from typing import List, Tuple, Union, Optional, Any, Sequence
from IPython.display import clear_output
import numpy as np
import fractions as f
import math
from scipy.spatial import Voronoi, voronoi_plot_2d, distance
from recordclass import make_dataclass, dataobject
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d, Axes3D
from itertools import combinations

def update_progress(progress: Union[float, int]) -> None:
  
    bar_length = 20
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
    if progress < 0:
        progress = 0
    if progress >= 1:
        progress = 1
        
    block = int(round(bar_length * progress))

    clear_output(wait = True)
    text = "Progress: [{0}] {1:.1f}%".format( "■" * block + "○" * (bar_length - block), progress * 100)
    print(text)


def to_sublists(lst: Sequence[Any], length: int = 2) -> List[Any]:
    """
    Split a list to sublists with specified length: e.g. a = [a,b,c,d,e]
    to_sublists(a) => [[a,b], [b,c], [c,d], [d,e]] 
    """
    return [lst[i:i+length] for i in range(len(lst)+1-length)]

def make_comb(a: Union[int, List[Any]], b: Union[int, List[Any]]) -> List[Any]:
    if all([isinstance (a, int), isinstance(b, int)]):
        comb = [(a, b)]
    elif any([isinstance (a, int), isinstance(b, int)]):
        try:
            comb = [(a, x) for x in b]
        except TypeError:
            comb = [(x, b) for x in a]
    elif all([isinstance (a, list), isinstance(b, list)]):
        if all([isinstance(a[0], int), isinstance(b[0], int)]):
            if a == b:
                comb = list(combinations(a, 2))
            else:
                comb = list(zip(a, b))
        elif any([isinstance(a[0], list), isinstance(b[0], list)]):
            try:
                comb = [list(zip(a, x)) for x in b]
            except TypeError:
                comb = [list(zip(x, b)) for x in a]
            comb = [val for sublist in comb for val in sublist]
        else:
            raise ValueError("Valid input of a and b: int, list, list of lists of int.")
    else:
        raise ValueError("Valid input of a and b: int, list, list of lists of int.")
    return comb

def cart2xys(pos: Union[np.ndarray, List[float]], cell: Any) -> np.ndarray:
    """
    Cartesian (absolute) position in angstrom to fractional position (scaled position in lattice).
    """
    pos = np.asarray(pos)
    bg = np.linalg.inv(cell.lattice)
    return np.dot(pos, bg)

def xys2cart(pos: Union[np.ndarray, List[float]], cell: Any) -> np.ndarray:
    """
    Fractional position (scaled position in lattice) to cartesian (absolute) position in angstrom.
    """
    pos = np.asarray(pos)
    return np.dot(pos, cell.lattice)

def translate_pretty(fractional: np.ndarray, pbc: Union[Sequence[bool], np.ndarray]) -> np.ndarray:
    """Translates atoms such that fractional positions are minimized."""

    for i in range(3):
        if not pbc[i]:
            continue

        indices = np.argsort(fractional[:, i])
        sp = fractional[indices, i]

        widths = (np.roll(sp, 1) - sp) % 1.0
        fractional[:, i] -= sp[np.argmin(widths)]
        fractional[:, i] %= 1.0
    return fractional
    
# def to_ase_atoms()


def retrieve_symbol(string: str) -> str:
    """function to remove numbers in a string, so that the atom dict keys can be converted to chemical symbols"""
    return ''.join([i for i in string if not i.isdigit()])

def mic_dist(pos1: np.ndarray, pos2: np.ndarray, cell: Any = None) -> Union[float, np.ndarray]:
    dc_cell = make_dataclass("Cell", 'lattice')
    if isinstance(cell, dataobject):
        pass
    else:
        cell = dc_cell(np.asarray(cell))
    a2b = pos2 - pos1
    a2b = cart2xys(a2b, cell)
    a2b -= np.round(a2b)
    a2b = xys2cart(a2b, cell)
    if a2b.ndim > 1:
        return np.linalg.norm(a2b, axis=1)
    return np.linalg.norm(a2b)

# Define functions to convert vectors between miller indices and cartesian coordinates
def get_gcd(ints: Sequence[int]) -> int:
    """
    Calculate the maximal common divisor of a list of integers
    """
    gcd = math.gcd(ints[0],ints[1])
    for i in range(2,len(ints)):
        gcd = math.gcd(gcd,ints[i])
    return gcd

class vector:
    """
    Vecotor object that can convert between miller indices and cartesian coordination (normalised).
    """
    def __init__(self, array: Sequence[float], cell: Any, name: str = "miller"):
        if len(array) != 3:
            raise Exception("The input array must has a length of exactly 3.")
        else:
            self.array = np.asarray(array)
        if name not in ["m", "miller", "c", "cartesian"]:
            raise Exception("Unrecognised name! Please use 'm' or 'miller' if the input array is a miller index, use 'c' or 'cartesian' if it is a coordinate.")
        else:
            self.name = name
        # if self.name == "m" or self.name == "miller":
        #     self.array = self.array.astype(int)
        self.cell = np.asarray(cell)
        
    def to_miller(self):
        if self.name == "m" or self.name == "miller":
            print("Already miller!")
        else:
            # obtain coord refer to the lattice vector
            if len(self.cell) == 3:
                pass
                
            elif len(self.cell) == 6:
                self.array = np.dot(self.array, self.cell.T)
                
            h = f.Fraction(self.array[0]).denominator
            k = f.Fraction(self.array[1]).denominator
            l = f.Fraction(self.array[2]).denominator
            
            self.array = self.array * h * k * l

            self.array = self.array.astype(int) // get_gcd(self.array.astype(int))
            self.name = "miller"
        # print(self.array)
        return self
        
    def to_cart(self, normalise = True):
        if self.name == "c" or self.name == "cartesian":
            print("Already cartesian!")
        else:
            self.array = np.dot(self.array.T, self.cell)
            if normalise:
                self.array = self.array / np.linalg.norm(self.array)
            self.name = "cartesian"
        # print(self.array)
        return self


class h_channel(Voronoi):
    def __init__(self, points, furthest_site=False, incremental=False, qhull_options=None):
        super().__init__(points, furthest_site, incremental, qhull_options)

    def sort_points(self, points: Sequence[Any]) -> List[int]:
        indices = [self.point_region[distance.cdist([point], self.points).argmin()] for point in points]
        return indices

def calc_freq(regions: Sequence[Any], timestep: Optional[float] = None) -> float:
    """
    Calculates the frequency of switching channels, unit times per ps
    """
    count = 0
    current_region = regions[0]
    freq = None
    
    if timestep is None:
        warnings.warn("No timestep specified, defaulting to 5 fs!")
        timestep = 5
        
    for i, region in enumerate(regions):
        if region != current_region:
            count += 1
            current_region = region
        else:
            pass
        
        freq = count * timestep / ((i + 1) * 1000)
        
    return freq


class Arrow3D(FancyArrowPatch):
  """Arrows in 3d anisotropy plot"""
  def __init__(self, xs, ys, zs, *args, **kwargs):
      super().__init__((0,0), (0,0), *args, **kwargs)
      self._verts3d = xs, ys, zs

  def do_3d_projection(self, renderer=None):
      xs3d, ys3d, zs3d = self._verts3d
      xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
      self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
      return np.min(zs)
  
  
  def get_basis(h_path: Any, cell: Any, miller: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Identifies two vectors that are perpendicular to the h_path,
    if h_path is in Cartesian coordinates rather than Miller indices, set miller = False"""
    # Create the vector object
    if miller:
        h_path = vector(h_path, cell = cell, name = "m")
        h_path = h_path.to_cart(normalise = True)
    else:
        h_path = vector(h_path, cell = cell, name = "c")
        h_path = h_path.to_cart(normalise = True)
    
    # Specify one vector that is orthogonal with the h_path
    indices = [0, 1, 2]
    idx = np.where(h_path.array != 0)[0][0]
    indices.remove(idx)
    basis_x = np.ones(3)
    basis_x[idx] = -(h_path.array[indices[0]] + h_path.array[indices[1]])/h_path.array[idx]
    basis_x = basis_x/np.linalg.norm(basis_x)
    
    # Calculate the other vector as the cross product
    basis_y = np.cross(h_path.array, basis_x)
    return basis_x, basis_y

def trans_coord(points: np.ndarray, cell: Any, W: np.ndarray, miller: bool = True) -> np.ndarray:
    """Change coordinate system of points from [[1,0,0],[0,1,0],[0,0,1]] to W"""
    # We need to get cartesion coordinates if the points are in Miller indices
    trans_pos = np.zeros(points.shape)
    if miller:
        for i, point in enumerate(points):
            point = vector(point, cell).to_cart(normalise = False).array
            trans_pos[i] = np.linalg.inv(W).dot(point)
    return trans_pos
