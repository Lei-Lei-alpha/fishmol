"""Analysis functions for MD trajectories.

Provides distribution functions (RDF, ADF, DDF, CDF), vector reorientation
dynamics (VRD), Van Hove correlation functions (VHCF), and scalar
distribution functions (SDF). All functions are designed to be statistically
sound and computationally efficient for large trajectory datasets.
"""

import warnings
import itertools
import numpy as np
from typing import Any, List, Optional, Tuple, Union, Dict
from recordclass import make_dataclass, asdict
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from fishmol.utils import to_sublists, make_comb, update_progress, cart2xys, xys2cart
from fishmol import style
from scipy.optimize import curve_fit
from scipy.special import gamma


# RDF
class RDF(object):
    """Radial distribution function g(r) between two atom groups.

    The RDF (or pair correlation function) describes how the density of particles 
    in at_g2 varies as a function of distance from a reference particle in at_g1. 
    It is a fundamental structural observable in statistical mechanics, defined as:

        g(r) = V / (N_A * N_B) * ⟨ Σ_i Σ_j δ(r - |r_j - r_i|) ⟩ / (4πr²)

    where V is the system volume and N_A, N_B are the number of atoms in the 
    respective groups. It is normalized such that g(r) → 1 at large r in a 
    uniform bulk liquid, indicating a loss of spatial correlation.

    Key Physical Metrics:
    - **Peak Positions (r_max)**: Define the coordination shells (1st neighbor, 2nd neighbor, etc.).
    - **Peak Heights (g_max)**: Indicate the strength/probability of finding a neighbor at that distance relative to a random gas.
    - **Coordination Number (CN)**: Obtained by integrating 4πr²ρ*g(r) up to the first minimum.

    Parameters
    ----------
    traj : Trajectory
        Trajectory object to analyse.
    at_g1 : list of int
        Indices of the reference atoms.
    at_g2 : list of int
        Indices of the neighbor atoms.
    nbins : int, optional
        Number of histogram bins (controls radial resolution). Default: 200.
    range : tuple of float, optional
        (r_min, r_max) in Å. If None, uses (0.0, 10.0). Default: None.

    Results dataclass fields (populated after calling calculate())
    -------------------------------------------------------------
    edges    : ndarray, shape (nbins+1,) — radial bin edges in Å.
    count    : ndarray, shape (nbins,) — accumulated raw distance histogram.
    x        : ndarray, shape (nbins,) — radial bin centres in Å.
    rdf      : ndarray, shape (nbins,) — g(r) values.
    t        : ndarray, shape (nframes,) — time axis in ps.
    pairs    : list of tuples — atom index pairs included in the calculation.
    scaler   : ndarray — per-frame pair distances.
    plot     : Figure — g(r) line plot (set when plot=True).
    com_plot : Figure — combined time-series + g(r) panel (set when com_plot=True).
    """

    def __init__(self, traj, at_g1, at_g2, nbins=200, range=None):
        if range is None: range = (0.0, 10.0)
        self.g1 = at_g1
        self.g2 = at_g2
        self.traj = traj
        settings = make_dataclass("Settings", "bins range")
        self.settings = settings(nbins, range)
        results = make_dataclass("Results", "edges count x rdf plot t pairs scaler com_plot")
        self.results = results()
        self.results.count, self.results.edges = np.histogram([-1], **asdict(self.settings))
        self.results.x = (self.results.edges[:-1] + self.results.edges[1:])/2
        self.results.t = np.linspace(0, self.traj.timestep * (self.traj.nframes - 1)/1000, self.traj.nframes)
        cell = np.asarray(self.traj.cell)
        self.volume = float(np.abs(np.linalg.det(cell)))
        
    def calculate(self, plot: bool = False, com_plot: bool = False, **kwargs: Any) -> Any:
        """Compute g(r) and optionally plot the result."""
        
        if com_plot:
            scaler_list = []
            for i, frame in enumerate(self.traj):
                _, dists = frame.dists(self.g1, self.g2, mic=True)
                scaler_list.append(dists)
                update_progress(i / self.traj.nframes)
            
            scaler = np.asarray(scaler_list)
            dists = scaler.flatten()
            if self.settings.range[1] is not None:
                dists = dists[dists <= self.settings.range[1]]
            self.results.scaler = scaler
            self.results.pairs = list(itertools.product(self.g1, self.g2))
        else:
            dists_list = []
            for i, frame in enumerate(self.traj):
                _, d = frame.dists(self.g1, self.g2, cutoff=self.settings.range[1], mic=True)
                dists_list.append(d)
                update_progress(i / self.traj.nframes)
            
            dists = np.concatenate(dists_list)
            self.results.scaler = dists
            self.results.pairs = list(itertools.product(self.g1, self.g2))

        count, _ = np.histogram(dists, **asdict(self.settings))
        self.results.count = count
        
        nA = len(self.g1)
        nB = len(self.g2)
        N_pairs = nA * nB
            
        vols = np.power(self.results.edges, 3)
        shell_vol = 4/3 * np.pi * np.diff(vols)
        
        density = nB / self.volume 
        
        self.results.rdf = self.results.count / (nA * self.traj.nframes * density * shell_vol)
        
        update_progress(1.0)
        
        if plot:
            fig, ax = plt.subplots(figsize = (4.2, 3.6))
            ax.plot(self.results.x, self.results.rdf, **kwargs)
            ax.set_xlabel(r"$r$ ($\AA$)")
            ax.set_ylabel(r"$g(r)$")
            plt.show()
            self.results.plot = fig
        
        if com_plot:
            fig = plt.figure(figsize=(4.8,3.6))
            ax = fig.add_axes([0.16, 0.16, 0.58, 0.75])
            [ax.plot(self.results.t, self.results.scaler[:,i], lw = 0.5) for i in range(self.results.scaler.shape[1])]
            ax.set_xlabel(r"$t$ (ps)")
            ax.set_ylabel(r"$r$ ($\AA$)")
            plt.legend(frameon = False, ncol = 2, bbox_to_anchor=(0.4, 0.9, 0.4, 0.5), loc='center', fontsize = "small")
            divider = make_axes_locatable(ax)
            ax_histy = divider.append_axes("right", 0.5, pad=0.05, sharey=ax)
            ax_histy.yaxis.tick_right()
            ax_histy.plot(self.results.rdf, self.results.x)
            ax_histy.set_xlabel(r"$g(r)$")
            plt.show()
            self.results.com_plot = fig
            
        return self.results

    def summary(self):
        """Print a scientific interpretation of the computed RDF."""
        print("="*75)
        print(" RADIAL DISTRIBUTION FUNCTION (RDF) SUMMARY")
        print("="*75)
        
        peaks_idx = []
        for i in range(1, len(self.results.rdf)-1):
            if self.results.rdf[i] > self.results.rdf[i-1] and self.results.rdf[i] > self.results.rdf[i+1]:
                if self.results.rdf[i] > 1.1: # Only report significant peaks
                    peaks_idx.append(i)
                    
        print(f" Analysis Scope : {len(self.g1)} reference atoms vs {len(self.g2)} target atoms")
        print(f" Bulk Density   : {len(self.g2) / self.volume:.4e} atoms / Å³")
        print("-" * 75)
        
        if not peaks_idx:
            print(" Structuring    : No significant coordination shells found (gas-like or highly dilute).")
        else:
            print(" STRUCTURAL PEAKS (Coordination Shells):")
            for i, p in enumerate(peaks_idx[:3]): # Report up to first 3 shells
                r_val = self.results.x[p]
                g_val = self.results.rdf[p]
                shell = ["1st", "2nd", "3rd"][i]
                print(f" {shell} Shell Peak: r = {r_val:.2f} Å  |  g(r) max = {g_val:.2f}")
                
        print("-" * 75)
        print(" INTERPRETATION GUIDE:")
        print(" - The first peak identifies the typical nearest-neighbor bond distance.")
        print(" - The depth of the first minimum (valley) indicates the stability of the shell.")
        print("   If the minimum reaches ~0, atoms rarely exchange between the 1st and 2nd shell.")
        print(" - If g(r) → 1 at large r, the system is a homogeneous bulk liquid.")
        print(" - If g(r) drops to 0 at large r, you are analyzing a finite cluster or droplet.")
        print("="*75)


# ADF
class ADF(object):
    """Angular distribution function g(α) for three-body angles.

    The ADF describes the probability distribution of the angle α formed by
    the atom triplet g1–g2–g3, where g2 is the central vertex. It reveals the
    local orientational order and geometric coordination around a central atom.

    When cone_correction=True (default), the raw histogram is divided by sin(α)
    to mathematically account for the increasing area of the solid-angle shell:

        g(α) = N_obs(α) / [ N_triplets * nframes * sin(α) ]

    Without the correction, a purely random gas of atoms will artificially peak
    at 90° simply because the solid-angle volume is largest at the equator.
    The sine-corrected g(α) integrates to a constant, allowing direct comparison
    between ordered structures (e.g. tetrahedral vs octahedral).

    Key Physical Metrics:
    - **Peak Positions**: Reveal the dominant coordination geometry (e.g. ~104.5° for water, ~109.5° for tetrahedral, ~90°/180° for octahedral).
    - **Peak Widths**: Broader peaks indicate higher thermal fluctuations or structural disorder in the molecular scaffold.

    Parameters
    ----------
    traj : Trajectory
        Trajectory object to analyse.
    at_g1 : list of int
        Atom indices of the first terminal group.
    at_g2 : list of int
        Atom indices of the central group (vertex).
    at_g3 : list of int
        Atom indices of the second terminal group.
    nbins : int, optional
        Number of histogram bins. Default: 200.
    range : tuple of float, optional
        (α_min, α_max) in degrees. Default: (0.0, 180.0).

    Results dataclass fields (populated after calling calculate())
    -------------------------------------------------------------
    x        : ndarray, shape (nbins,) — bin centre angles in degrees.
    adf      : ndarray, shape (nbins,) — g(α) values.
    edges    : ndarray, shape (nbins+1,) — bin edge angles in degrees.
    count    : ndarray, shape (nbins,) — accumulated raw angle histogram.
    pairs    : list — atom index triplets included in the calculation.
    scaler   : ndarray, shape (nframes, ntriplets) — per-frame angle values.
    t        : ndarray, shape (nframes,) — time axis in ps.
    plot     : Figure — g(α) line plot.
    com_plot : Figure — combined time-series + g(α) panel.
    """

    def __init__(self, traj, at_g1, at_g2, at_g3, nbins=200, range=(0.0, 180.0)):
        self.g1 = at_g1
        self.g2 = at_g2
        self.g3 = at_g3
        self.traj = traj
        settings = make_dataclass("Settings", "bins range")
        self.settings = settings(nbins, range)
        _, edges = np.histogram([-1], **asdict(self.settings))
        results = make_dataclass("Results", "edges count x adf plot t pairs scaler com_plot")
        self.results = results()
        self.results.edges = edges
        self.results.x = (edges[:-1] + edges[1:]) / 2
        self.results.count = np.zeros(self.settings.bins)
        self.results.t = np.linspace(0, self.traj.timestep * (self.traj.nframes - 1)/1000, self.traj.nframes)
        
    def calculate(self, cone_correction: bool = True, plot: bool = False, com_plot: bool = False, **kwargs: Any) -> Any:
        """Compute g(α) and optionally plot the result."""
        angles = []
        for i, frame in enumerate(self.traj):
            pairs, angle = frame.angles(self.g1, self.g2, self.g3, mic = True)
            count = np.histogram(angle, **asdict(self.settings))[0]
            self.results.count += count
            angles.append(angle)
            update_progress(i / self.traj.nframes)
        
        self.results.pairs = pairs
        self.results.scaler = np.asarray(angles)
        
        N = len(pairs)
        if cone_correction:
            sin_theta = np.maximum(np.sin(self.results.x * np.pi / 180), 1e-10)
            self.results.adf = self.results.count / (N * self.traj.nframes * sin_theta)
        else:
            self.results.adf = self.results.count / (N * self.traj.nframes)
        
        update_progress(1.0)
        
        if plot:
            fig, ax = plt.subplots(figsize = (4.2, 3.6))
            ax.plot(self.results.x, self.results.adf, **kwargs)
            ax.set_xlabel(r"$\alpha$ ($^{\circ}$)")
            ax.set_ylabel(r"$g(\alpha)$")
            plt.show()
            self.results.plot = fig
        
        if com_plot:
            fig = plt.figure(figsize=(4.8,3.6))
            ax = fig.add_axes([0.16, 0.16, 0.58, 0.75])
            [ax.plot(self.results.t, self.results.scaler[:,i], lw = 0.5) for i in range(self.results.scaler.shape[1])]
            ax.set_xlabel(r"$t$ (ps)")
            ax.set_ylabel(r"$\alpha$ ($^{\circ}$)")
            plt.legend(frameon = False, ncol = 2, bbox_to_anchor=(0.4, 0.9, 0.4, 0.5), loc='center', fontsize = "small")
            divider = make_axes_locatable(ax)
            ax_histy = divider.append_axes("right", 0.5, pad=0.05, sharey=ax)
            ax_histy.yaxis.tick_right()
            ax_histy.plot(self.results.adf, self.results.x)
            ax_histy.set_xlabel(r"$g(\alpha)$")
            plt.show()
            self.results.com_plot = fig
            
        return self.results

    def summary(self):
        """Print a scientific interpretation of the computed ADF."""
        print("="*75)
        print(" ANGULAR DISTRIBUTION FUNCTION (ADF) SUMMARY")
        print("="*75)
        
        peaks_idx = []
        for i in range(1, len(self.results.adf)-1):
            if self.results.adf[i] > self.results.adf[i-1] and self.results.adf[i] > self.results.adf[i+1]:
                if self.results.adf[i] > 0.05 * np.max(self.results.adf): # Noise filter
                    peaks_idx.append(i)
                    
        print(f" Analysis Scope : {len(self.results.pairs)} distinct triplets tracked.")
        print("-" * 75)
        
        if not peaks_idx:
            print(" Structuring    : No distinct angular preference observed.")
        else:
            print(" DOMINANT GEOMETRIES (Angular Peaks):")
            for i, p in enumerate(peaks_idx[:3]): # Report up to first 3 peaks
                ang_val = self.results.x[p]
                mag_val = self.results.adf[p]
                
                # Simple geometry guesser
                geo = "Unknown"
                if 100 < ang_val < 109: geo = "Bent / Water-like"
                elif 109 <= ang_val <= 110: geo = "Tetrahedral"
                elif 118 <= ang_val <= 122: geo = "Trigonal Planar"
                elif 88 <= ang_val <= 92: geo = "Octahedral / Square Planar"
                elif 170 <= ang_val <= 180: geo = "Linear"
                
                print(f" Peak {i+1}: α = {ang_val:.1f}° ({geo})  |  Intensity = {mag_val:.2e}")
                
        print("-" * 75)
        print(" INTERPRETATION GUIDE:")
        print(" - The solid-angle correction (1/sin(α)) has been applied.")
        print(" - Sharp, intense peaks imply a rigid, well-defined molecular framework.")
        print(" - Broad peaks indicate structural flexibility or high-temperature thermal bending.")
        print("="*75)


# DDF
class DDF(object):
    """Dihedral distribution function g(δ) for four-body torsions.

    The DDF computes the probability density of the dihedral angle δ. It reveals
    the conformational preferences (e.g., trans vs. gauche) of molecules or 
    the relative orientation between two molecular planes.

    The distribution is properly normalized such that it integrates to exactly 1
    over the range [-180°, 180°]:

        g(δ) = N_obs(δ) / [ N_dihedrals * nframes * Δδ ]

    where Δδ is the bin width.

    Key Physical Metrics:
    - **Trans State (δ ≈ 180° / -180°)**: Typically the lowest energy, most extended conformation for alkyl chains.
    - **Gauche States (δ ≈ ±60°)**: Higher energy coiled conformations.
    - **Planarity (δ ≈ 0°)**: Indicates strong π-conjugation or steric locking.

    Parameters
    ----------
    traj : Trajectory
        Trajectory object to analyse.
    at_g : list
        - Legacy mode: list of four lists of indices [g1, g2, g3, g4].
        - Specific mode: list of 4-index (atom chain) or 6-index (plane-plane) sublists.
    nbins : int, optional
        Number of histogram bins. Default: 200.
    range : tuple of float, optional
        (δ_min, δ_max) in degrees. Default: (-180.0, 180.0).

    Results dataclass fields
    ------------------------
    x        : ndarray — bin centre angles in degrees.
    ddf      : ndarray — g(δ) values.
    angles   : ndarray, shape (nframes, n_dihedrals) — per-frame dihedral values.
    """

    def __init__(self, traj, at_g, nbins=200, range=(-180.0, 180.0)):
        self.g = at_g
        self.traj = traj
        settings = make_dataclass("Settings", "bins range")
        self.settings = settings(nbins, range)
        _, edges = np.histogram([-1], **asdict(self.settings))
        results = make_dataclass("Results", "edges count x ddf plot t angles com_plot")
        self.results = results()
        self.results.edges = edges
        self.results.x = (edges[:-1] + edges[1:]) / 2
        self.results.count = np.zeros(self.settings.bins)
        self.results.t = np.linspace(0, self.traj.timestep * (self.traj.nframes - 1)/1000, self.traj.nframes)
        
    def calculate(self, plot: bool = False, com_plot: bool = False, **kwargs: Any) -> Any:
        """Compute g(δ) and optionally plot."""
        angles = []
        for i, frame in enumerate(self.traj):
            pairs, angle = frame.dihedrals(self.g, mic = True)
            count = np.histogram(angle, **asdict(self.settings))[0]
            self.results.count += count
            angles.append(angle)
            update_progress(i / self.traj.nframes)
        
        self.results.angles = np.asarray(angles)
        N = len(pairs)
        bin_width = self.results.edges[1] - self.results.edges[0]
        # Proper probability density normalization
        self.results.ddf = self.results.count / (N * self.traj.nframes * bin_width)
        
        update_progress(1.0)
        
        if plot:
            fig, ax = plt.subplots(figsize = (4.2, 3.6))
            ax.plot(self.results.x, self.results.ddf, **kwargs)
            ax.set_xlabel(r"$\delta$ ($^{\circ}$)")
            ax.set_ylabel(r"$g(\delta)$")
            plt.show()
            self.results.plot = fig
        
        if com_plot:
            fig = plt.figure(figsize=(4.8,3.6))
            ax = fig.add_axes([0.16, 0.16, 0.58, 0.75])
            [ax.plot(self.results.t, self.results.angles[:,i], lw = 0.5) for i in range(self.results.angles.shape[1])]
            ax.set_xlabel(r"$t$ (ps)")
            ax.set_ylabel(r"$\delta$ ($^{\circ}$)")
            divider = make_axes_locatable(ax)
            ax_histy = divider.append_axes("right", 0.5, pad=0.05, sharey=ax)
            ax_histy.yaxis.tick_right()
            ax_histy.plot(self.results.ddf, self.results.x)
            ax_histy.set_xlabel(r"$g(\delta)$")
            plt.show()
            self.results.com_plot = fig
            
        return self.results

    def summary(self):
        """Print a scientific interpretation of the computed DDF."""
        print("="*75)
        print(" DIHEDRAL DISTRIBUTION FUNCTION (DDF) SUMMARY")
        print("="*75)
        
        # Calculate circular mean
        rad_angles = np.deg2rad(self.results.angles)
        mean_cos = np.mean(np.cos(rad_angles))
        mean_sin = np.mean(np.sin(rad_angles))
        circ_mean = np.rad2deg(np.arctan2(mean_sin, mean_cos))
        
        print(f" Analysis Scope : {self.results.angles.shape[1]} distinct dihedrals tracked.")
        print(f" Circular Mean  : {circ_mean:.1f}°")
        print("-" * 75)
        print(" INTERPRETATION GUIDE:")
        print(" - δ ≈ 180° or -180° : Trans conformation (often the global energy minimum).")
        print(" - δ ≈ ±60°          : Gauche conformation (sterically hindered).")
        print(" - δ ≈ 0°            : Cis/Planar conformation (rare unless π-conjugated).")
        print(" - Rapidly jumping time-series in com_plot indicates low rotational barriers.")
        print("="*75)


# CDF
class CDF(object):
    """Combined (joint) distribution function correlating two scalar observables.

    The CDF reveals the 2D joint probability density of finding a system in a 
    state characterized by specific values of two observables simultaneously 
    (e.g., a specific bond distance AND a specific bond angle).

    Mathematically, it computes the normalized 2D histogram of the two input 
    arrays:

        P(x, y) = N_obs(x, y) / [ N_total * Δx * Δy ]

    Key Physical Applications:
    - **Hydrogen Bonding**: Correlating D-A distance with D-H...A angle to map the free energy surface of the H-bond.
    - **Conformational Analysis**: Ramachandran plots (φ vs. ψ dihedrals) to identify stable secondary structures in polymers or proteins.

    Parameters
    ----------
    scaler1, scaler2 : Results
        Results objects from RDF, ADF, or DDF calculations. The `.scaler` 
        attributes of these objects are used as the raw (x, y) data pairs.
    names : list of str, optional
        Display names for the marginal panels (e.g. ["Distance (Å)", "Angle (°)"]).
    range : tuple of lists, optional
        ([x_min, x_max], [y_min, y_max]). If None, ranges are determined automatically.

    Results dataclass fields
    ------------------------
    cdf      : ndarray, 2D — Normalized joint probability density.
    xedges, yedges : ndarrays — Bin boundaries.
    xcenters, ycenters : ndarrays — Not directly set in current implementation, relies on s1.x and s2.x.
    plot     : Figure — 2D contour plot with marginal 1D distributions.
    """
    def __init__(self, scaler1: Any, scaler2: Any, names: Optional[List[str]] = None, range: Optional[Tuple[List[float], List[float]]]  = None) -> None:
        self.s1 = scaler1
        self.s2 = scaler2
        self.names = names

        def get_dist(obj, name):
            if hasattr(obj, name): return getattr(obj, name)
            for attr in ['rdf', 'adf', 'ddf', 'cdf']:
                if hasattr(obj, attr): return getattr(obj, attr)
            raise AttributeError(f"Could not find distribution data in Results object.")

        if range is None:
            v1 = get_dist(self.s1, names[0] if names else "")
            v2 = get_dist(self.s2, names[1] if names else "")
            range = ([0, v1.max()],[0, v2.max()])
        
        settings = make_dataclass("Settings", "range")
        self.settings = settings(range)
        results = make_dataclass("Results", "xedges yedges xcenters ycenters x y cdf plot")
        self.results = results()
        
    def calculate(self, plot: bool = True, **kwargs: Any) -> Any:
        """Compute the 2D joint histogram and plot the results."""
        def get_dist(obj, name):
            if hasattr(obj, name): return getattr(obj, name)
            for attr in ['rdf', 'adf', 'ddf', 'cdf']:
                if hasattr(obj, attr): return getattr(obj, attr)
            return None

        self.results.xedges = self.s1.edges
        self.results.yedges = self.s2.edges
        x = self.s1.scaler.flatten()
        y = self.s2.scaler.flatten()

        # density=True normalizes the histogram so the integral over the range is 1
        H, xedges, yedges = np.histogram2d(x, y, bins=(self.results.xedges, self.results.yedges), density=True, **asdict(self.settings))
        self.results.cdf = H.T
        
        if plot:
            fig = plt.figure(figsize=(4.8, 4.0))
            ax  = fig.add_axes([0.16, 0.16, 0.82, 0.75])
            X, Y = np.meshgrid(self.s1.x, self.s2.x)
            levels = np.linspace(self.results.cdf.min(), self.results.cdf.max(), 50)
            CS = ax.contourf(X, Y, self.results.cdf, cmap=style.cdf_cmap, levels=levels)
            ax.set_xlabel(self.names[0] if self.names else "")
            ax.set_ylabel(self.names[1] if self.names else "")
            divider = make_axes_locatable(ax)
            ax_histx = divider.append_axes("top", 0.5, pad=0.05, sharex=ax)
            ax_histx.xaxis.tick_top()
            ax_histx.plot(self.s1.x, get_dist(self.s1, self.names[0] if self.names else ""))
            ax_histy = divider.append_axes("right", 0.5, pad=0.05, sharey=ax)
            ax_histy.yaxis.tick_right()
            ax_histy.plot(get_dist(self.s2, self.names[1] if self.names else ""), self.s2.x)
            plt.colorbar(CS, pad=0.035, location="right", ax=ax)
            plt.show()
            self.results.plot = fig
        return self.results

    def summary(self):
        """Print a scientific interpretation of the computed CDF."""
        print("="*75)
        print(" COMBINED DISTRIBUTION FUNCTION (CDF) SUMMARY")
        print("="*75)
        
        # Find global maximum
        max_idx = np.unravel_index(np.argmax(self.results.cdf), self.results.cdf.shape)
        max_x = self.s1.x[max_idx[1]] # Transposed geometry
        max_y = self.s2.x[max_idx[0]]
        max_density = self.results.cdf[max_idx]
        
        x_label = self.names[0] if self.names else "Variable 1"
        y_label = self.names[1] if self.names else "Variable 2"
        
        print(f" Correlating : '{x_label}' vs '{y_label}'")
        print("-" * 75)
        print(" MOST PROBABLE STATE (Global Maximum):")
        print(f" {x_label} = {max_x:.2f}")
        print(f" {y_label} = {max_y:.2f}")
        print(f" Peak Density = {max_density:.2e}")
        
        print("-" * 75)
        print(" INTERPRETATION GUIDE:")
        print(" - The contour map effectively displays the Free Energy Surface (FES).")
        print("   Higher density (red/warm regions) = Lower Free Energy = More Stable.")
        print(" - Diagonal streaks indicate strong coupling between the two variables.")
        print(" - Circular or isolated peaks indicate stable, independent conformational states.")
        print("="*75)


# VRD
class VRD(object):
    """Vector Reorientation Dynamics via Legendre polynomial autocorrelation.

    The VRD quantifies how quickly a molecular vector (e.g. an O–H bond, a C–C
    axis) loses memory of its initial orientation. It computes the time
    autocorrelation function of the l-th order Legendre polynomial:

        C_l(t) = ⟨ P_l( u(t_0) · u(t_0 + t) ) ⟩

    where u(t) is the unit vector at time t and the angle brackets denote an
    ensemble average over all vectors and all time origins t_0.

        P_1(x) = x                         (l=1) (Dielectric relaxation, IR)
        P_2(x) = (3x² − 1) / 2             (l=2) (NMR, Raman scattering)
        P_3(x) = (5x³ − 3x) / 2            (l=3)

    C_l(t) decays from 1 at t=0 toward 0 as the vector randomises. The decay is
    fitted with the Kohlrausch–Williams–Watts (KWW) stretched exponential:

        C(t) = exp[ −(t / τ)^β ]

    Parameters
    ----------
    traj : Trajectory, optional
        Trajectory object to analyse. Provide either ``traj`` + ``spec`` or
        a pre-computed ``spec`` array.
    spec : list of [list of int, list of int] or ndarray
        When ``traj`` is given: a two-element list ``[g1_indices, g2_indices]``
        defining the atom pairs whose inter-atomic vector is tracked.
        When ``traj`` is None: a 3D ndarray of shape (nframes, nvectors, 3) 
        containing the pre-computed vector time series.
    timestep : float, optional
        Trajectory timestep in fs. Required when ``traj`` is None; ignored
        otherwise (taken from ``traj.timestep``).
    num : int, optional
        Maximum number of frames for the lag time (window size). 
        t_max = timestep * (num − 1) / 1000 ps. Default: 2000.
    sampling : int, optional
        Frame stride within the correlation window. A value of 5 uses every 
        5th lag time. Default: 5.
    skip : int, optional
        Stride between time origins (t_0). Default: 10.

    Results dataclass fields
    ------------------------
    t          : ndarray, shape (num//sampling,) — lag time axis in ps.
    C_t        : ndarray — ensemble-averaged C_l(t) values.
    C_t_error  : ndarray — standard error of the mean across different origins.
    t_fit      : ndarray, shape (200,) — dense time axis for the fitted curve.
    C_t_fit    : ndarray — KWW fitted values on t_fit.
    fit_params : ndarray — fitted [τ, β] parameters.
    tau        : float — fitted relaxation time τ in ps.
    beta       : float — fitted stretching exponent β.
    tau_eff    : float — effective (integrated) relaxation time in ps.
    plot       : Figure — C_l(t) plot.
    """
    def __init__(self, traj: Any = None, spec: Any = None, timestep: Optional[float] = None, num: int = 2000, sampling: int = 5, skip: int = 10) -> None:
        results_cls = make_dataclass("Results", "t C_t C_t_error t_fit C_t_fit fit_params tau beta tau_eff plot l")
        self.results = results_cls()
        self.traj = traj
        self.spec = spec
        self.num = num
        self.sampling = sampling
        self.skip = skip
        self.t_step = traj.timestep if traj else (timestep if timestep else 5.0)
        self.lags = np.arange(0, self.num, self.sampling)
        self.results.t = self.lags * self.t_step / 1000.0
    
    def calculate(self, l: int = 2, fit: bool = False, plot: bool = True, log_scale: bool = False, **kwargs: Any) -> Any:
        """Compute C_l(t) using dense time origins."""
        self.results.l = l
        if self.traj is not None:
            is_specific = len(self.spec) > 0 and hasattr(self.spec[0], '__len__') and len(self.spec[0]) == 2 and isinstance(self.spec[0][0], (int, np.integer))
            combs = self.spec if is_specific else make_comb(*self.spec)
            vecs = np.stack([f.vecs(combs=combs, normalise=True, mic=True) for f in self.traj])
        else:
            vecs = np.asarray(self.spec, dtype=float)
            if vecs.ndim == 2: vecs = vecs[:, np.newaxis, :]
            vecs /= np.linalg.norm(vecs, axis=-1, keepdims=True)

        n_frames, n_vecs, _ = vecs.shape
        t0_indices = np.arange(0, n_frames - self.lags[-1], self.skip)
        res_per_origin = np.zeros((len(t0_indices), len(self.lags)))

        for i, t0 in enumerate(t0_indices):
            u0, ut = vecs[t0], vecs[t0 + self.lags]
            dots = np.sum(u0 * ut, axis=-1)
            if l == 1: pl = dots
            elif l == 2: pl = 0.5 * (3 * dots**2 - 1)
            elif l == 3: pl = 0.5 * (5 * dots**3 - 3 * dots)
            res_per_origin[i] = pl.mean(axis=1)
            update_progress(i / len(t0_indices))

        self.results.C_t = res_per_origin.mean(axis=0)
        self.results.C_t_error = res_per_origin.std(axis=0) / np.sqrt(len(t0_indices))

        if fit:
            def kww(t, tau, b): return np.exp(-np.power(np.maximum(t, 0) / tau, b))
            idx_e = np.where(self.results.C_t < 1/np.e)[0]
            tau0 = self.results.t[idx_e[0]] if len(idx_e) > 0 else self.results.t[-1]
            p, _ = curve_fit(kww, self.results.t, self.results.C_t, p0=[tau0, 0.5])
            self.results.fit_params, self.results.tau, self.results.beta = p, p[0], p[1]
            self.results.t_fit = np.linspace(self.results.t.min(), self.results.t.max(), 200)
            self.results.C_t_fit = kww(self.results.t_fit, *p)
            self.results.tau_eff = (p[0] / p[1]) * gamma(1.0 / p[1])
        
        update_progress(1.0)
        if plot:
            fig, ax = plt.subplots(figsize=(4.2, 3.6))
            ax.errorbar(self.results.t, self.results.C_t, yerr=self.results.C_t_error, fmt='o', ms=4, **kwargs)
            if fit: ax.plot(self.results.t_fit, self.results.C_t_fit, 'r-', lw=2)
            ax.set_xlabel("t (ps)"); ax.set_ylabel(f"$C_{l}(t)$")
            if log_scale: ax.set_yscale('log')
            plt.show(); self.results.plot = fig
        return self.results

    def summary(self):
        """Print a scientific interpretation of the computed VRD."""
        print("="*75)
        print(" VECTOR REORIENTATION DYNAMICS (VRD) SUMMARY")
        print("="*75)
        
        print(f" Legendre Polynomial Order (l) : {self.results.l}")
        
        if hasattr(self.results, 'tau'):
            print("-" * 75)
            print(" STRETCHED EXPONENTIAL (KWW) FIT RESULTS:")
            print(f" Relaxation Time (tau)  : {self.results.tau:.4f} ps")
            print(f" Stretching Exponent (β): {self.results.beta:.4f}")
            print(f" Effective Time (τ_eff) : {self.results.tau_eff:.4f} ps")
            
            print("\n FIT INTERPRETATION:")
            if self.results.beta > 0.95:
                print(" - β ≈ 1 : Simple Debye (exponential) relaxation. Rotational environment is uniform.")
            else:
                print(" - β < 1 : Stretched exponential relaxation. Rotational dynamics are highly")
                print("           heterogeneous (e.g. sub-ensembles of molecules rotating at different")
                print("           speeds due to complex caging or hydrogen bonding networks).")
        
        print("-" * 75)
        print(" GENERAL INTERPRETATION GUIDE:")
        print(" - The curve C(t) shows how fast molecules 'forget' their initial orientation.")
        print(" - An initial rapid drop (within < 0.2 ps) corresponds to 'librations'")
        print("   (frustrated rotations or wobbling within the local solvent cage).")
        print(" - The slower, long-time tail corresponds to true structural reorientation.")
        print(" - l=1 maps to Dielectric relaxation/IR spectra.")
        print(" - l=2 maps to NMR relaxation and Raman scattering.")
        print("="*75)


# VHCF
class VHCF(object):
    """Van Hove correlation function G(r, τ) and non-Gaussian parameter analysis.

    The Van Hove correlation function characterises both the spatial structure and
    the temporal dynamics of a system. Two complementary modes are available:

    **Self-part G_s(r, τ)** (``at_g2=None``):
        Tracks the same set of atoms over time. G_s(r, τ) is the probability that
        a tagged atom has displaced by a distance r in time τ. For purely Fickian
        diffusion in three dimensions, G_s is a Gaussian. Deviations indicate
        dynamical heterogeneity such as cage trapping, hopping, or glassy dynamics.

        .. warning::
            The self-part requires **unwrapped** (non-wrapped) coordinates. Do not
            call ``traj.wrap2box()`` before this analysis; use ``traj.calib()`` only.
            Wrapped coordinates cause atoms crossing box boundaries to appear to jump
            by ~L instead of a small thermal displacement, corrupting the MSD and
            the non-Gaussian parameter α₂.

    **Distinct-part G_d(r, τ)** (``at_g2`` provided):
        Tracks two different atom groups. G_d(r, τ) is the probability of finding an
        atom from g2 at distance r from an atom in g1 at time τ later. At τ = 0 it
        reduces to the radial distribution function g(r). Pair distances are computed
        under the minimum image convention. As τ → ∞ the correlations decay and
        G_d → 1.

    Additional observables computed alongside G(r, τ):

    - **Radial probability distribution** RPD(r, τ) = 4π r² G(r, τ): integrates to 1
      for the self-part and reveals the most probable displacement distance.
    - **Non-Gaussian parameter** α₂(τ) = 3⟨r⁴⟩ / (5⟨r²⟩²) − 1 (self-part only):
      zero for Gaussian diffusion; positive values signal heterogeneous dynamics.
      The time τ* at which α₂ peaks marks the characteristic cage-escape time.
    - **Mean squared displacement** ⟨Δr²⟩(τ) (self-part only): equivalent to the
      MSD computed from ``fishmol.msd``, shown on a log–log scale to identify
      ballistic (slope 2), caged (slope ≈ 0), and diffusive (slope 1) regimes.
    - **Intermediate scattering function** F(k, τ): the spatial Fourier transform
      of G(r, τ), directly related to neutron or X-ray scattering experiments.

    Parameters
    ----------
    traj : Trajectory
        Trajectory object to analyse.
    at_g1 : list of int
        Atom indices of the first group (the reference / tagged atoms).
    at_g2 : list of int or None, optional
        Atom indices of the second group. If None (default), the self-part G_s is
        computed by tracking the atoms in at_g1 over time.
    r_bins : int, optional
        Number of radial histogram bins. Default: 100.
    r_range : list of float, optional
        [r_min, r_max] in Å for the spatial histogram. Default: [0.0, 5.0].
    tau_sep : float or None, optional
        Time separation between successive τ values, in fs. If None (default),
        the trajectory timestep is used. Values smaller than the timestep are
        silently clipped to the timestep.
    tau_range : list of float, optional
        [τ_min, τ_max] in ps for the correlation time axis. If the requested
        maximum exceeds the trajectory length, it is capped automatically with
        a warning. Default: [0.0, 5.0].

    Results dataclass fields (populated after calling calculate())
    -------------------------------------------------------------
    r       : ndarray, shape (r_bins,) — radial bin centres in Å.
    r_edges : ndarray, shape (r_bins+1,) — radial bin edges in Å.
    tau     : ndarray — time axis in ps.
    g_rt    : ndarray, shape (n_tau, r_bins) — G(r, τ) values.
    rpd     : ndarray, shape (n_tau, r_bins) — radial probability distribution
              4π r² G(r, τ).
    alpha2  : ndarray, shape (n_tau,) — non-Gaussian parameter α₂(τ)
              (self-part only; zero otherwise).
    msd     : ndarray, shape (n_tau,) — mean squared displacement in Å²
              (self-part only; zero otherwise).
    isf     : ndarray — intermediate scattering function F(k, τ)
              (populated by plot_isf()).
    plot    : Figure — most recently produced plot figure.
    """
    def __init__(self, traj, at_g1, at_g2=None, r_bins=100, r_range=[0.0, 5.0], tau_sep=None, tau_range=[0.0, 5.0]):
        self.traj = traj
        self.g1 = at_g1
        self.g2 = at_g2
        self.is_self = self.g2 is None
        
        if self.is_self:
            self.pairs = np.asarray(list(zip(self.g1, self.g1)))
        else:
            self.pairs = np.asarray([x for x in set(itertools.product(self.g1, self.g2)) if x[0] != x[1]])
            
        settings = make_dataclass("Settings", ["bins", "range"])
        self.settings = settings(r_bins, r_range)
        
        # --- Robust Time Step Alignment ---
        if tau_sep is None:
            warnings.warn('tau_sep not defined, defaulting to trajectory timestep.')
            self.tau_sep = float(self.traj.timestep)
        else:
            if tau_sep < self.traj.timestep:
                self.tau_sep = float(self.traj.timestep)
                warnings.warn(f'tau_sep cannot be smaller than trajectory timestep. Defaulting to: {self.tau_sep}')
            else:
                steps = max(1, round(tau_sep / self.traj.timestep))
                self.tau_sep = steps * self.traj.timestep
        
        # --- Robust Trajectory Length Capping ---
        max_tau_ps = (self.traj.nframes - 1) * self.traj.timestep / 1000.0
        req_tau_max_ps = float(tau_range[1])

        if req_tau_max_ps > max_tau_ps:
            warnings.warn(f"Requested max tau ({req_tau_max_ps} ps) exceeds trajectory length. Capping to {max_tau_ps:.2f} ps.")
            req_tau_max_ps = max_tau_ps

        self.tau_range = [float(tau_range[0]), req_tau_max_ps]
        tau_step_ps = self.tau_sep / 1000.0

        results_cls = make_dataclass("Results", ["r_edges", "r", "tau", "rpd", "g_rt", "plot", "alpha2", "msd", "isf"])
        self.results = results_cls(
            r_edges=None, r=None, tau=None, rpd=None, g_rt=None, plot=None, alpha2=None, msd=None, isf=None
        )
        _, self.results.r_edges = np.histogram([-1], **asdict(self.settings))
        self.results.r = (self.results.r_edges[:-1] + self.results.r_edges[1:]) / 2

        self.results.tau = np.arange(self.tau_range[0], self.tau_range[1] + (tau_step_ps * 0.1), tau_step_ps)
        
        self.results.rpd = np.zeros((len(self.results.tau), len(self.results.r)))
        self.results.g_rt = np.zeros((len(self.results.tau), len(self.results.r)))
        self.results.alpha2 = np.zeros(len(self.results.tau))
        self.results.msd = np.zeros(len(self.results.tau))
        
        # --- Robust Volume Calculation ---
        cell = np.asarray(self.traj.cell)
        try:
            vol_scalar = np.abs(np.dot(cell[0], np.cross(cell[1], cell[2])))
        except (IndexError, ValueError):
            vol_scalar = 0.0

        if vol_scalar < 1e-6:
            if not self.is_self:
                warnings.warn("Trajectory cell volume is 0.0 or invalid! Distinct G_d(r,t) normalization "
                              "requires bulk density. Falling back to volume = 1.0.")
            vol_scalar = 1.0
            
        self.volume = np.full(shape=self.settings.bins, fill_value=vol_scalar, dtype=np.float64)

    def calculate(self, plot=False, tau_sel=None, r_sel=None, plot_vs="r", levels=60, cmap=None, save_fig=None, **kwargs):
        """Compute G(r, τ) using raw moments and sliding windows."""
        t0_stride = max(1, int(round(self.tau_sep / self.traj.timestep)))
        bin_w = np.diff(self.results.r_edges)
        vols = 4/3 * np.pi * np.diff(np.power(self.results.r_edges, 3))
        
        if self.is_self:
            pos_all = np.stack([f.pos[self.g1] for f in self.traj])
            for i, tau in enumerate(self.results.tau):
                sep = int(round(1000 * tau / self.traj.timestep))
                if sep >= self.traj.nframes: continue
                t0s = np.arange(0, self.traj.nframes - sep, t0_stride)
                if len(t0s) == 0: continue
                dists = np.linalg.norm((pos_all[t0s + sep] - pos_all[t0s]).reshape(-1, 3), axis=1)
                r2, r4 = np.mean(dists**2), np.mean(dists**4)
                self.results.msd[i] = r2
                self.results.alpha2[i] = (3 * r4) / (5 * r2**2) - 1 if r2 > 1e-12 else 0
                c, _ = np.histogram(dists, **asdict(self.settings))
                self.results.rpd[i] = c / (len(dists) * bin_w)
                self.results.g_rt[i] = self.results.rpd[i] / (4 * np.pi * self.results.r**2)
                update_progress(i / len(self.results.tau))
        else:
            density = len(self.pairs) / self.volume[0]
            p1 = np.stack([f.pos[self.pairs[:,0]] for f in self.traj])
            p2 = np.stack([f.pos[self.pairs[:,1]] for f in self.traj])
            for i, tau in enumerate(self.results.tau):
                sep = int(round(1000 * tau / self.traj.timestep))
                if sep >= self.traj.nframes: continue
                t0s = np.arange(0, self.traj.nframes - sep, t0_stride)
                all_d = []
                for t0 in t0s:
                    d = xys2cart(cart2xys(p2[t0+sep]-p1[t0], self.traj.cell) % 1.0 - 0.5, self.traj.cell)
                    all_d.append(np.linalg.norm(d, axis=1))
                dists = np.concatenate(all_d)
                c, _ = np.histogram(dists, **asdict(self.settings))
                self.results.g_rt[i] = c / (density * vols * len(t0s))
                self.results.rpd[i] = c / (len(t0s) * len(self.pairs) * bin_w)
                update_progress(i / len(self.results.tau))
        update_progress(1.0)
        
        if plot:
            self.plot_probability_density(tau_sel=tau_sel, r_sel=r_sel, plot_vs=plot_vs, levels=levels, cmap=cmap, save_fig=save_fig)
        return self.results

    def _get_cmap(self, cmap):
        if cmap is not None: return cmap
        try: return style.cdf_cmap
        except: return "RdYlBu_r"

    def plot_probability_density(self, tau_sel=None, r_sel=None, plot_vs="r", levels=50, cmap=None, tau_lim=None, r_lim=None, saturation_factor=1.0, save_fig=None):
        """Plot the Van Hove correlation function G(r, τ) as a 2D contour."""
        cmap = self._get_cmap(cmap)
        zmin, zmax = np.nanmin(self.results.g_rt), np.nanmax(self.results.g_rt)
        if zmin == zmax == 0.0: zmax = 1.0 
        vmax_val = zmin + saturation_factor * (zmax - zmin if zmax > zmin else 1.0)
        levels_arr = np.linspace(zmin, vmax_val, levels)

        print("-" * 75)
        print(f" RAW DENSITY G(r,t) INTERPRETATION (Plot vs {plot_vs.upper()})")
        if self.is_self:
            if plot_vs == "r":
                print(" - Visual : A massive spike at r=0 that drops and widens over time.")
            else:
                print(" - Visual : Decay curves at fixed r. At r=0, shows the residence time in the initial cage.")
        else:
            if plot_vs == "r":
                print(" - Visual : Structural peaks that melt into a flat bulk density (1.0) over time.")
            else:
                print(" - Visual : At a fixed shell distance (e.g., 1st neighbor), shows the lifetime ")
                print("            and decay rate of the structural cage.")
        print("-" * 75)

        fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6), constrained_layout=True) if (tau_sel is not None or r_sel is not None) else plt.subplots(figsize=(4.8, 3.6))
        ax_contour = axes[1] if (tau_sel is not None or r_sel is not None) else axes

        if tau_sel is not None and plot_vs == "r":
            tau_sel = np.asarray(tau_sel)
            idx = (tau_sel * 1000 // self.tau_sep).astype(int) - 1
            for i in idx:
                if 0 <= i < len(self.results.tau):
                    axes[0].plot(self.results.r, self.results.g_rt[i,:], label=f"{self.results.tau[i]:.2f} ps")
            if r_lim is not None: axes[0].set_xlim(*r_lim)
            axes[0].set_xlabel(r"Distance ($\mathrm{\AA}$)")
            axes[0].set_ylabel(r"Density $G(r, \tau)$")
            axes[0].set_title(r"Density vs Distance")
            axes[0].legend()

        elif r_sel is not None and plot_vs == "tau":
            r_sel = np.asarray(r_sel)
            idx = [np.argmin(np.abs(self.results.r - r_val)) for r_val in r_sel]
            for i in idx:
                if 0 <= i < len(self.results.r):
                    axes[0].plot(self.results.tau, self.results.g_rt[:,i], label=f"{self.results.r[i]:.2f} $\\mathrm{{\\AA}}$")
            if tau_lim is not None: axes[0].set_xlim(*tau_lim)
            axes[0].set_xlabel(r"$\tau$ (ps)")
            axes[0].set_ylabel(r"Density $G(r, \tau)$")
            axes[0].set_title(r"Density vs Time")
            axes[0].legend()

        if plot_vs == "r":
            CS = ax_contour.contourf(self.results.r, self.results.tau, self.results.g_rt, vmax=vmax_val, levels=levels_arr, cmap=cmap)
            ax_contour.contour(self.results.r, self.results.tau, self.results.g_rt, colors="k", vmax=vmax_val, levels=levels_arr, linewidths=0.1)
            ax_contour.set_xlabel(r"Distance ($\mathrm{\AA}$)")
            ax_contour.set_ylabel(r"$\tau$ (ps)")
        else:
            CS = ax_contour.contourf(self.results.tau, self.results.r, self.results.g_rt.T, vmax=vmax_val, levels=levels_arr, cmap=cmap)
            ax_contour.contour(self.results.tau, self.results.r, self.results.g_rt.T, colors="k", vmax=vmax_val, levels=levels_arr, linewidths=0.1)
            ax_contour.set_xlabel(r"$\tau$ (ps)")
            ax_contour.set_ylabel(r"Distance ($\mathrm{\AA}$)")

        ax_contour.set_title("Density G(r,t)")
        
        if r_lim is not None: ax_contour.set_ylim(*r_lim) if plot_vs == "tau" else ax_contour.set_xlim(*r_lim)
        if tau_lim is not None: ax_contour.set_xlim(*tau_lim) if plot_vs == "tau" else ax_contour.set_ylim(*tau_lim)

        cbar = plt.colorbar(CS, pad=0.035, location="right", ax=ax_contour)
        cbar.set_label(r"Density $G(r, \tau)$")
        
        if save_fig:
            fig.savefig(save_fig, dpi=300, bbox_inches='tight')
        self.results.plot = fig
        plt.show()

    def plot_radial_probability(self, tau_sel=None, r_sel=None, plot_vs="r", levels=50, cmap=None, tau_lim=None, r_lim=None, saturation_factor=1.0, save_fig=None):
        """Plot the radial probability distribution RPD(r, τ) = 4π r² G(r, τ)."""
        cmap = self._get_cmap(cmap)
        
        zmin, zmax = np.nanmin(self.results.rpd), np.nanmax(self.results.rpd)
        if zmin == zmax == 0.0: zmax = 1.0
        vmax_val = zmin + saturation_factor * (zmax - zmin if zmax > zmin else 1.0)
        levels_arr = np.linspace(zmin, vmax_val, levels)

        print("-" * 75)
        print(f" RPD VISUALIZATION INTERPRETATION (Plot vs {plot_vs.upper()})")
        if self.is_self:
            if plot_vs == "r":
                print(" - Meaning: At what distance 'r' is the atom most likely to be found right now?")
            else:
                print(" - Meaning: At a fixed distance (e.g., hopping site), when is the probability flux highest?")
                print(" - Visual : The peak in time tells you the characteristic hopping time to that specific site.")
        else:
            if plot_vs == "r":
                print(" - Meaning: How many neighbor atoms are contained in the shell at distance 'r'?")
            else:
                print(" - Meaning: How does the number of neighbors at a fixed distance change over time?")
        print("-" * 75)

        fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6), constrained_layout=True) if (tau_sel is not None or r_sel is not None) else plt.subplots(figsize=(4.8, 3.6))
        ax_contour = axes[1] if (tau_sel is not None or r_sel is not None) else axes

        if tau_sel is not None and plot_vs == "r":
            tau_sel = np.asarray(tau_sel)
            idx = (tau_sel * 1000 // self.tau_sep).astype(int) - 1
            for i in idx:
                if 0 <= i < len(self.results.tau):
                    axes[0].plot(self.results.r, self.results.rpd[i,:], label=f"{self.results.tau[i]:.2f} ps")
            if r_lim is not None: axes[0].set_xlim(*r_lim)
            axes[0].set_xlabel(r"Distance ($\mathrm{\AA}$)")
            axes[0].set_ylabel(r"Radial Probability $P(r, \tau)$")
            axes[0].set_title(r"RPD vs Distance")
            axes[0].legend()

        elif r_sel is not None and plot_vs == "tau":
            r_sel = np.asarray(r_sel)
            idx = [np.argmin(np.abs(self.results.r - r_val)) for r_val in r_sel]
            for i in idx:
                if 0 <= i < len(self.results.r):
                    axes[0].plot(self.results.tau, self.results.rpd[:,i], label=f"{self.results.r[i]:.2f} $\\mathrm{{\\AA}}$")
            if tau_lim is not None: axes[0].set_xlim(*tau_lim)
            axes[0].set_xlabel(r"$\tau$ (ps)")
            axes[0].set_ylabel(r"Radial Probability $P(r, \tau)$")
            axes[0].set_title(r"RPD vs Time")
            axes[0].legend()

        if plot_vs == "r":
            CS = ax_contour.contourf(self.results.r, self.results.tau, self.results.rpd, vmax=vmax_val, levels=levels_arr, cmap=cmap)
            ax_contour.contour(self.results.r, self.results.tau, self.results.rpd, colors="k", vmax=vmax_val, levels=levels_arr, linewidths=0.1)
            ax_contour.set_xlabel(r"Distance ($\mathrm{\AA}$)")
            ax_contour.set_ylabel(r"$\tau$ (ps)")
        else:
            CS = ax_contour.contourf(self.results.tau, self.results.r, self.results.rpd.T, vmax=vmax_val, levels=levels_arr, cmap=cmap)
            ax_contour.contour(self.results.tau, self.results.r, self.results.rpd.T, colors="k", vmax=vmax_val, levels=levels_arr, linewidths=0.1)
            ax_contour.set_xlabel(r"$\tau$ (ps)")
            ax_contour.set_ylabel(r"Distance ($\mathrm{\AA}$)")

        ax_contour.set_title("Early-Time Vibration Cage (RPD)" if (plot_vs=="tau" and self.is_self) else "Radial Probability Distribution")
        
        if r_lim is not None: ax_contour.set_ylim(*r_lim) if plot_vs == "tau" else ax_contour.set_xlim(*r_lim)
        if tau_lim is not None: ax_contour.set_xlim(*tau_lim) if plot_vs == "tau" else ax_contour.set_ylim(*tau_lim)

        cbar = plt.colorbar(CS, pad=0.035, location="right", ax=ax_contour)
        cbar.set_label(r"Radial Probability $P(r, \tau)$")
        
        if save_fig:
            fig.savefig(save_fig, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_msd(self, save_fig=None):
        """Plot the mean squared displacement ⟨Δr²⟩(τ) on a log–log scale."""
        if not self.is_self:
            print("-" * 75)
            print(" NOTE: Mean Squared Displacement (MSD) calculation skipped.")
            print("       MSD is strictly a single-particle property valid only for the")
            print("       Self-Part (tracking the same atoms over time).")
            print("-" * 75)
            return
        
        fig, ax = plt.subplots()
        valid = (self.results.tau > 0) & (self.results.msd > 1e-12)
        ax.plot(self.results.tau[valid], self.results.msd[valid], 'k-', lw=2)
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r"Time $\tau$ (ps)")
        ax.set_ylabel(r"MSD $\langle \Delta r^2 \rangle$ ($\mathrm{\AA}^2$)")
        ax.set_title("Mean Squared Displacement")
        ax.grid(True, which="both", ls="--", alpha=0.5)
        plt.tight_layout()
        
        if save_fig:
            fig.savefig(save_fig, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_isf(self, k_val=2.5, save_fig=None):
        """Compute and plot the intermediate scattering function F(k, τ)."""
        self.results.isf = np.zeros(len(self.results.tau))
        dr = np.diff(self.results.r_edges)
        
        for i in range(len(self.results.tau)):
            kr = k_val * self.results.r
            sinc = np.sin(kr) / kr
            sinc[np.isnan(sinc)] = 1.0 
            
            if self.is_self:
                self.results.isf[i] = np.sum(self.results.rpd[i] * sinc * dr)
            else:
                # The prefactor for distinct ISF should be the number density of the 
                # target particles (nB / V), not the pair density, to ensure it is an intensive property.
                # using len(set(self.pairs[:, 1])) guarantees unique target atoms.
                nB = len(set(self.pairs[:, 1]))
                density = nB / self.volume[0]
                integrand = 4 * np.pi * (self.results.r**2) * density * (self.results.g_rt[i] - 1.0) * sinc
                self.results.isf[i] = np.sum(integrand * dr)

        fig, ax = plt.subplots()
        ax.plot(self.results.tau, self.results.isf, 'b-', lw=2)
        
        if self.is_self:
            ax.axhline(1/np.e, color='r', linestyle='--', label=r'$1/e$ decay ($\tau_\alpha$)')
            ax.set_ylabel(f"$F_s(k, \\tau)$")
            ax.set_title(f"Self-ISF ($k={k_val}$ $\\mathrm{{\\AA}}^{{-1}}$)")
        else:
            ax.axhline(0, color='k', linestyle='-', linewidth=0.5)
            ax.set_ylabel(f"$F_d(k, \\tau)$")
            ax.set_title(f"Distinct-ISF ($k={k_val}$ $\\mathrm{{\\AA}}^{{-1}}$)")

        ax.set_xlabel(r"Time $\tau$ (ps)")
        ax.legend()
        plt.tight_layout()
        
        if save_fig:
            fig.savefig(save_fig, dpi=300, bbox_inches='tight')
        plt.show()

    def summary(self):
        """Print a scientific interpretation of all computed observables."""
        print("="*75)
        print(" VAN HOVE CORRELATION FUNCTION - COMPREHENSIVE SUMMARY")
        print("="*75)
        mode = "Self-Part (G_s)" if self.is_self else "Distinct-Part (G_d)"
        print(f"Calculation Mode : {mode}")
        print(f"Time Range       : {self.tau_range[0]} to {self.tau_range[1]} ps")
        print(f"Spatial Range    : {self.settings.range[0]} to {self.settings.range[1]} Å")
        
        if self.is_self:
            max_alpha2_idx = np.argmax(self.results.alpha2)
            max_alpha2_val = self.results.alpha2[max_alpha2_idx]
            tau_star = self.results.tau[max_alpha2_idx]
            
            print("-" * 75)
            print(" NON-GAUSSIAN DYNAMICS (alpha_2):")
            print(f" Peak alpha_2    : {max_alpha2_val:.4f}")
            print(f" Peak time (tau*): {tau_star:.4f} ps")
            
            if max_alpha2_val < 0.1:
                print(" Status          : The system exhibits simple, Fickian/Gaussian diffusion.")
            else:
                print(" Status          : The system exhibits strong dynamical heterogeneity.")
                print("                   Atoms are likely 'caged' by neighbors and undergo ")
                print("                   discrete hopping/jump events rather than smooth diffusion.")
        
        print("-" * 75)
        print(" DETAILED VISUALIZATION INTERPRETATIONS:")
        if self.is_self:
            print(" 1. Spatial Domain (Plot vs Distance 'r'):")
            print("    - Density G(r,t): Displays the unweighted point density. It fundamentally ")
            print("      peaks at r=0. Watch the decay of the peak at r=0. If secondary peaks ")
            print("      emerge at larger distances while the r=0 peak is prominent, it confirms jumping.")
            print("    - RPD(r,t): True displacement probability scaled by shell volume (4*pi*r^2).")
            print("      The peak shifts outward over time. Secondary peaks at longer times ")
            print("      (e.g., 2.5 - 3.0 Å) are the 'smoking gun' of discontinuous hopping.")
            print("")
            print(" 2. Time Domain (Plot vs Time 'tau'):")
            print("    - Slicing at r=0 shows the residence time (exactly how long an atom stays caged).")
            print("    - Slicing at r > 0 (a hopping site) reveals the characteristic arrival time ")
            print("      when the maximum number of atoms transition to that specific lattice site.")
            print("")
            print(" 3. Mean Squared Displacement [MSD] (plot_msd):")
            print("    - Describes macroscopic diffusion mapping the average displacement.")
            print("    - Log-Log Slope Interpretation: ")
            print("      Slope=2 (Ballistic/Inertial) -> Slope~0 (Caging) -> Slope=1 (Fickian Diffusion).")
            print("")
            print(" 4. Self-Intermediate Scattering Function [F_s(k,t)] (plot_isf):")
            print("    - Represents the time-domain equivalent of incoherent neutron scattering.")
            print("    - Decay time to 1/e (~0.368) defines the alpha-relaxation time (tau_alpha).")
            print("    - A two-step decay indicates glassy dynamics and pronounced caging effects.")
        else:
            print(" 1. Spatial Domain (Plot vs Distance 'r'):")
            print("    - Density G_d(r,t): Shows how clustered neighbor atoms are compared to the ")
            print("      average bulk density. Oscillates with peaks/valleys representing coordination ")
            print("      shells. Over time, these structural peaks 'melt' and flatten to 1.0.")
            print("    - RPD_d(r,t): Integrating the very first peak at t=0 yields the exact ")
            print("      Coordination Number for this specific pair of atoms.")
            print("")
            print(" 2. Time Domain (Plot vs Time 'tau'):")
            print("    - Slicing at a fixed structural peak (e.g., the 1st neighbor distance) ")
            print("      visualizes the exact lifetime and decay rate of the structural cage.")
            print("")
            print(" 3. Distinct-Intermediate Scattering Function [F_d(k,t)] (plot_isf):")
            print("    - Represents coherent scattering (collective structural relaxation).")
            print("    - The rate at which the collective 'cage' dissolves. Starts at a value ")
            print("      proportional to S(k)-1 and decays to 0 as the structure randomizes.")
        print("="*75)


# SDF
class SDF(object):
    """Scalar distribution function g(δ) for an arbitrary observable.

    The SDF acts as a robust histogramming wrapper to compute the normalized 
    probability density function of any 1D scalar property (e.g., local density, 
    energy, or coordination number).

    The distribution is normalized such that:
        ∫ g(δ) dδ = 1

    Mathematically, this is:
        g(δ) = N_obs(δ) / [ N_total * Δδ ]

    Parameters
    ----------
    traj : Trajectory
        Trajectory object to analyse (used primarily to extract the time axis).
    scaler : ndarray or list
        The 1D array of scalar values to be histogrammed.
    nbins : int, optional
        Number of histogram bins. Default: 200.
    range : tuple of float, optional
        (δ_min, δ_max). If None, uses the min and max of the scaler array.

    Results dataclass fields
    ------------------------
    x        : ndarray — bin centres.
    sdf      : ndarray — probability density g(δ).
    t        : ndarray — time axis in ps.
    """
    def __init__(self, traj, scaler, nbins=200, range=None):
        self.traj = traj
        s = np.asarray(scaler, dtype=float)
        if range is None: range = (s.min(), s.max())
        self.settings = make_dataclass("Settings", "bins range")(nbins, range)
        self.results = make_dataclass("Results", "edges count x sdf plot t scaler com_plot")()
        self.results.scaler = s
        self.results.t = np.linspace(0, traj.timestep * (traj.nframes - 1) / 1000, traj.nframes)

    def calculate(self, plot=False, x_label=None, y_label=None, com_plot=False, **kwargs):
        """Compute normalization: ∫ g(δ) dδ = 1."""
        c, e = np.histogram(self.results.scaler, **asdict(self.settings))
        self.results.edges, self.results.count, self.results.x = e, c, (e[:-1] + e[1:]) / 2
        self.results.sdf = c / (len(self.results.scaler) * (e[1] - e[0]))
        if plot:
            fig, ax = plt.subplots(figsize=(4.2, 3.6))
            ax.plot(self.results.x, self.results.sdf, **kwargs)
            ax.set_xlabel(x_label if x_label else r"$\delta$")
            ax.set_ylabel(y_label if y_label else r"$g(\delta)$")
            plt.show(); self.results.plot = fig
        return self.results

    def summary(self):
        """Print a scientific interpretation of the computed SDF."""
        print("="*75)
        print(" SCALAR DISTRIBUTION FUNCTION (SDF) SUMMARY")
        print("="*75)
        
        # Compute basic statistics
        mean_val = np.mean(self.results.scaler)
        std_val = np.std(self.results.scaler)
        peak_idx = np.argmax(self.results.sdf)
        
        print(f" Data Points    : {len(self.results.scaler)}")
        print(f" Global Mean    : {mean_val:.4f}")
        print(f" Standard Dev.  : {std_val:.4f}")
        print(f" Peak Value (x) : {self.results.x[peak_idx]:.4f} (Most probable state)")
        print("-" * 75)
        print(" INTERPRETATION GUIDE:")
        print(" - The SDF is normalized as a true probability density.")
        print(" - The integral of the curve across all x equals exactly 1.0.")
        print(" - Broad distributions indicate large fluctuations or heterogeneity in the property.")
        print(" - Multiple peaks indicate distinct structural or thermodynamic states.")
        print("="*75)
