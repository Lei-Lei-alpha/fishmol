"""A range of functions to analyse the trajectory object"""

import numpy as np
import itertools
import warnings
from typing import List, Tuple, Union, Optional, Any, Sequence
from recordclass import make_dataclass, asdict
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from fishmol.utils import to_sublists, make_comb, update_progress
from fishmol import style
from scipy.optimize import curve_fit


# RDF
class RDF(object):
    """
    Pair distribution function:
    Calculated by histogramming distances between all particles in `g1` and `g2` while taking
    periodic boundary conditions into account via the minimum image
    convention.

    -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    Arguments
    at_g1 : list of int, selected atoms to be calculated
    at_g2 : list of int, selected atoms to be calculated
    nbins : int (optional), number of bins (resolution) in the histogram
    range : tuple or list (optional), the size of the RDF
    -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    Returns a dataclass containing the analysis results
    x :  np.array, radii over which g(r) is computed
    rdf : 
    plot : 
    t : 
    pairs : 
    scaler : np.array distances
    com_plot : 
    """
    def __init__(self, traj: Any, at_g1: Union[List[int], Any], at_g2: Union[List[int], Any], nbins: int = 200, range: Tuple[float, float] = (0.0, 15.0)) -> None:
        self.g1 = at_g1
        self.g2 = at_g2
        self.traj = traj
        settings = make_dataclass("Settings", "bins range")
        self.settings = settings(nbins, range)
        results = make_dataclass("Results", "edges count x rdf plot t pairs scaler com_plot")
        self.results = results
        self.results.count, self.results.edges = np.histogram([-1], **asdict(self.settings))
        self.results.x = (self.results.edges[:-1] + self.results.edges[1:])/2
        self.results.t = np.linspace(0, self.traj.timestep * (self.traj.nframes - 1)/1000, self.traj.nframes)
        # Need to know average volume
        self.volume = np.zeros(self.settings.bins) + np.asarray(self.traj.cell[0]).dot(np.cross(np.asarray(self.traj.cell[1]), np.asarray(self.traj.cell[2])))
        
    def calculate(self, plot: bool = False, com_plot: bool = False, **kwargs: Any) -> Any:
        """
        Calculate the RDF of two atom groups by calling this function.
         
        """
        dists = np.asarray([self.traj.frames[i].dists(self.g1, self.g2, cutoff = self.settings.range[1], mic = True)[1] for i in range(self.traj.nframes)])
        count = np.histogram(dists, **asdict(self.settings))[0]
        self.results.count = count
        # Use the volume of the simulation box
        
        # Number of each selection
        nA = len(self.g1)
        nB = len(self.g2)
        N = nA * nB
            
        # Volume in each radial shell
        vols = np.power(self.results.edges, 3)
        vol = 4/3 * np.pi * np.diff(vols)
        # Average number density
        density = N / self.volume # number of particles per volume
        # Save pairs as label
        self.results.pairs = list(itertools.product(self.g1, self.g2))
        # Distances for the temporal plot
        self.results.scaler = dists
        # rdf
        self.results.rdf = np.asarray(self.results.count / (density * vol * self.traj.nframes))
        
        if plot:
            fig, ax = plt.subplots(figsize = (4.2, 3.6))
            ax.plot(self.results.x, self.results.rdf, **kwargs)
            ax.set_xlabel(r"$r$ ($\AA$)")
            ax.set_ylabel(r"$g(r)$")
            plt.show()
            self.results.plot = fig
        
        if com_plot:
            fig = plt.figure(figsize=(4.8,3.6))
            ax  = fig.add_axes([0.20, 0.16, 0.685, 0.75])
            for i in range(self.results.scaler.shape[1]):
                ax.plot(self.results.t, self.results.scaler[:,i], label = f"Pair {self.results.pairs[i]}", **kwargs)

            ax.set_xlabel(r"$t$ (ps)")
            ax.set_ylabel(r"$r$ ($\mathrm{\AA}$)")
            plt.legend(frameon = False, ncol = 2, bbox_to_anchor=(0.4, 0.9, 0.4, 0.5), loc='center', fontsize = "small")
            divider = make_axes_locatable(ax)
            ax_histy = divider.append_axes("right", 0.5, pad=0.05, sharey=ax)
            ax_histy.yaxis.tick_right()
            ax_histy.plot(self.results.rdf, self.results.x)
            plt.show()
            self.results.com_plot = fig
            
        return self.results

# ADF
class ADF(object):
    """Angular distribution function
    -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    Arguments
    at_g1 : AtomGroup
      First AtomGroup
    at_g2 : AtomGroup
      Second AtomGroup
    nbins : int (optional)
          Number of bins in the histogram
    range : tuple or list (optional)
          The size of the RDF
    verbose : bool (optional)
          Show detailed progress of the calculation if set to ``True``
    -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    Returns
    g_r : (n_radii) np.array
        radial distribution function values g(r).
    radii : (n_radii) np.array
        radii over which g(r) is computed
    """
    def __init__(self, traj: Any, at_g1: Any, at_g2: Any, at_g3: Any, nbins: int = 200, range: Tuple[float, float] = (0.0, 180.0)) -> None:
        self.g1 = at_g1
        self.g2 = at_g2
        self.g3 = at_g3
        self.traj = traj
        settings = make_dataclass("Settings", "bins range")
        self.settings = settings(nbins, range)
        count, edges = np.histogram([-1], **asdict(self.settings))
        count = count.astype(np.float64)
        count *= 0.0
        results = make_dataclass("Results", "edges count x adf plot t pairs scaler com_plot")
        self.results = results
        self.results.edges = edges
        self.results.x = (edges[:-1] + edges[1:])/2
        self.results.count = count
        self.results.t = np.linspace(0, self.traj.timestep * (self.traj.nframes - 1)/1000, self.traj.nframes)
        
    def calculate(self, cone_correction: bool = True, plot: bool = False, com_plot: bool = False, **kwargs: Any) -> Any:
        angles = []
        for frame in self.traj:
            pairs, angle = frame.angles(self.g1, self.g2, self.g3, mic = True)
            count = np.histogram(angle, **asdict(self.settings))[0]
            self.results.count += count
            angles.append(angle)
        
        self.results.pairs = pairs
        self.results.scaler = np.asarray(angles)
        # Number of each selection
        NA = len(self.g1)
        NB = len(self.g2)
        NC = len(self.g3)
        
        # Calculate ADF
        if cone_correction:
            self.results.adf = np.asarray(self.results.count / (NA * NB * NC * self.traj.nframes * np.sin(self.results.x * np.pi / 180))) # cone correction, convert degree to radians
        else:
            self.results.adf = np.asarray(self.results.count / (NA * NB * NC * self.traj.nframes))
        
        if plot:
            fig, ax = plt.subplots(figsize = (4.2, 3.6))
            ax.plot(self.results.x, self.results.adf, **kwargs)
            ax.set_xlabel(r"$\alpha$ ($^{\circ}$)")
            ax.set_ylabel(r"$g(\alpha)$")
            plt.show()
            self.results.plot = fig
        
        if com_plot:
            fig = plt.figure(figsize=(4.8,3.6))
            ax  = fig.add_axes([0.20, 0.16, 0.685, 0.75])
            for i in range(self.results.scaler.shape[1]):
                ax.plot(self.results.t, self.results.scaler[:,i], label = f"Pair {self.results.pairs[i]}", **kwargs)

            ax.set_xlabel(r"$t$ (ps)")
            ax.set_ylabel(r"$\alpha$ ($^{\circ}$)")
            plt.legend(frameon = False, ncol = 2, bbox_to_anchor=(0.4, 0.9, 0.4, 0.5), loc='center', fontsize = "small")
            divider = make_axes_locatable(ax)
            ax_histy = divider.append_axes("right", 0.5, pad=0.05, sharey=ax)
            ax_histy.yaxis.tick_right()
            ax_histy.plot(self.results.adf, self.results.x)
            plt.show()
            self.results.com_plot = fig
            
        return self.results

# DDF
class DDF(object):
    """Dihedral distribution function
    -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    Arguments
    at_g1 : AtomGroup
      First AtomGroup
    at_g2 : AtomGroup
      Second AtomGroup
    nbins : int (optional)
          Number of bins in the histogram
    range : tuple or list (optional)
          The size of the RDF
    -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    Returns
    g_r : (n_radii) np.array
        radial distribution function values g(r).
    radii : (n_radii) np.array
        radii over which g(r) is computed
    """
    def __init__(self, traj: Any, at_g: Any, nbins: int = 200, range: Tuple[float, float] = (0.0, 180.0)) -> None:
        self.g = at_g
        self.traj = traj
        settings = make_dataclass("Settings", "bins range")
        self.settings = settings(nbins, range)
        count, edges = np.histogram([-1], **asdict(self.settings))
        count = count.astype(np.float64)
        count *= 0.0
        results = make_dataclass("Results", "edges count x ddf plot t angles com_plot")
        self.results = results
        self.results.edges = edges
        self.results.x = (edges[:-1] + edges[1:])/2
        self.results.count = count
        self.results.t = np.linspace(0, self.traj.timestep * (self.traj.nframes - 1)/1000, self.traj.nframes)
        
    def calculate(self, plot: bool = False, com_plot: bool = False, **kwargs: Any) -> Any:
        angles = []
        for frame in self.traj:
            pairs, angle = frame.dihedrals(self.g, mic = True)
            count = np.histogram(angle, **asdict(self.settings))[0]
            self.results.count += count
            angles.append(angle)
        
        self.results.angles = np.asarray(angles)
        
        N = len(pairs)
        
        # Calculate DDF
        self.results.ddf = np.asarray(self.results.count) / (N * self.traj.nframes)
        
        if plot:
            fig, ax = plt.subplots(figsize = (4.2, 3.6))
            ax.plot(self.results.x, self.results.ddf, **kwargs)
            ax.set_xlabel(r"$\delta$ ($^{\circ}$)")
            ax.set_ylabel(r"$g(\delta)$")
            plt.show()
            self.results.plot = fig
        
        if com_plot:
            fig = plt.figure(figsize=(4.8,3.6))
            ax  = fig.add_axes([0.20, 0.16, 0.685, 0.75])
            for i in range(self.results.angles.shape[1]):
                ax.plot(self.results.t, self.results.angles[:,i], **kwargs)

            ax.set_xlabel(r"$t$ (ps)")
            ax.set_ylabel(r"$\delta$ ($^{\circ}$)")
            divider = make_axes_locatable(ax)
            ax_histy = divider.append_axes("right", 0.5, pad=0.05, sharey=ax)
            ax_histy.yaxis.tick_right()
            ax_histy.plot(self.results.ddf, self.results.x)
            plt.show()
            self.results.com_plot = fig
            
        return self.results


# CDF
class CDF(object):
    """Combined distribution function
    -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    Arguments
    at_g1 : AtomGroup
      First AtomGroup
    at_g2 : AtomGroup
      Second AtomGroup
    nbins : int (optional)
          Number of bins in the histogram
    range : tuple or list (optional)
          The size of the RDF
    verbose : bool (optional)
          Show detailed progress of the calculation if set to ``True``
    -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    Returns
    g_r : (n_radii) np.array
        radial distribution function values g(r).
    radii : (n_radii) np.array
        radii over which g(r) is computed
    """
    def __init__(self, scaler1: Any, scaler2: Any, names: Optional[List[str]] = None, range: Optional[Tuple[List[float], List[float]]]  = None) -> None:
        self.s1 = scaler1
        self.s2 = scaler2
        self.names = names
        if range is None:
            range = ([0, getattr(self.s1, names[0]).max()],[0, getattr(self.s2, names[1]).max()])
        settings = make_dataclass("Settings", "range")
        self.settings = settings(range)
        results = make_dataclass("Results", "xedges yedges xcenters ycenters x y cdf plot")
        self.results = results
        
    def calculate(self, plot: bool = True, **kwargs: Any) -> Any:
        self.results.xedges = self.s1.edges
        self.results.yedges = self.s2.edges

        x = self.s1.scaler.flatten()
        y = self.s2.scaler.flatten()

        self.results.cdf, self.results.xedges, self.results.yedges = np.histogram2d(x, y, bins = (self.results.xedges, self.results.yedges), **asdict(self.settings))
        self.results.cdf = self.results.cdf.T # Transpose so that the plot is correct
        
        if plot:
            fig = plt.figure(figsize=(4.8, 4.0))
            ax  = fig.add_axes([0.16, 0.16, 0.82, 0.75])

            X, Y = np.meshgrid(self.s1.x, self.s2.x)
            levels = np.linspace(self.results.cdf.min(), self.results.cdf.max(), 50)
            CS = ax.contourf(X, Y, self.results.cdf, cmap = style.cdf_cmap, levels = levels)

            # ax.set_xlabel(r"$r$ ($\AA$)")
            # ax.set_ylabel(r"$\alpha$ ($^{\circ}$)")
            # ax.set_xlim(2,5)

            divider = make_axes_locatable(ax)

            ax_histx = divider.append_axes("top", 0.5, pad=0.05, sharex=ax)
            ax_histx.xaxis.set_tick_params(labelbottom=False)
            ax_histx.yaxis.set_tick_params(labelleft=False)
            ax_histx.xaxis.tick_top()
            ax_histx.plot(self.s1.x, getattr(self.s1, self.names[0]))

            ax_histy = divider.append_axes("right", 0.5, pad=0.05, sharey=ax)
            ax_histy.xaxis.set_tick_params(labelbottom=False)
            ax_histy.yaxis.set_tick_params(labelleft=False)
            ax_histy.yaxis.tick_right()
            ax_histy.plot(getattr(self.s2, self.names[1]), self.s2.x)

            plt.colorbar(CS, pad=0.035, location = "right", ax = ax)
            # plt.savefig("cdf_water14_15_water143.jpg", dpi = 600)
            self.results.plot = fig
            plt.show()
            
        return self.results

# VRD
class VRD(object):
    """Vector Reorientation Dynamics
    -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    Arguments
    spec : np.array or list of two atoms (two atom groups) that determine the vector
    timestep : the timestep of the vector when using np.array input
    traj : the input Trajectory object
    num : int, length of frames to calculate the reorientation dynamics
    skip : int, frames to skip = skip - 1. Determines the sampling step of the Trajectory, eg. skip = 1, use all frames in the Trajectory: skip = 10, skip 9 frames.
    -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    Returns
    self.results:
    - t: time calculated
    - C_t: the average c_t
    - C_t_error: the error of c_t
    """
    def __init__(self, traj: Any = None, spec: Any = None, timestep: Optional[float] = None, num: int = 2000, sampling: int = 5, skip: int = 10) -> None:
        results = make_dataclass("Results", "t C_t C_t_error t_fit C_t_fit fit_params plot")
        self.results = results
        self.traj = traj
        self.spec = spec
        self.num = num
        self.sampling = sampling
        self.skip = skip
        if traj is not None:
            self.traj = traj
            self.t_step = traj.timestep
        elif traj is None:
            if isinstance(self.spec, np.ndarray):
                if timestep is not None:
                    self.t_step = timestep
                else:
                    warnings.warn("VRD created with array of vectors without specifying timestep, using default timestep of 5 fs, please specify the timestep you are using if this is incorrect.")
                    self.t_step = 5
            else:
                raise Exception("Please either specify\n:(i)a Trajectory object and a list of the indices of two atoms\nor\n(ii) a np.array object of vector data and the timestep of the vector data.")
        else:
            raise Exception("Please either specify\n:(i)a Trajectory object and a list of the indices of two atoms\nor\n(ii) a np.array object of vector data and the timestep of the vector data.")
        self.results.t = np.linspace(0, self.t_step * (self.num - 1) / 1000, num = self.num//self.sampling)
    
    def calculate(self, l: int = 3, mean: bool = True, fit: bool = False, plot: bool = True, log_scale: bool = False, **kwargs: Any) -> Any:
        if self.traj is not None:
            if any([self.spec[0] is None, self.spec[1] is None]):
                raise ValueError("Please specify atom groups ")
            else:
                combs = make_comb(*self.spec)
            frame_chunks = to_sublists(self.traj.frames, self.num)[::self.skip]
            # n_select_frames = len(frame_chunks[0])

            dot_products = np.zeros((len(frame_chunks), len(self.results.t), len(combs)))
            
            for i, frame_chunk in enumerate(frame_chunks):
                select = frame_chunk[::self.sampling]
                dot_products[i,:] = np.asarray([np.diagonal(frame.vecs(combs = combs, absolute = False, normalise = True, mic = True).dot(
                    select[0].vecs(combs = combs, absolute = False, normalise = True, mic = True).T)) for frame in select])
                update_progress(i / len(frame_chunks))
            
        # If the vec is an array of vectors without traj
        else:
            vec_chunks = to_sublists(self.spec, self.num)[::self.skip]
            dot_products = np.zeros((len(vec_chunks), len(self.results.t)))
            for i, vec_chunk in enumerate(vec_chunks):
                select = vec_chunk[::self.sampling]
                dot_products[i] = np.asarray([vecs.dot(select[0]) / (np.linalg.norm(vecs) * np.linalg.norm(select[0])) for vecs in select])
                update_progress(i / len(vec_chunks))

        n_chunks = len(frame_chunks) if self.traj is not None else len(vec_chunks)
        if mean:
            if self.traj is not None:
                dot_products = np.hstack(dot_products)
            else:
                dot_products = dot_products.T
                
        if l == 1:
            if mean:
                self.results.C_t = dot_products.mean(axis = 1)
                self.results.C_t_error = dot_products.std(axis = 1) / (n_chunks)**0.5
            else:
                self.results.C_t = dot_products.mean(axis = 0)
                self.results.C_t_error = dot_products.std(axis = 0) / (n_chunks)**0.5
        
        elif l == 2:
            if mean:
                self.results.C_t = ((3 * (dot_products)**2 - 1)/2).mean(axis = 1)
                self.results.C_t_error = ((3 * (dot_products)**2 - 1)/2).std(axis = 1) / (n_chunks)**0.5
            else:
                self.results.C_t = ((3 * (dot_products)**2 - 1)/2).mean(axis = 0)
                self.results.C_t_error = ((3 * (dot_products)**2 - 1)/2).std(axis = 0) / (n_chunks)**0.5

        elif l == 3:
            if mean:
                self.results.C_t = ((5 * (dot_products)**3 - 3 * (dot_products))/2).mean(axis = 1)
                self.results.C_t_error = (((5 * (dot_products)**3 - 3 * (dot_products))/2)).std(axis = 1) / (n_chunks)**0.5
            else:
                self.results.C_t = ((5 * (dot_products)**3 - 3 * (dot_products))/2).mean(axis = 0)
                self.results.C_t_error = (((5 * (dot_products)**3 - 3 * (dot_products))/2)).std(axis = 0) / (n_chunks)**0.5
        
        else:
            raise ValueError("l = 1, 2 or 3")

        def kww_func_fit(x, y, tau = 1, beta = 0.4, maxfev = 10000):
            """
            Fit the data with Kohlrausch-Willliams-Watts equation
            """
            def kww_func(t, tau, beta):
                return np.exp(-(t/tau)**beta)

            params,_ = curve_fit(kww_func, x, y, p0=[tau, beta], maxfev = maxfev)
            x_fit = np.linspace(x.min(), x.max(), num = 200)
            y_fit = kww_func(x_fit, *params)
            print("The fitted KWW function paramters are:\nalpha: {0}, beta: {1}".format(*params))
            return x_fit, y_fit, params
        
        if fit:
            self.results.t_fit = np.linspace(self.results.t.min(), self.results.t.max(), num = 200)
            if len(self.results.C_t.shape) == 1:
                self.results.C_t_fit = np.zeros(200)
                self.results.fit_params = np.zeros(2)
                _, self.results.C_t_fit, self.results.fit_params = kww_func_fit(self.results.t, self.results.C_t)
            elif self.results.C_t.shape[-1] == 1:
                self.results.C_t_fit = np.zeros(200)
                self.results.fit_params = np.zeros(2)
                _, self.results.C_t_fit, self.results.fit_params = kww_func_fit(self.results.t, self.results.C_t.flatten())
            else:
                self.results.C_t_fit = np.zeros((200, self.results.C_t.shape[-1]))
                self.results.fit_params = np.zeros((2, self.results.C_t.shape[-1]))
                for i in range(self.results.C_t.shape[-1]):
                    _, self.results.C_t_fit[:, i], self.results.fit_params[i] = kww_func_fit(self.results.t, self.results.C_t[:, i])
        
        update_progress(1)
        
        # delete temp variable to release some memory
        if 'frame_chunks' in locals(): del frame_chunks
        if 'vec_chunks' in locals(): del vec_chunks
        del dot_products, select
        
        # Plot the results    
        if plot:
            fig, ax = plt.subplots(figsize = (4.2, 3.6))
            if len(self.results.C_t.shape) == 1:
                ax.scatter(self.results.t, self.results.C_t, **kwargs)
                if fit:
                    ax.plot(self.results.t_fit, self.results.C_t_fit, color = "#525252", lw = 2)
            else:
                [ax.scatter(self.results.t, self.results.C_t[:,i], **kwargs) for i in range(self.results.C_t.shape[1])]
                if fit:
                    if len(self.results.C_t_fit.shape) == 1:
                        ax.plot(self.results.t_fit, self.results.C_t_fit, color = "#525252", lw = 2)
                    else:
                        [ax.plot(self.results.t_fit, self.results.C_t_fit[:,i], color = "#525252", lw = 2) for i in range(self.results.C_t.shape[1])]
            
            ax.set_xlabel(r"$t$ (ps)")
            ax.set_ylabel(f"$C^{l}_t$")
            
            if log_scale:
                plt.semilogy()
            plt.show()
            self.results.plot = fig
            
        return self.results


class VHCF(object):
    """
    Van Hove Correlation Function & Non-Gaussian Parameter Analysis
    ----------------------------------------------------------------------
    Arguments:
        traj      : Trajectory object containing .frames, .timestep, .nframes, .cell
        at_g1     : list of int, selected atom indices (Group 1)
        at_g2     : list of int, selected atom indices (Group 2, optional). 
                    If None, calculates the Self-Van Hove Function G_s(r,t) and alpha_2.
        r_bins    : int, number of bins for spatial resolution
        r_range   : list of float, [min_distance, max_distance]
        tau_sep   : int/float, time interval step size (in fs)
        tau_range : list of float, [min_time, max_time] for correlation (in ps)
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
                              "requires bulk density. Falling back to volume = 1.0 to prevent math collapse. "
                              "Contour shapes will be correct, but Y-axis magnitudes will be unscaled.")
            vol_scalar = 1.0
            
        self.volume = np.full(shape=self.settings.bins, fill_value=vol_scalar, dtype=np.float64)

    def calculate(self, plot=False, tau_sel=None, r_sel=None, plot_vs="r", zlim=None, levels=60, cmap=None, save_fig=None, **kwargs):
        N_pairs = len(self.pairs)
        bin_widths = np.diff(self.results.r_edges)
        
        if self.is_self:
            for i, tau in enumerate(self.results.tau):
                sep = int(round(1000 * tau / self.traj.timestep))
                n_frames_eval = self.traj.nframes - sep
                
                if n_frames_eval <= 0: continue

                dists = np.asarray([np.linalg.norm(self.traj.frames[j + sep].pos[self.pairs[:,1]] - self.traj.frames[j].pos[self.pairs[:,0]], axis=1) for j in range(n_frames_eval)])
                
                r2_mean = np.mean(dists**2)
                r4_mean = np.mean(dists**4)
                self.results.msd[i] = r2_mean
                
                if r2_mean > 1e-12:
                    self.results.alpha2[i] = (3.0 * r4_mean) / (5.0 * (r2_mean**2)) - 1.0
                else:
                    self.results.alpha2[i] = 0.0

                counts, _ = np.histogram(dists, **asdict(self.settings))
                
                self.results.rpd[i] = counts / (n_frames_eval * N_pairs * bin_widths)
                self.results.g_rt[i] = self.results.rpd[i] / (4 * np.pi * self.results.r**2)
                
        else:
            vols = 4/3 * np.pi * np.diff(np.power(self.results.r_edges, 3))
            density = N_pairs / self.volume[0] 
            
            for i, tau in enumerate(self.results.tau):
                sep = int(round(1000 * tau / self.traj.timestep))
                n_frames_eval = self.traj.nframes - sep
                
                if n_frames_eval <= 0: continue

                dists = np.asarray([np.linalg.norm(self.traj.frames[j + sep].pos[self.pairs[:,1]] - self.traj.frames[j].pos[self.pairs[:,0]], axis=1) for j in range(n_frames_eval)])
                counts, _ = np.histogram(dists, **asdict(self.settings))
                
                self.results.g_rt[i] = counts / (density * vols * n_frames_eval)
                self.results.rpd[i] = counts / (n_frames_eval * N_pairs * bin_widths)

        if plot:
            self.plot_probability_density(tau_sel=tau_sel, r_sel=r_sel, plot_vs=plot_vs, levels=levels, cmap=cmap, save_fig=save_fig)
            
        return self.results

    def _get_cmap(self, cmap):
        if cmap is None:
            from colour import Color
            from matplotlib.colors import LinearSegmentedColormap
            ramp_colors = ["#ffffff", "#9ecae1", "#2166ac", "#1a9850", "#ffff33", "#b2182b", "#67000d"]
            return LinearSegmentedColormap.from_list('my_list', [Color(c1).rgb for c1 in ramp_colors])
        return cmap

    def plot_probability_density(self, tau_sel=None, r_sel=None, plot_vs="r", levels=50, cmap=None, tau_lim=None, r_lim=None, saturation_factor=1.0, save_fig=None):
        """Visualizes the raw probability density function G(r,t)."""
        cmap = self._get_cmap(cmap)
        
        zmin, zmax = np.nanmin(self.results.g_rt), np.nanmax(self.results.g_rt)
        if zmin == zmax == 0.0: zmax = 1.0 
        vmax_val = zmin + saturation_factor * (zmax - zmin)

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
            CS = ax_contour.contourf(self.results.r, self.results.tau, self.results.g_rt, vmax=vmax_val, levels=levels, cmap=cmap)
            ax_contour.contour(self.results.r, self.results.tau, self.results.g_rt, colors="k", vmax=vmax_val, levels=levels, linewidths=0.1)
            ax_contour.set_xlabel(r"Distance ($\mathrm{\AA}$)")
            ax_contour.set_ylabel(r"$\tau$ (ps)")
        else:
            CS = ax_contour.contourf(self.results.tau, self.results.r, self.results.g_rt.T, vmax=vmax_val, levels=levels, cmap=cmap)
            ax_contour.contour(self.results.tau, self.results.r, self.results.g_rt.T, colors="k", vmax=vmax_val, levels=levels, linewidths=0.1)
            ax_contour.set_xlabel(r"$\tau$ (ps)")
            ax_contour.set_ylabel(r"Distance ($\mathrm{\AA}$)")

        ax_contour.set_title("Density G(r,t)")
        
        if r_lim is not None: ax_contour.set_ylim(*r_lim) if plot_vs == "tau" else ax_contour.set_xlim(*r_lim)
        if tau_lim is not None: ax_contour.set_xlim(*tau_lim) if plot_vs == "tau" else ax_contour.set_ylim(*tau_lim)

        cbar = plt.colorbar(CS, pad=0.035, location="right", ax=ax_contour)
        cbar.set_label(r"Density $G(r, \tau)$")
        
        if save_fig:
            fig.savefig(save_fig, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_radial_probability(self, tau_sel=None, r_sel=None, plot_vs="r", levels=50, cmap=None, tau_lim=None, r_lim=None, saturation_factor=1.0, save_fig=None):
        """Visualizes the Radial Probability Distribution (RPD = 4 * pi * r^2 * G(r,t))."""
        cmap = self._get_cmap(cmap)
        
        zmin, zmax = np.nanmin(self.results.rpd), np.nanmax(self.results.rpd)
        if zmin == zmax == 0.0: zmax = 1.0
        vmax_val = zmin + saturation_factor * (zmax - zmin)

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
            CS = ax_contour.contourf(self.results.r, self.results.tau, self.results.rpd, vmax=vmax_val, levels=levels, cmap=cmap)
            ax_contour.contour(self.results.r, self.results.tau, self.results.rpd, colors="k", vmax=vmax_val, levels=levels, linewidths=0.1)
            ax_contour.set_xlabel(r"Distance ($\mathrm{\AA}$)")
            ax_contour.set_ylabel(r"$\tau$ (ps)")
        else:
            CS = ax_contour.contourf(self.results.tau, self.results.r, self.results.rpd.T, vmax=vmax_val, levels=levels, cmap=cmap)
            ax_contour.contour(self.results.tau, self.results.r, self.results.rpd.T, colors="k", vmax=vmax_val, levels=levels, linewidths=0.1)
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
        """Visualizes the Mean Squared Displacement on a log-log scale."""
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
        """Computes and plots the Intermediate Scattering Function F(k,t)."""
        self.results.isf = np.zeros(len(self.results.tau))
        dr = np.diff(self.results.r_edges)
        
        for i in range(len(self.results.tau)):
            kr = k_val * self.results.r
            sinc = np.sin(kr) / kr
            sinc[np.isnan(sinc)] = 1.0 
            
            if self.is_self:
                self.results.isf[i] = np.sum(self.results.rpd[i] * sinc * dr)
            else:
                density = len(self.pairs) / self.volume[0]
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
        """Prints a comprehensive scientific interpretation of the calculated parameters and visualisations."""
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