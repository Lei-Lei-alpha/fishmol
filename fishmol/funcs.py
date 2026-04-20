"""Analysis functions for MD trajectories.

Provides distribution functions (RDF, ADF, DDF, CDF), vector reorientation
dynamics (VRD), and the Van Hove correlation function (VHCF) for post-processing
Trajectory objects produced by fishmol.trj.
"""

import numpy as np
import itertools
import warnings
from typing import List, Tuple, Union, Optional, Any, Sequence
from recordclass import make_dataclass, asdict
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from fishmol.utils import to_sublists, make_comb, update_progress, cart2xys, xys2cart
from fishmol import style
from scipy.optimize import curve_fit


# RDF
class RDF(object):
    """Radial distribution function g(r) between two atom groups.

    The RDF describes how atomic density varies as a function of distance from a
    reference atom. It is normalised to unity at large r in a homogeneous bulk liquid:

        g(r) = count(r) / [ nA * nB * (4π r² Δr / V) * nframes ]

    where nA and nB are the sizes of the two atom groups, V is the simulation box
    volume, Δr is the bin width, and nframes is the total number of trajectory frames.
    Pairwise distances are computed under the minimum image convention (MIC) so that
    periodic boundary conditions are correctly handled.

    Parameters
    ----------
    traj : Trajectory
        Trajectory object to analyse.
    at_g1 : list of int
        Atom indices forming the first selection (reference group).
    at_g2 : list of int
        Atom indices forming the second selection (neighbour group).
    nbins : int, optional
        Number of histogram bins (controls radial resolution). Default: 200.
    range : tuple of float, optional
        (r_min, r_max) in Å over which g(r) is computed. Default: (0.0, 15.0).

    Results dataclass fields (populated after calling calculate())
    -------------------------------------------------------------
    x        : ndarray, shape (nbins,) — bin centre radii in Å.
    rdf      : ndarray, shape (nbins,) — g(r) values.
    edges    : ndarray, shape (nbins+1,) — bin edge positions in Å.
    count    : ndarray, shape (nbins,) — raw pair count histogram.
    pairs    : list of (int, int) — all (g1, g2) index pairs included.
    scaler   : ndarray, shape (nframes, npairs) — per-frame pair distances in Å;
               used by com_plot to show the time evolution of individual pairs.
    t        : ndarray, shape (nframes,) — time axis in ps.
    plot     : Figure — g(r) line plot (set when plot=True).
    com_plot : Figure — combined time-series + g(r) panel (set when com_plot=True).
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
        cell = np.asarray(self.traj.cell)
        self.volume = float(cell[0].dot(np.cross(cell[1], cell[2])))
        
    def calculate(self, plot: bool = False, com_plot: bool = False, **kwargs: Any) -> Any:
        """Compute g(r) and optionally plot the result.

        Parameters
        ----------
        plot : bool, optional
            If True, display a g(r) vs r line plot. Default: False.
        com_plot : bool, optional
            If True, display a combined panel showing the time evolution of each
            pair distance alongside the g(r) curve. Default: False.
        **kwargs
            Additional keyword arguments forwarded to matplotlib plot calls
            (e.g. color, linewidth, label).

        Returns
        -------
        results : dataclass
            Populated results object; see class docstring for field descriptions.
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
    """Angular distribution function g(α) for three-body angles.

    The ADF describes the probability distribution of the bond angle α formed by
    the triplet g1–g2–g3, where g2 is the central atom group. All angles are
    computed under the minimum image convention.

    When cone_correction=True (default), the raw histogram is divided by sin(α)
    to account for the increasing area of the solid-angle shell at angle α:

        g(α) = count(α) / [ nA * nB * nC * nframes * sin(α) ]

    Without the correction, the distribution is biased toward larger angles simply
    because there is more solid-angle available. The corrected g(α) integrates to a
    constant, allowing direct comparison between different systems.

    Parameters
    ----------
    traj : Trajectory
        Trajectory object to analyse.
    at_g1 : list of int
        Atom indices of the first terminal group (one end of the angle).
    at_g2 : list of int
        Atom indices of the central group (vertex of the angle).
    at_g3 : list of int
        Atom indices of the second terminal group (other end of the angle).
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
    pairs    : array — atom index triplets included in the calculation.
    scaler   : ndarray, shape (nframes, npairs) — per-frame angle values in degrees.
    t        : ndarray, shape (nframes,) — time axis in ps.
    plot     : Figure — g(α) line plot (set when plot=True).
    com_plot : Figure — combined time-series + g(α) panel (set when com_plot=True).
    """
    def __init__(self, traj: Any, at_g1: Any, at_g2: Any, at_g3: Any, nbins: int = 200, range: Tuple[float, float] = (0.0, 180.0)) -> None:
        self.g1 = at_g1
        self.g2 = at_g2
        self.g3 = at_g3
        self.traj = traj
        settings = make_dataclass("Settings", "bins range")
        self.settings = settings(nbins, range)
        _, edges = np.histogram([-1], **asdict(self.settings))
        results = make_dataclass("Results", "edges count x adf plot t pairs scaler com_plot")
        self.results = results
        self.results.edges = edges
        self.results.x = (edges[:-1] + edges[1:]) / 2
        self.results.count = np.zeros(self.settings.bins)
        self.results.t = np.linspace(0, self.traj.timestep * (self.traj.nframes - 1)/1000, self.traj.nframes)
        
    def calculate(self, cone_correction: bool = True, plot: bool = False, com_plot: bool = False, **kwargs: Any) -> Any:
        """Compute g(α) and optionally plot the result.

        Parameters
        ----------
        cone_correction : bool, optional
            Divide the raw histogram by sin(α) to correct for solid-angle geometry.
            Should be True (default) for physically meaningful angular distributions.
            Set to False only if you need the raw, uncorrected histogram shape.
        plot : bool, optional
            If True, display a g(α) vs α line plot. Default: False.
        com_plot : bool, optional
            If True, display a combined panel showing the angle time series alongside
            the g(α) curve. Default: False.
        **kwargs
            Additional keyword arguments forwarded to matplotlib plot calls.

        Returns
        -------
        results : dataclass
            Populated results object; see class docstring for field descriptions.
        """
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
            # Divide by sin(θ) to correct for the solid-angle shell (2π sinθ dθ).
            # np.maximum guards against near-zero values at θ ≈ 0° and θ ≈ 180°.
            sin_theta = np.maximum(np.sin(self.results.x * np.pi / 180), 1e-10)
            self.results.adf = self.results.count / (NA * NB * NC * self.traj.nframes * sin_theta)
        else:
            self.results.adf = self.results.count / (NA * NB * NC * self.traj.nframes)
        
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
    """Dihedral distribution function g(δ) for torsion angles.

    The DDF describes the probability distribution of the torsion angle δ defined
    by four consecutive bonded atoms A–B–C–D. The torsion angle is the angle
    between the plane containing A–B–C and the plane containing B–C–D, measured
    about the central B–C bond axis.

    Unlike the ADF, no sin(δ) correction is applied: dihedral angles are azimuthal
    quantities (uniform geometric probability) and their distribution is directly
    proportional to the raw histogram counts.

        g(δ) = count(δ) / [ N_dihedrals * nframes ]

    where N_dihedrals is the number of unique four-atom torsion combinations and
    nframes is the total number of trajectory frames. All dihedral angles are
    computed under the minimum image convention.

    Parameters
    ----------
    traj : Trajectory
        Trajectory object to analyse.
    at_g : list of list of int
        Atom index groups defining the four-body torsion combinations. The groups
        are passed directly to ``Atoms.dihedrals()``.
    nbins : int, optional
        Number of histogram bins. Default: 200.
    range : tuple of float, optional
        (δ_min, δ_max) in degrees. Default: (0.0, 180.0).

    Results dataclass fields (populated after calling calculate())
    -------------------------------------------------------------
    x        : ndarray, shape (nbins,) — bin centre angles in degrees.
    ddf      : ndarray, shape (nbins,) — g(δ) values.
    edges    : ndarray, shape (nbins+1,) — bin edge angles in degrees.
    count    : ndarray, shape (nbins,) — accumulated raw torsion histogram.
    angles   : ndarray, shape (nframes, ndihedrals) — per-frame torsion values in degrees.
    t        : ndarray, shape (nframes,) — time axis in ps.
    plot     : Figure — g(δ) line plot (set when plot=True).
    com_plot : Figure — combined time-series + g(δ) panel (set when com_plot=True).
    """
    def __init__(self, traj: Any, at_g: Any, nbins: int = 200, range: Tuple[float, float] = (0.0, 180.0)) -> None:
        self.g = at_g
        self.traj = traj
        settings = make_dataclass("Settings", "bins range")
        self.settings = settings(nbins, range)
        _, edges = np.histogram([-1], **asdict(self.settings))
        results = make_dataclass("Results", "edges count x ddf plot t angles com_plot")
        self.results = results
        self.results.edges = edges
        self.results.x = (edges[:-1] + edges[1:]) / 2
        self.results.count = np.zeros(self.settings.bins)
        self.results.t = np.linspace(0, self.traj.timestep * (self.traj.nframes - 1)/1000, self.traj.nframes)
        
    def calculate(self, plot: bool = False, com_plot: bool = False, **kwargs: Any) -> Any:
        """Compute g(δ) and optionally plot the result.

        Parameters
        ----------
        plot : bool, optional
            If True, display a g(δ) vs δ line plot. Default: False.
        com_plot : bool, optional
            If True, display a combined panel showing the torsion angle time series
            alongside the g(δ) curve. Default: False.
        **kwargs
            Additional keyword arguments forwarded to matplotlib plot calls.

        Returns
        -------
        results : dataclass
            Populated results object; see class docstring for field descriptions.
        """
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
    """Combined (joint) distribution function correlating two scalar observables.

    The CDF is a 2D histogram that reveals the joint probability of two structural
    quantities occurring simultaneously across all trajectory frames. A typical use
    case is correlating a pair distance (from an RDF) with a bond angle (from an ADF)
    to expose structural correlations that are invisible in the individual 1D
    distributions. For example, a CDF of O···O distance vs O–H···O angle can pinpoint
    the geometry of hydrogen-bonded configurations.

    The CDF takes two already-computed Results objects (e.g., from RDF and ADF) and
    their flat per-frame observable arrays (stored in ``results.scaler``) as input.
    The 2D histogram bins are taken directly from the edges of those Results objects,
    so the spatial and angular axes are automatically consistent with the prior 1D
    analysis.

    Parameters
    ----------
    scaler1 : Results dataclass
        Results object from a prior RDF, ADF, or DDF calculation. Must contain
        ``edges`` and ``scaler`` attributes, plus the attribute named by ``names[0]``
        (e.g., ``rdf`` or ``adf``) for the marginal plot.
    scaler2 : Results dataclass
        Results object for the second observable, with the same requirements.
    names : list of str, optional
        Two-element list giving the attribute names of the 1D distributions to plot
        in the marginal panels (e.g., ``["rdf", "adf"]``). Required when using the
        default range or when plot=True.
    range : tuple of [float, float], optional
        ``([x_min, x_max], [y_min, y_max])`` defining the extent of the 2D histogram.
        Defaults to ``([0, scaler1.max], [0, scaler2.max])``.

    Results dataclass fields (populated after calling calculate())
    -------------------------------------------------------------
    cdf     : ndarray, shape (nbins_y, nbins_x) — 2D joint histogram counts (transposed
              so the first axis corresponds to the y-axis observable).
    xedges  : ndarray — bin edges for the x-axis observable (from scaler1).
    yedges  : ndarray — bin edges for the y-axis observable (from scaler2).
    plot    : Figure — 2D contour plot with marginal 1D panels (set when plot=True).
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
        """Compute the 2D joint histogram and optionally plot the result.

        Parameters
        ----------
        plot : bool, optional
            If True (default), display a filled contour plot with the 1D marginal
            distributions of each observable in side panels.
        **kwargs
            Additional keyword arguments forwarded to ``np.histogram2d``
            (e.g. normed, weights).

        Returns
        -------
        results : dataclass
            Populated results object; see class docstring for field descriptions.
        """
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
    """Vector Reorientation Dynamics via Legendre polynomial autocorrelation.

    The VRD quantifies how quickly a molecular vector (e.g. an O–H bond, a C–C
    axis) loses memory of its initial orientation. It computes the time
    autocorrelation function of the l-th order Legendre polynomial:

        C_l(t) = < P_l( u(0) · u(t) ) >

    where u(t) is the unit vector at time t and the angle brackets denote an
    ensemble average over all vectors and all time origins. The three polynomials
    available are:

        P_1(x) = x                         (l=1, related to IR absorption)
        P_2(x) = (3x² − 1) / 2            (l=2, related to NMR T₁ and Raman)
        P_3(x) = (5x³ − 3x) / 2           (l=3, related to higher-order spectra)

    C_l(t) decays from 1 at t=0 toward 0 as the vector randomises. The decay is
    fitted with the Kohlrausch–Williams–Watts (KWW) stretched exponential:

        C(t) = exp( −(t / τ)^β )

    where τ is the relaxation time (ps) and β ∈ (0, 1] is the stretching exponent
    (β=1 reduces to a single exponential; β<1 indicates heterogeneous dynamics).

    The trajectory is divided into non-overlapping chunks of ``num`` frames. Within
    each chunk, the autocorrelation is computed using the first frame as the time
    origin, then the results are averaged over all chunks. Reducing ``skip`` uses
    more chunks and reduces statistical error at the cost of correlation between
    samples.

    Parameters
    ----------
    traj : Trajectory, optional
        Trajectory object to analyse. Provide either ``traj`` + ``spec`` or
        a pre-computed ``spec`` array.
    spec : list of [list of int, list of int] or ndarray
        When ``traj`` is given: a two-element list ``[g1_indices, g2_indices]``
        defining the atom pairs whose inter-atomic vector is tracked.
        When ``traj`` is None: a 2D ndarray of shape (nframes, 3) containing
        the pre-computed vector time series.
    timestep : float, optional
        Trajectory timestep in fs. Required when ``traj`` is None; ignored
        otherwise (taken from ``traj.timestep``).
    num : int, optional
        Number of frames per analysis chunk. Determines the maximum lag time:
        t_max = timestep * (num − 1) / 1000 ps. Default: 2000.
    sampling : int, optional
        Frame stride within each chunk. A value of 5 uses every 5th frame,
        reducing memory use and computation while coarsening time resolution.
        Default: 5.
    skip : int, optional
        Chunk stride: only every ``skip``-th chunk is used. Increasing ``skip``
        reduces correlations between chunks at the cost of statistical precision.
        Default: 10.

    Results dataclass fields (populated after calling calculate())
    -------------------------------------------------------------
    t          : ndarray, shape (num//sampling,) — lag time axis in ps.
    C_t        : ndarray — ensemble-averaged C_l(t) values.
    C_t_error  : ndarray — standard error of the mean across chunks.
    t_fit      : ndarray, shape (200,) — dense time axis for the fitted curve
                 (set when fit=True).
    C_t_fit    : ndarray — KWW fitted values on t_fit (set when fit=True).
    fit_params : ndarray — fitted [τ, β] parameters (set when fit=True).
    plot       : Figure — C_l(t) scatter plot, optionally with the KWW fit curve
                 (set when plot=True).
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
        """Compute C_l(t) and optionally fit and plot the result.

        Parameters
        ----------
        l : int, optional
            Order of the Legendre polynomial (1, 2, or 3). Default: 3.
            l=2 is the most commonly reported value (NMR, Raman);
            l=1 gives the slowest-decaying autocorrelation.
        mean : bool, optional
            If True (default), average C_l(t) over all chunks and vector pairs,
            producing a single decay curve with an error estimate.
            If False, return one curve per chunk.
        fit : bool, optional
            If True, fit C_l(t) with the KWW stretched exponential and store the
            fitted curve and parameters in the results. Default: False.
        plot : bool, optional
            If True (default), display C_l(t) as a scatter plot, with the KWW fit
            overlaid if fit=True.
        log_scale : bool, optional
            If True, display the y-axis on a logarithmic scale. Useful for
            identifying multi-exponential or power-law decay behaviour. Default: False.
        **kwargs
            Additional keyword arguments forwarded to ``ax.scatter`` (e.g. color, s).

        Returns
        -------
        results : dataclass
            Populated results object; see class docstring for field descriptions.
        """
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
                # Compute reference vectors once; reuse for all frames in this chunk.
                # (A * B).sum(axis=1) is O(N) vs np.diagonal(A @ B.T) which is O(N²).
                vecs_0 = select[0].vecs(combs=combs, absolute=False, normalise=True, mic=True)
                dot_products[i, :] = np.stack([
                    (frame.vecs(combs=combs, absolute=False, normalise=True, mic=True) * vecs_0).sum(axis=1)
                    for frame in select
                ])
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
            """Fit C_l(t) with the KWW stretched exponential exp(−(t/τ)^β)."""
            def kww_func(t, tau, beta):
                return np.exp(-(t/tau)**beta)

            params,_ = curve_fit(kww_func, x, y, p0=[tau, beta], maxfev = maxfev)
            x_fit = np.linspace(x.min(), x.max(), num = 200)
            y_fit = kww_func(x_fit, *params)
            print("The fitted KWW function parameters are:\ntau: {0:.4f} ps, beta: {1:.4f}".format(*params))
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
                              "requires bulk density. Falling back to volume = 1.0 to prevent math collapse. "
                              "Contour shapes will be correct, but Y-axis magnitudes will be unscaled.")
            vol_scalar = 1.0
            
        self.volume = np.full(shape=self.settings.bins, fill_value=vol_scalar, dtype=np.float64)

    def calculate(self, plot=False, tau_sel=None, r_sel=None, plot_vs="r", zlim=None, levels=60, cmap=None, save_fig=None, **kwargs):
        """Compute G(r, τ), RPD, α₂, and MSD, then optionally plot.

        Parameters
        ----------
        plot : bool, optional
            If True, call ``plot_probability_density`` after the calculation.
            Default: False.
        tau_sel : list of float, optional
            Selected τ values in ps for which a 1D slice through G(r, τ) is plotted
            alongside the 2D contour (used when plot=True and plot_vs="r").
        r_sel : list of float, optional
            Selected r values in Å for which a 1D slice through G(r, τ) is plotted
            alongside the 2D contour (used when plot=True and plot_vs="tau").
        plot_vs : {"r", "tau"}, optional
            Axis orientation for the 2D contour: "r" (default) puts distance on
            the x-axis and time on the y-axis; "tau" swaps the axes.
        zlim : ignored
            Reserved for future use.
        levels : int, optional
            Number of contour levels in the 2D plot. Default: 60.
        cmap : matplotlib colormap, optional
            Colormap for the 2D contour plot. Defaults to a built-in diverging ramp.
        save_fig : str or None, optional
            If provided, save the figure to this file path at 300 dpi.

        Returns
        -------
        results : dataclass
            Populated results object; see class docstring for field descriptions.
        """
        N_pairs = len(self.pairs)
        bin_widths = np.diff(self.results.r_edges)
        
        if self.is_self:
            # Self-part tracks the same atom over time: MIC must NOT be applied here
            # because it would alias large displacements that cross the box boundary.
            # The trajectory must use unwrapped (non-wrapped) coordinates for correct
            # MSD and non-Gaussian parameter values. Call traj.calib() but NOT wrap2box().
            warnings.warn(
                "VHCF self-part requires unwrapped coordinates. "
                "Do not call wrap2box() before this calculation; "
                "use calib() only so atoms are free to diffuse beyond one box length.",
                stacklevel=2,
            )
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
            cell = self.traj.frames[0].cell

            # Vectorised MIC distance for (N,3) position arrays — defined once.
            def _mic_norm(pos_a, pos_b):
                d = cart2xys(pos_b - pos_a, cell)
                d -= np.round(d)
                return np.linalg.norm(xys2cart(d, cell), axis=1)

            for i, tau in enumerate(self.results.tau):
                sep = int(round(1000 * tau / self.traj.timestep))
                n_frames_eval = self.traj.nframes - sep

                if n_frames_eval <= 0: continue

                dists = np.asarray([_mic_norm(
                    self.traj.frames[j].pos[self.pairs[:, 0]],
                    self.traj.frames[j + sep].pos[self.pairs[:, 1]],
                ) for j in range(n_frames_eval)])
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
        """Plot the Van Hove correlation function G(r, τ) as a 2D contour.

        For the self-part, G_s peaks sharply at r = 0 at short τ and broadens as
        atoms diffuse. For the distinct-part, G_d shows structural peaks at τ = 0
        that flatten toward 1 as the liquid structure decorrelates over time.

        Parameters
        ----------
        tau_sel : list of float, optional
            τ values in ps for which a 1D G(r) slice is plotted (requires plot_vs="r").
        r_sel : list of float, optional
            r values in Å for which a 1D G(τ) slice is plotted (requires plot_vs="tau").
        plot_vs : {"r", "tau"}, optional
            Axis orientation: "r" puts distance on the x-axis (default); "tau" swaps.
        levels : int, optional
            Number of contour levels. Default: 50.
        cmap : matplotlib colormap, optional
            Colormap; defaults to the built-in diverging ramp.
        tau_lim : tuple of float, optional
            (τ_min, τ_max) axis limits in ps.
        r_lim : tuple of float, optional
            (r_min, r_max) axis limits in Å.
        saturation_factor : float, optional
            Fraction of the colour range to use; values < 1 saturate the high end
            and reveal fine structure at low density. Default: 1.0.
        save_fig : str or None, optional
            File path to save the figure at 300 dpi.
        """
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
        """Plot the radial probability distribution RPD(r, τ) = 4π r² G(r, τ).

        RPD is the probability density weighted by shell volume. Unlike G(r, τ),
        which diverges at r = 0 for the self-part, RPD goes to zero at r = 0 and
        peaks at the most probable displacement distance. Integrating RPD over r
        gives 1 for the self-part at any τ. Secondary peaks at intermediate τ values
        are a signature of discrete atomic hopping between lattice sites.

        Parameters
        ----------
        tau_sel : list of float, optional
            τ values in ps for 1D RPD(r) slices (requires plot_vs="r").
        r_sel : list of float, optional
            r values in Å for 1D RPD(τ) slices (requires plot_vs="tau").
        plot_vs : {"r", "tau"}, optional
            Axis orientation. Default: "r".
        levels : int, optional
            Number of contour levels. Default: 50.
        cmap : matplotlib colormap, optional
            Colormap; defaults to the built-in diverging ramp.
        tau_lim : tuple of float, optional
            (τ_min, τ_max) axis limits in ps.
        r_lim : tuple of float, optional
            (r_min, r_max) axis limits in Å.
        saturation_factor : float, optional
            Fraction of the colour range to display. Default: 1.0.
        save_fig : str or None, optional
            File path to save the figure at 300 dpi.
        """
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
        """Plot the mean squared displacement ⟨Δr²⟩(τ) on a log–log scale.

        Only available for the self-part (``at_g2=None``). The log–log slope reveals
        the diffusion regime: slope ≈ 2 indicates ballistic (inertial) motion at very
        short times; slope ≈ 0 indicates caging; slope = 1 indicates Fickian diffusion.

        Parameters
        ----------
        save_fig : str or None, optional
            File path to save the figure at 300 dpi.
        """
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
        """Compute and plot the intermediate scattering function F(k, τ).

        The ISF is the spatial Fourier transform of G(r, τ) at wavevector magnitude k,
        directly comparable to incoherent (self-part) or coherent (distinct-part)
        neutron and X-ray scattering data:

            F_s(k, τ) = ∫ RPD(r, τ) · sin(kr)/(kr) dr    (self-part)
            F_d(k, τ) = ∫ 4π r² ρ [G_d(r, τ) − 1] · sin(kr)/(kr) dr   (distinct-part)

        For the self-part, F_s decays from 1 to 0; the time at which F_s = 1/e defines
        the alpha-relaxation time τ_α. A two-step decay indicates glassy dynamics.

        Parameters
        ----------
        k_val : float, optional
            Wavevector magnitude in Å⁻¹ at which the ISF is evaluated. Choose a value
            near the first peak of the static structure factor S(k) for maximum
            sensitivity to structural relaxation. Default: 2.5 Å⁻¹.
        save_fig : str or None, optional
            File path to save the figure at 300 dpi.
        """
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
        """Print a scientific interpretation of all computed observables.

        Summarises the calculation mode, time and spatial ranges, and interprets the
        non-Gaussian parameter α₂ peak for the self-part. Also describes the physical
        meaning of each visualisation method to guide analysis.
        """
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

    The SDF computes the normalised probability density of any scalar quantity δ
    extracted from a trajectory — for example a bond length, torsion angle,
    coordination number, or an energy-like observable.  It is defined as

        g(δ) = count(δ) / (N · Δδ)

    where N is the total number of observations and Δδ is the bin width, so that
    ∫ g(δ) dδ = 1.

    This is a general-purpose wrapper around :func:`numpy.histogram` that attaches
    the result to a results dataclass and optionally produces a publication-ready
    line plot or a combined time-series / distribution panel.

    Parameters
    ----------
    traj : Trajectory
        Trajectory object.  Used only to derive the time axis for the optional
        time-series panel; the scalar observations are passed separately via
        *scaler*.
    scaler : array-like, shape (N,)
        Pre-computed scalar observations.  Typically one value per trajectory
        frame, but any 1-D sequence is accepted.
    nbins : int, optional
        Number of histogram bins (controls resolution). Default: 200.
    range : tuple of float, optional
        (δ_min, δ_max) over which the SDF is computed.  If *None* the data
        range [min(scaler), max(scaler)] is used automatically.

    Results dataclass fields (populated after calling calculate())
    -------------------------------------------------------------
    x        : ndarray, shape (nbins,) — bin-centre values of δ.
    sdf      : ndarray, shape (nbins,) — normalised probability density g(δ).
    count    : ndarray, shape (nbins,) — raw histogram counts.
    edges    : ndarray, shape (nbins+1,) — bin edges.
    t        : ndarray — time axis in ps derived from traj.timestep.
    scaler   : ndarray — the scalar observations as supplied.
    plot     : Figure or None — line-plot figure if plot=True was requested.
    com_plot : Figure or None — combined time-series/SDF figure if com_plot=True.

    Examples
    --------
    >>> oh_dists = [frame.dist(0, 1, mic=True) for frame in traj.frames]
    >>> sdf = SDF(traj, scaler=oh_dists, nbins=150, range=(0.5, 3.5))
    >>> results = sdf.calculate(plot=True, x_label=r"$r_{OH}$ (Å)")
    """

    def __init__(self, traj, scaler, nbins=200, range=None):
        self.traj = traj
        scaler_arr = np.asarray(scaler, dtype=float)
        if range is None:
            range = (float(scaler_arr.min()), float(scaler_arr.max()))
        settings_dc = make_dataclass("Settings", "bins range")
        self.settings = settings_dc(nbins, range)
        results_dc = make_dataclass("Results", "edges count x sdf plot t scaler com_plot")
        self.results = results_dc
        self.results.scaler = scaler_arr
        self.results.t = np.linspace(
            0,
            self.traj.timestep * (self.traj.nframes - 1) / 1000,
            self.traj.nframes,
        )
        self.results.plot = None
        self.results.com_plot = None

    def calculate(self, plot=False, x_label=None, y_label=None, com_plot=False, **kwargs):
        """Compute the scalar distribution function and optionally plot results.

        Parameters
        ----------
        plot : bool, optional
            If True, produce a line plot of g(δ) vs δ.
        x_label : str, optional
            x-axis label for the line plot and time-series panel.
            Default: ``r"$\\delta$"``.
        y_label : str, optional
            y-axis label (g(δ) axis). Default: ``r"$g(\\delta)$"``.
        com_plot : bool, optional
            If True, produce a combined figure with a time-series panel on the
            left and a rotated SDF panel on the right, sharing the δ axis.
        **kwargs
            Additional keyword arguments forwarded to
            :func:`matplotlib.pyplot.plot`.

        Returns
        -------
        results : dataclass
            Populated results dataclass (see class docstring).

        Notes
        -----
        The distribution is normalised so that ∫ g(δ) dδ = 1.  Dividing the
        raw counts by N · Δδ gives a true probability density rather than a
        count histogram, making results comparable across different bin widths
        and dataset sizes.

        Interpretation
        --------------
        - The peak position of g(δ) identifies the most probable value of the
          observable — e.g., the preferred bond length or torsion angle.
        - The width (FWHM) reflects the thermal fluctuations: broader peaks
          indicate a more flexible degree of freedom.
        - For bond lengths, compare the peak position with crystallographic values
          to assess force-field accuracy.
        - For torsion angles, distinct peaks at ±60°/180° indicate gauche/trans
          populations characteristic of chain conformation statistics.
        """
        count, edges = np.histogram(self.results.scaler, **asdict(self.settings))
        self.results.edges = edges
        self.results.count = count
        self.results.x = (edges[:-1] + edges[1:]) / 2
        bin_width = edges[1] - edges[0]
        N = int(self.results.scaler.shape[0])
        self.results.sdf = count / (N * bin_width)

        if plot:
            fig, ax = plt.subplots(figsize=(4.2, 3.6))
            ax.plot(self.results.x, self.results.sdf, **kwargs)
            ax.set_xlabel(r"$\delta$" if x_label is None else x_label)
            ax.set_ylabel(r"$g(\delta)$" if y_label is None else y_label)
            plt.tight_layout()
            plt.show()
            self.results.plot = fig

        if com_plot:
            t = self.results.t
            scaler = self.results.scaler
            if len(scaler) != len(t):
                warnings.warn(
                    "Length of scaler array does not match traj.nframes; "
                    "the time-series panel may be misleading."
                )
            n_pts = min(len(scaler), len(t))
            fig = plt.figure(figsize=(4.8, 3.6))
            ax = fig.add_axes([0.20, 0.16, 0.56, 0.75])
            ax.plot(t[:n_pts], scaler[:n_pts], **kwargs)
            ax.set_xlabel(r"$t$ (ps)")
            ax.set_ylabel(r"$\delta$" if x_label is None else x_label)

            divider = make_axes_locatable(ax)
            ax_hist = divider.append_axes("right", 0.5, pad=0.05, sharey=ax)
            ax_hist.yaxis.tick_right()
            ax_hist.plot(self.results.sdf, self.results.x, **kwargs)
            ax_hist.set_xlabel(r"$g(\delta)$" if y_label is None else y_label)
            plt.show()
            self.results.com_plot = fig

        peak_idx = np.argmax(self.results.sdf)
        mean_val = np.average(self.results.x, weights=self.results.sdf)

        print("")
        print(" Scalar Distribution Function (SDF)")
        print("=" * 50)
        print(f"  Observations  : {N}")
        print(f"  Bins          : {self.settings.bins}")
        print(f"  Range         : [{self.settings.range[0]:.4g}, {self.settings.range[1]:.4g}]")
        print(f"  Peak at δ     : {self.results.x[peak_idx]:.4g}")
        print(f"  Weighted mean : {mean_val:.4g}")
        print("=" * 50)
        print("")
        print(" Interpretation:")
        print("  - Peak position: the most probable value of the observable.")
        print("  - Peak width (FWHM): reflects thermal fluctuations; broader")
        print("    peaks indicate a more flexible degree of freedom.")
        print("  - Weighted mean: centre of mass of the distribution,")
        print("    equal to <δ> averaged over all observations.")
        print("=" * 50)

        return self.results