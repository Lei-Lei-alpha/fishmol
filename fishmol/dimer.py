import re
import json
import warnings
import itertools
import numpy as np
from fishmol import trj, data
from recordclass import make_dataclass, dataobject, astuple, asdict
from iteration_utilities import deepflatten
import matplotlib.pyplot as plt
from colour import Color
from matplotlib.colors import LinearSegmentedColormap
from fishmol.utils import to_sublists, update_progress, mic_dist, retrieve_symbol
from scipy.optimize import curve_fit
from scipy.integrate import quad
from matplotlib.ticker import MultipleLocator

plt.rcParams.update({
    "font.size": 9,
    "font.family": "sans-serif",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.figsize": (4.2,3.6),
    "figure.subplot.left": 0.21,
    "figure.subplot.right": 0.96,
    "figure.subplot.bottom": 0.18,
    "figure.subplot.top": 0.93,
    "legend.frameon": False,
})

ramp_colors = ["#ffffff", "#9ecae1", "#2166ac", "#1a9850", "#ffff33", "#b2182b", "#67000d"]
color_ramp = LinearSegmentedColormap.from_list( 'my_list', [ Color( c1 ).rgb for c1 in ramp_colors ] )

class dimers(object):
    """Frame-by-frame dimer pair detection from geometry criteria.

    A *dimer* is defined as any pair of atoms (or atom groups) whose
    inter-atomic distance — and optionally the D–H···A angle for hydrogen-bond
    donors — satisfies user-supplied geometry thresholds in a given trajectory
    frame.  The class covers three use cases:

    1. **Simple atom-pair dimers** — two lists of atom indices; distance only.
    2. **Centre-of-mass dimers** — two lists of molecule indices; CoM distance.
    3. **Hydrogen-bond dimers** — donor list formatted as [D, [H1, H2, ...]];
       filtered by both D···A distance and ∠D–H···A angle.

    After calling :meth:`make_pairs` (to enumerate candidate pairs) and
    :meth:`pair_filter` (to apply geometry criteria), the per-frame dimer list
    is stored in ``results.dimers`` as a list of dicts and optionally serialised
    to a JSON file for use with :class:`DLDDF` or :func:`auto_cor`.

    Parameters
    ----------
    traj : Trajectory
        Trajectory object to analyse.
    at_g1 : list
        First atom group.  For hydrogen-bond analysis pass a list of
        ``[D_idx, [H1_idx, H2_idx, ...]]`` donor entries.  For simple distance
        or CoM analysis pass a flat list of integer indices.
    at_g2 : list of int
        Indices of acceptor atoms or the second molecule group.
    criteria : list of [distance_criterion, angle_criterion], optional
        ``distance_criterion`` — one of:

        * ``"1.05vdw_sum"`` / ``"1.1coval_sum"`` — scaled sum of van der
          Waals / covalent radii (only valid when ``com=False``).
        * A float giving the cutoff distance in Å.

        ``angle_criterion`` — minimum ∠D–H···A in degrees (float), or a
        flexible string like ``"120f"`` that scales the threshold with the
        instantaneous D–H bond length.  Pass ``None`` if no angle filter is
        required.  Default: ``[None, None]``.
    com : bool, optional
        If True, compute the inter-group distance as the distance between
        centres of mass.  Default: False.

    Results dataclass fields (populated after calling pair_filter())
    ---------------------------------------------------------------
    pairs  : ndarray — all candidate pairs enumerated by make_pairs().
    dimers : list of dict — one entry per frame, each containing:
             ``"frame"`` (int), ``"pairs"`` (list) and ``"dists"`` (list)
             for distance-only mode; or ``"d_h_a_pairs"``, ``"d_a_dists"``,
             and ``"d_h_a_angles"`` for H-bond mode.

    Examples
    --------
    >>> # H-bond analysis: oxygen donors with their hydrogen atoms
    >>> at_g1 = [[0, [1, 2]], [8, [9, 10]]]   # two water O–H donors
    >>> at_g2 = [0, 8, 16]                     # acceptor oxygens
    >>> hb = dimers(traj, at_g1, at_g2, criteria=["1.02vdw_sum", "120f"])
    >>> hb.make_pairs()
    >>> hb.pair_filter(mic=True, filename="hbonds.json")
    """
    def __init__(self, traj, at_g1, at_g2, criteria = [None, None], com = False):
        self.traj = traj.wrap2box()
        self.g1 = at_g1
        self.g2 = at_g2
        criteria_dc = make_dataclass("criteria", "distance angle")
        self.criteria = criteria_dc(*criteria)
        if self.criteria.distance is None:
            warnings.warn("Distance criteria is mandatory! The dimer will be resolved by default distance settings! To ensure the accuracy of dimer resolvation, specify the geometry criteria by passing a distance (and an angle if applicable)")
            self.criteria.distance = "1.05vdw_sum"
        self.com = com
        results = make_dataclass("Results", "pairs dimers")
        self.results = results
                  
    def make_pairs(self):
        """Enumerate all candidate pairs from the two atom groups.

        Generates the Cartesian product of ``at_g1`` × ``at_g2``, removes
        self-pairs (where the same atom appears in both groups), and removes
        duplicate reverse pairs (i, j) / (j, i) when the two groups overlap.
        The result is stored in ``results.pairs`` as an integer ndarray of
        shape (n_pairs, 2) for distance-only mode, or (n_pairs, 3) for
        H-bond mode (columns: D, H, A).
        """
        # com = self.com
        # When calculates the distance of centre of masses of two molecules groups
        if self.com:
            self.results.pairs = list(map(list, [x for x in itertools.product(self.g1, self.g2) if x[0] != x[1]]))
            # Remove any duplicates if at_g1 and at_g2 are the same
            for pair1 in self.results.pairs:
                for pair2 in self.results.pairs:
                    if pair1[0] == pair2[1] and pair1[1] == pair2[0]:
                        self.results.pairs.remove(pair2)
            self.results.pairs = np.asarray(self.results.pairs)
        # Otherwise
        else:
            # simple case: make pairs from the the listed atoms in at_g1 and at_g2
            if self.criteria.angle is None:
                if all([isinstance(idx, int) for idx in self.g1]):
                    self.results.pairs = list(map(list, [x for x in itertools.product(self.g1, self.g2) if x[0] != x[1]]))
                    # Remove any duplicates if at_g1 and at_g2 are the same
                    for pair1 in self.results.pairs:
                        for pair2 in self.results.pairs:
                            if pair1[0] == pair2[1] and pair1[1] == pair2[0]:
                                self.results.pairs.remove(pair2)
                    self.results.pairs = np.asarray(self.results.pairs)
                else:
                    raise Exception("at_g1 contains lists, however com is set to False and no angle criterion is provided! Please set 'com = True' if you want to calculate the distance between centre of masses of molecules, please specify angle criterion if you want to filter pairs using angle criterion. Otherwise, please use a list of int as ag_g1.")
            # The case when ag_g1 is a H-bond donor list
            else:
                # The case of one donor in at_g1
                if isinstance(self.g1[0], int):
                    try:
                        self.results.pairs = list(set(itertools.product([self.g1[0]], self.g1[1], self.g2)))
                    except TypeError:
                        self.results.pairs = list(set(itertools.product([self.g1[0]], self.g1[1], [self.g2])))
                # The case there are many donors in at_g1
                else:
                    self.results.pairs = []
                    for donor in self.g1:
                        for acceptor in self.g2:
                            if donor[0] == acceptor:
                                symbol = self.traj.frames[0][acceptor].symbs
                                print(f"Donor and acceptor is the same atom: {symbol} {acceptor}, skipped.")
                                continue
                            else:
                                [self.results.pairs.append(x) for x in list(map(list, itertools.product([donor[0]], donor[1], [acceptor])))]
                    self.results.pairs = np.asarray(self.results.pairs)
        
    def pair_filter(self, mic=True, filename=None):
        """Apply geometry criteria to identify dimers in each frame.

        Iterates over all trajectory frames, computes the relevant geometry
        observables (distance, and optionally D–H···A angle) for every
        candidate pair, and retains only those that satisfy the criteria set
        at construction time.

        Parameters
        ----------
        mic : bool, optional
            Apply the minimum image convention when computing distances and
            angles.  Should be True for periodic systems.  Default: True.
        filename : str, optional
            If provided, serialise ``results.dimers`` to a JSON file with this
            name.  The file can be passed directly to :func:`auto_cor` or
            :class:`DLDDF`.

        Notes
        -----
        For the flexible angle criterion (``"<angle>f"``), the threshold
        ∠_min scales with the instantaneous D–H bond length as
        ``angle_f × r(D–H) / (1.05 × (R_D + R_H))``, loosening the criterion
        for elongated bonds to capture H-bond formation even during large
        thermal fluctuations.
        """
        self.results.dimers = []
        # com = self.com
        if self.com:
            for i, frame in enumerate(self.traj.frames):
                if mic:
                    dists = [mic_dist(frame[pair[0].tolist()].calc_com(), frame[pair[1].tolist()].calc_com(), cell = self.traj.cell) for pair in self.results.pairs]
                else:
                    dists = np.linalg.norm(np.asarray([frame[pair[1].tolist()].calc_com() - frame[pair[0].tolist()].calc_com() for pair in self.results.pairs]), axis = 1)
                idx = dists <= self.criteria.distance
                self.results.dimers.append(
                    {
                        "frame": i,
                        "pairs": self.results.pairs[idx].squeeze().tolist(),
                        "dists": dists[idx].tolist(),
                    }
                )
                update_progress(i/self.traj.nframes)
                
        else:
            if isinstance(self.criteria.distance, str):
                temp = re.compile(r"([\d\.]+)([a-zA-Z_*]+)")
                res = temp.match(self.criteria.distance).groups()
                if res[1] in ["vdw", "vdw_sum", "vdw_r"]:
                    self.criteria.distance = np.asarray([float(res[0]) * (data.vdW_R[self.traj.frames[0][pair[0].tolist()].symbs[0]] + data.vdW_R[self.traj.frames[0][pair[1].tolist()].symbs[0]]) for pair in self.results.pairs])
                elif res[1] in ["coval", "coval_sum", "coval_r"]:
                    self.criteria.distance = np.asarray([float(res[0]) * (data.val_R[self.traj.frames[0][pair[0].tolist()].symbs[0]] + data.val_R[self.traj.frames[0][pair[1].tolist()].symbs[0]]) for pair in self.results.pairs])
                else:
                    raise Exception("Unrecognised string specifier for distance criterion! Please use float + 'vdw_sum' or float + 'coval_sum' as distance criterion, eg '1.05vdw_sum' or '1.5coval_sum'.")
                # Dimers based on distance only
            if self.criteria.angle is None:
                for i, frame in enumerate(self.traj.frames):
                    if mic:
                        dists = np.asarray([frame.dist(pair[0], pair[1], mic = True) for pair in self.results.pairs])
                    else:
                        dists = np.linalg.norm(frame[self.results.pairs[:,1].tolist()].pos - frame[self.results.pairs[:,0].tolist()].pos, axis = 1)
                    idx = dists <= self.criteria.distance
                    self.results.dimers.append(
                        {
                            "frame": i,
                            "pairs": self.results.pairs[idx].squeeze().tolist(),
                            "dists": dists[idx].tolist(),
                        }
                    )
                    update_progress(i/self.traj.nframes)
            # Complicated case, using distance and angle criteria
            else:
                dt = np.dtype([('distance', np.float64), ('angle', np.float64)])
                if isinstance(self.criteria.angle, str):
                    temp = re.compile(r"([\d\.]+)([a-zA-Z]+)")
                    res = temp.match(self.criteria.angle).groups()
                    
                    for i, frame in enumerate(self.traj.frames):
                        if mic:
                            dist_ang = np.asarray([(frame.dist(pair[0], pair[2], mic = True), frame.angle(*pair.tolist(), mic = True)) for pair in self.results.pairs], dtype = dt)
                            d_h_dists = np.asarray([frame.dist(*pair[:-1].tolist(), mic = True) for pair in self.results.pairs])
                            # Add a fluctuation to the angle around the specified angle
                            self.criteria.angle =float(res[0]) * d_h_dists / (1.05 * np.array([data.val_R[self.traj.frames[0][pair[0].tolist()].symbs[0]] + data.val_R[self.traj.frames[0][pair[1].tolist()].symbs[0]] for pair in self.results.pairs]))
                        else:
                            dist_ang = np.asarray([(frame.dist(pair[0], pair[2], mic = False), frame.angle(*pair.tolist(), mic = False)) for pair in self.results.pairs], dtype = dt)
                            d_h_dists = np.asarray([frame.dist(*pair[:-1].tolist(), mic = False) for pair in self.results.pairs])
                            # Add a fluctuation to the angle around the specified angle
                            self.criteria.angle =float(res[0]) * d_h_dists / (1.05 * np.array([data.val_R[self.traj.frames[0][pair[0].tolist()].symbs[0]] + data.val_R[self.traj.frames[0][pair[1].tolist()].symbs[0]] for pair in self.results.pairs]))
                        idx = np.where((dist_ang["distance"] <= self.criteria.distance) & (dist_ang["angle"] >= self.criteria.angle))
                        self.results.dimers.append(
                            {
                                "frame": i,
                                "d_h_a_pairs": self.results.pairs[idx].squeeze().tolist(),
                                "d_a_dists": dist_ang["distance"][idx].tolist(),
                                "d_h_a_angles": dist_ang["angle"][idx].tolist(),
                            }
                        )
                        update_progress(i/self.traj.nframes)
                else:
                    for i, frame in enumerate(self.traj.frames):
                        if mic:
                            dist_ang = np.asarray([(frame.dist(pair[0], pair[2], mic = True), frame.angle(*pair.tolist(), mic = True)) for pair in self.results.pairs], dtype = dt)
                            d_h_dists = np.asarray([frame.dist(*pair[:-1].tolist(), mic = True) for pair in self.results.pairs])
                        else:
                            dist_ang = np.asarray([(frame.dist(pair[0], pair[2], mic = False), frame.angle(*pair.tolist(), mic = False)) for pair in self.results.pairs], dtype = dt)
                            d_h_dists = np.asarray([frame.dist(*pair[:-1].tolist(), mic = False) for pair in self.results.pairs])
                        idx = np.where((dist_ang["distance"] <= self.criteria.distance) & (dist_ang["angle"] >= self.criteria.angle))
                        self.results.dimers.append(
                            {
                                "frame": i,
                                "d_h_a_pairs": self.results.pairs[idx].squeeze().tolist(),
                                "d_a_dists": dist_ang["distance"][idx].tolist(),
                                "d_h_a_angles": dist_ang["angle"][idx].tolist(),
                            }
                        )
                        update_progress(i/self.traj.nframes)
                        
        if filename is not None:
            import json
            with open(filename, "w") as json_file:
                json.dump(self.results.dimers, json_file, indent=4, separators=(", ", ": "))
            json_file.close()
        update_progress(1)


class DLDDF(object):
    """Dimer Lifetime–Displacement Distribution Function (DLDDF).

    The DLDDF is a 2-D joint probability distribution that correlates the
    *lifetime* τ of a dimer pair (the number of consecutive frames during which
    the pair satisfies the geometry criteria) with the *centre-of-mass
    displacement* Δr of that pair during the same interval:

        DLDDF(τ, Δr) = P(lifetime = τ, displacement = Δr)

    It is constructed by dividing the trajectory into non-overlapping windows
    of *nframes* frames, locating all active dimer pairs in the first frame of
    each window, tracking their survival until they break, and accumulating the
    (lifetime, displacement) pair into a 2-D histogram.

    The DLDDF reveals the coupling between structural relaxation (how long
    dimers survive) and translational dynamics (how far the partners move while
    associated).  Short-lived dimers with large displacements indicate a
    diffusive mechanism, while long-lived pairs with small displacements
    suggest a caged or localised regime.

    Parameters
    ----------
    traj : Trajectory
        Trajectory object to analyse.  Coordinates are wrapped into the
        simulation box on initialisation.
    dimer_settings : dict, optional
        Keyword arguments forwarded to the :class:`dimers` constructor
        (``at_g1``, ``at_g2``, ``criteria``, ``com``).  Required when calling
        :meth:`dimer_res`; not needed if ``results.dimers`` is set directly.
    nbins : int, optional
        Number of bins along each axis of the 2-D histogram. Default: 100.
    range : ndarray, shape (2, 2), optional
        ``[(τ_min, τ_max), (Δr_min, Δr_max)]`` — axis ranges in ps and Å.
        Default: ``[(0.0, 2.0), (0.0, 5.0)]``.

    Results dataclass fields (populated after calling calculate())
    -------------------------------------------------------------
    dlddf     : ndarray, shape (nbins, nbins) — 2-D histogram (transposed for
                correct contour-plot orientation: rows = Δr, cols = τ).
    r_edges   : ndarray, shape (nbins+1,) — bin edges of the displacement axis.
    tau_edges : ndarray, shape (nbins+1,) — bin edges of the lifetime axis.
    r         : ndarray, shape (nbins,) — displacement bin centres in Å.
    tau       : ndarray, shape (nbins,) — lifetime bin centres in ps.
    dists     : list of float — all recorded displacements.
    taus      : list of float — all recorded lifetimes.
    dimers    : list of dict — per-frame dimer data (populated by dimer_res).
    plot      : Figure or None — contourf figure if plot=True was requested.

    Notes
    -----
    Periodic boundary conditions are applied via MIC when computing CoM
    displacements (``mic=True``), ensuring that pairs that cross box boundaries
    during their lifetime are handled correctly.

    Examples
    --------
    >>> dimer_settings = dict(at_g1=g1, at_g2=g2, criteria=[3.5, None])
    >>> dlddf = DLDDF(traj, dimer_settings=dimer_settings, nbins=100,
    ...               range=np.array([(0.0, 1.0), (0.0, 4.0)]))
    >>> dlddf.dimer_res(filename="dimers.json")   # resolve + cache to disk
    >>> results = dlddf.calculate(nframes=200, plot=True)
    """
    def __init__(self, traj, dimer_settings = None, nbins = 100, range = np.array([(0.0, 2.0), (0.0, 5.0)])):
        self.traj = traj.wrap2box()
        self.dimer_settings = dimer_settings
        settings_dc = make_dataclass("settings", "bins, range")
        self.settings = settings_dc(nbins, range)
        results = make_dataclass("Results", "dimers r_edges tau_edges r tau dists taus dlddf plot")
        self.results = results
        self.results.dlddf, self.results.r_edges, self.results.tau_edges = np.histogram2d([-1], [-1], **asdict(self.settings))
        self.results.r = (self.results.r_edges[:-1] + self.results.r_edges[1:])/2
        self.results.tau = (self.results.tau_edges[:-1] + self.results.tau_edges[1:])/2
        self.results.dists = []
        self.results.taus = []
        
    def dimer_res(self, mic=True, filename=None):
        """Resolve dimers and store results for subsequent DLDDF calculation.

        Constructs a :class:`dimers` instance from ``dimer_settings``, calls
        :meth:`~dimers.make_pairs` and :meth:`~dimers.pair_filter`, then stores
        the per-frame dimer list in ``results.dimers``.

        Parameters
        ----------
        mic : bool, optional
            Apply the minimum image convention during pair filtering.
            Default: True.
        filename : str, optional
            If provided, the resolved dimer data are also written to a JSON
            file under this name (useful for caching long calculations).
        """
        dimers_data = dimers(self.traj, **self.dimer_settings)
        dimers_data.make_pairs()
        dimers_data.pair_filter(mic = mic, filename = filename)
        self.results.dimers = dimers_data.results.dimers
    
    def calculate(self, nframes=100, mic=True, plot=False, bins=100, levels=25,
                  xlims=[0, 0.2], ylims=[0, 0.8], filename=None, **kwargs):
        """Compute the DLDDF 2-D histogram and optionally plot a contour map.

        The trajectory is divided into non-overlapping windows of *nframes*
        frames.  For each window the active dimer pairs in the first frame are
        tracked forward until they break; the pair's (lifetime, CoM
        displacement) is then accumulated into a 2-D histogram.

        Parameters
        ----------
        nframes : int, optional
            Number of consecutive frames per window. Smaller values sample
            shorter lifetimes at higher statistics; larger values capture
            longer-lived dimers but reduce the number of independent windows.
            Default: 100.
        mic : bool, optional
            Apply the minimum image convention when computing CoM
            displacements. Default: True.
        plot : bool, optional
            If True, produce a filled contour plot of the DLDDF.  Default: False.
        bins : int, optional
            Passed to :func:`numpy.histogram2d` (overrides ``nbins`` from
            ``__init__`` if provided). Default: 100.
        levels : int, optional
            Number of contour levels in the plot. Default: 25.
        xlims : list of float, optional
            x-axis (lifetime) limits in ps for the plot. Default: [0, 0.2].
        ylims : list of float, optional
            y-axis (displacement) limits in Å for the plot. Default: [0, 0.8].
        filename : str, optional
            If provided, save the plot to this path at 600 dpi.
        **kwargs
            Additional keyword arguments forwarded to
            :func:`matplotlib.axes.Axes.contourf`.

        Returns
        -------
        results : dataclass
            Populated results dataclass (see class docstring).

        Notes
        -----
        The histogram bins are set by ``self.settings`` (``nbins`` and
        ``range`` supplied at construction time), not by the ``bins``
        parameter, which is currently unused.  Use ``range`` at construction
        to control the axis extents.

        Interpretation
        --------------
        - **Short τ, small Δr**: dimers that break quickly and barely move —
          typical of thermal fluctuations at the boundary of the geometry
          criterion (transient contacts).
        - **Short τ, large Δr**: rapid breaking with significant displacement —
          indicative of fast diffusive exchange typical in liquid-like regimes.
        - **Long τ, small Δr**: long-lived, localised pairs — characteristic of
          strongly bound dimers or caged dynamics (e.g. ion pairs in viscous
          electrolytes or hydrogen-bonded networks).
        - **Diagonal ridge**: a positive correlation between lifetime and
          displacement suggests that pairs survive until they diffuse away,
          consistent with a Brownian escape mechanism.
        """
        # Use the dimers object to resolve dimers
        key = "pairs"
        if key not in list(self.results.dimers[0].keys()):
            key = "d_h_a_pairs"
        
        dimer_chunks = to_sublists(self.results.dimers, nframes)
        
        for i, dimer_chunk in enumerate(dimer_chunks):
            select = dimer_chunk
            pairs_0 = select[0][key]
            # Real copy the pairs_0 list so that the pairs_0 do not change during mutation
            pairs_ref = pairs_0[:]
            frame_idx_0 = select[0]["frame"]

            # Pass if the first frame in selected dimer_chunk is empty
            if pairs_0 == []:
                # print("Empty pool, pass")
                continue

            else:
                # Iterate through the frames to identify the lifetime and displacement of the pair
                for j, dimer_list in enumerate(select[1:]):
                    pairs = dimer_list[key]
                    frame_idx = dimer_list["frame"]
                    # No pairs in current frame, all pairs are dead
                    if pairs == []:
                        dead_pair = pairs_0[:]
                        # print("All pairs dead!")

                        # Only one pair in the pool
                        if all(isinstance(x, int) for x in pairs_0):
                            if pairs_0 == []:
                                break
                            else:
                            # Calculate the displacement and lifetime
                                if mic:
                                    dist = mic_dist(self.traj.frames[frame_idx_0][dead_pair].calc_com(), self.traj.frames[frame_idx][dead_pair].calc_com(), cell = self.traj.cell)
                                else:
                                    dist = np.linalg.norm(self.traj.frames[frame_idx][dead_pair].calc_com() - self.traj.frames[frame_idx_0][dead_pair].calc_com())
                                lifetime = (j + 1) * self.traj.timestep/1000
                                self.results.dists.append(dist)
                                self.results.taus.append(lifetime)

                        # More than one pairs in the pool
                        else:
                            dead_pairs = [x for x in pairs_0 if x != pairs]

                            # Calculate the displacement and lifetime
                            for dead_pair in dead_pairs:
                                if mic:
                                    dist = mic_dist(self.traj.frames[frame_idx_0][dead_pair].calc_com(), self.traj.frames[frame_idx][dead_pair].calc_com(), cell = self.traj.cell)
                                else:
                                    dist = np.linalg.norm(self.traj.frames[frame_idx][dead_pair].calc_com() - self.traj.frames[frame_idx_0][dead_pair].calc_com())
                                lifetime = (j + 1) * self.traj.timestep/1000
                                self.results.dists.append(dist)
                                self.results.taus.append(lifetime) 

                        break
                    # if 
                    # If there is only one pair in current frame
                    elif all(isinstance(x, int) for x in pairs):
                        if pairs_0 == []:
                            break
                        if all(isinstance(x, int) for x in pairs_0):
                            if pairs_0 == pairs:
                                pass
                        else:
                            dead_pairs = [x for x in pairs_0 if x != pairs]

                            # Calculate the displacement and lifetime
                            for dead_pair in dead_pairs:
                                if mic:
                                    dist = mic_dist(self.traj.frames[frame_idx_0][dead_pair].calc_com(), self.traj.frames[frame_idx][dead_pair].calc_com(), cell = self.traj.cell)
                                else:
                                    dist = np.linalg.norm(self.traj.frames[frame_idx][dead_pair].calc_com() - self.traj.frames[frame_idx_0][dead_pair].calc_com())
                                lifetime = (j + 1) * self.traj.timestep/1000
                                self.results.dists.append(dist)
                                self.results.taus.append(lifetime)

                            [pairs_ref.remove(x) for x in dead_pairs]
                            pairs_0 = pairs_ref[:]

                    # If there are more than one pair in current frame, the pairs will be list of lists
                    else:
                        # Only one pair in the pool
                        if all(isinstance(x, int) for x in pairs_0):
                            if pairs_0 == []:
                                break
                            # Dimmer dead
                            elif pairs_0 not in pairs:
                                # print(f"Dimer {pairs_0} dead, removed from pool.")
                                dead_pair = pairs_0[:]
                                # Calculate the dislpacement and lifetime
                                if mic:
                                    dist = mic_dist(self.traj.frames[frame_idx_0][dead_pair].calc_com(), self.traj.frames[frame_idx][dead_pair].calc_com(), cell = self.traj.cell)
                                else:
                                    dist = np.linalg.norm(self.traj.frames[frame_idx][dead_pair].calc_com() - self.traj.frames[frame_idx_0][dead_pair].calc_com())
                                lifetime = (j + 1) * self.traj.timestep/1000
                                self.results.dists.append(dist)
                                self.results.taus.append(lifetime)

                                pairs_ref = []
                                pairs_0 = pairs_ref[:]

                                break
                            # Dimmer survives    
                            else:
                                pass

                        # More than one pairs exist in the pool
                        else:
                            for item in pairs_0:
                                if item not in pairs:

                                    dead_pair = list(deepflatten(item))
                                    # Calculate the dislpacement and lifetime
                                    if mic:
                                        dist = mic_dist(self.traj.frames[frame_idx_0][dead_pair].calc_com(), self.traj.frames[frame_idx][dead_pair].calc_com(), cell = self.traj.cell)
                                    else:
                                        dist = np.linalg.norm(self.traj.frames[frame_idx][dead_pair].calc_com() - self.traj.frames[frame_idx_0][dead_pair].calc_com())
                                    lifetime = (j + 1) * self.traj.timestep/1000
                                    self.results.dists.append(dist)
                                    self.results.taus.append(lifetime)
                                    try:
                                        pairs_ref.remove(item)
                                        pairs_0 = pairs_ref[:]

                                    except ValueError:
                                        print(item)
                                        print(pairs_0)
                                        print("Value error!")
                                        pass
            update_progress(i/len(dimer_chunks))
        
        x = np.asarray(self.results.taus)
        y = np.asarray(self.results.dists)

        H, xedges, yedges = np.histogram2d(x, y, **asdict(self.settings))

        self.results.dlddf = H = H.T # Transpose so that the plot is correct
        
        if plot:

            fig = plt.figure(figsize=(4.8, 4.0))
            ax  = fig.add_axes([0.16, 0.16, 0.82, 0.75])
            
            xcenters = (xedges[:-1] + xedges[1:])/2
            ycenters = (yedges[:-1] + yedges[1:])/2

            X, Y = np.meshgrid(xcenters, ycenters)

            CS = ax.contourf(X, Y, H, cmap = color_ramp, vmin = 0.01 * H.max(), levels = levels, **kwargs)
            ax.contour(X, Y, H, colors ="k", levels = 25, linewidths = 0.1)

            ax.set_xlabel(r"Lifetime (ps)")
            ax.set_ylabel(r"Displacement (${\AA}$)")
            ax.set_xlim(xlims)
            ax.set_ylim(ylims)

            plt.colorbar(CS, pad=0.035, location = "right", ax = ax)
            
            if filename is not None:
                plt.savefig(filename, dpi = 600)

            plt.show()
            
        update_progress(1)
                
        return self.results


#----Useful functions to make donor-accepter pairs----#
def find_d_a(atoms_dict):
    """
    Creat donor, acceptor dicts from atoms_dicts. Include one water to creat donor, acceptor for specified water.
    """
    donors = []
    acceptors = []
    for a_dict in atoms_dict:
        symbols = [retrieve_symbol(key) for key in list(a_dict.keys())] # retrieve chemical symbol from atoms_dict keys
        if symbols[0] == "F":
            acceptors += (a_dict,)
        elif symbols[0] == "N":
            if len(symbols) == 2:
                donors += (a_dict,)
            else:
                combs = [{"N": a_dict["N"], x: a_dict[x]} for x in list(a_dict.keys()) if x != "N"]
                donors += (combs[0], combs[1])

        elif symbols[0] == "O":
            if symbols[0] == symbols[1]:
                acceptors += (a_dict,)
            elif len(symbols) == 2 and symbols[0] != symbols[1]:
                donors += (a_dict,)
            else:
                combs = [{"O": a_dict["O"], x: a_dict[x]} for x in list(a_dict.keys()) if x != "O"]
                donors += (combs[0], combs[1])
                acceptors += ({"O": a_dict["O"]},)
    print(f"Done! {len(donors)} donors and {len(acceptors)} acceptors were found!\n" + "Acceptors:\n", acceptors, "\n Donors:\n", donors)
    return donors, acceptors

def pair_d_a(donors, acceptors):
    """
    Make d_a_pairs and d_h_a_pairs from the donors and acceptors dict.
    """
    d_a_pairs = []
    d_h_a_pairs = []
    for donor in donors:
        d_atom = [{x: donor[x]} for x in list(donor.keys()) if not x.startswith("H")]
        for acceptor in acceptors:
            keys = list(acceptor.keys())

            if len(keys) == 1:
                symbol = list(d_atom[0].keys())[0]
                if symbol == "O" and d_atom[0][symbol] == acceptor[keys[0]]: # exclude combinations where donor and acceptor are the same oxygen atom.
                    print(f"Donor and acceptor is the same atom: {symbol} {acceptor[keys[0]]}, skipped.")
                    continue
                else:
                    d_a_pairs += [[d_atom[0], acceptor],]
                    d_h_a_pairs += [[donor, acceptor],]
            else:
                for key in keys:
                    d_a_pairs += [[d_atom[0], {key: acceptor[key]}],]
                    d_h_a_pairs += [[donor, {key: acceptor[key]}],]
    print(f"Unique donor-acceptor/donor-hydrogen-acceptor combinations: {len(d_a_pairs)}")
    print(f"Sample donor-acceptor pair: {d_a_pairs[0]}\nSample donor-hydrogen-acceptor pair: {d_h_a_pairs[0]}")
    return d_a_pairs, d_h_a_pairs

def res_h(d_a_pairs, d_h_a_pairs, traj):
    """
    Calculates the H-bond information in each frame of the trajectory.
    """
    results = []
    for i, frame in enumerate(traj.frames):
        hbonds = [] #list to store all hbonds in current frame
        for d_a_pair, d_h_a_pair in zip(d_a_pairs, d_h_a_pairs):
            # Sum of van der Waals radii
            symbols = [list(item.keys()) for item in d_a_pair]
            symbols = [item for sublist in symbols for item in sublist] # flatten the list of lists
            symbols = [retrieve_symbol(symbol) for symbol in symbols] # Remove numbers and get chemical symbol from atoms_dict to be ready to pass to the vdW radii dict
            vdW_sum = data.vdW_R[symbols[0]] + data.vdW_R[symbols[1]]

            # not a H-bond of D-A distance is greater than their vdW radii times 1.02, 1.02 to take bond length change during MD simulation.
            r_d_a = frame.dist(*d_a_pair[0].values(), *d_a_pair[1].values(), mic = True) # calculate the D-A distance
            if r_d_a <= 1.02 * vdW_sum:
                # calculate the D-H⋅⋅⋅A angle
                d_h_a_ang = frame.angle(*d_h_a_pair[0].values(), *d_h_a_pair[1].values(), mic = True)  # D-H···A angle               
                d_h = frame.dist(*d_h_a_pair[0].values(), mic = True) # calculate the D-H length

                # the D-H⋅⋅⋅A angle criteria used: the D-H⋅⋅⋅A angle is close to a right angle refer to the D-H⋅⋅⋅A angle - R(D⋅⋅⋅A) plot
                # an angle range is included considering the oscillation of bond lenghth and anlgle
                if d_h_a_ang >= (np.rad2deg(np.arctan2(r_d_a, d_h)) + 180)*3/8:
                # if d_h_a_ang >= 90:
                    # Store current H-bond
                    hbonds.append(
                            {
                                "donor": d_h_a_pair[0],
                                "acceptor": d_h_a_pair[1],
                                "R(D-A)": r_d_a,
                                "DHA_ang": d_h_a_ang,
                                "r(D-H)": d_h,
                            }
                    )
        results.append({f"frame": i, "n_hbonds": len(hbonds), "hbonds": hbonds})
        update_progress(i/len(traj.frames))
    return results


def bi_exp(t, amp1, tau1, tau2):
    """Biexponential decay model C(t) = A·exp(−t/τ₁) + (1−A)·exp(−t/τ₂)."""
    return amp1*np.exp(-t/tau1) + (1-amp1)*np.exp(-t/tau2)


def bi_exp_fit(x, y, amp1=0.5, tau1=100, tau2=150):
    """Fit a biexponential decay to H-bond autocorrelation data.

    The model is C(t) = A · exp(−t/τ₁) + (1−A) · exp(−t/τ₂), which captures
    two distinct relaxation processes: a fast component associated with
    librational motions and a slow component representing structural H-bond
    breaking and reformation.

    Parameters
    ----------
    x : ndarray
        Time axis (ps).
    y : ndarray
        Autocorrelation values C(t), expected to decay from ~1 toward 0.
    amp1 : float, optional
        Initial guess for the fast-component amplitude A ∈ (0, 1). Default: 0.5.
    tau1 : float, optional
        Initial guess for the fast relaxation time τ₁ (same units as x).
        Default: 100.
    tau2 : float, optional
        Initial guess for the slow relaxation time τ₂ (same units as x).
        Default: 150.

    Returns
    -------
    x_fit : ndarray
        Dense time axis for plotting the fitted curve (200 points).
    y_fit : ndarray
        Fitted biexponential values at ``x_fit``.
    params : ndarray
        Optimised parameters (A, τ₁, τ₂).

    Notes
    -----
    The effective (integrated) H-bond lifetime is

        τ_eff = A · τ₁ + (1−A) · τ₂

    Use :func:`calc_tau` for the exact value via numerical integration.
    """
    params, _ = curve_fit(bi_exp, x, y, p0=[amp1, tau1, tau2])
    x_fit = np.linspace(x.min(), x.max(), num=200)
    y_fit = bi_exp(x_fit, *params)
    print("The fitted biexponential function parameters are:\nA: {0:.4f}, tau1: {1:.4f}, tau2: {2:.4f}".format(*params))
    return x_fit, y_fit, params


def calc_tau(amp1, tau1, tau2):
    """Compute the effective H-bond lifetime by integrating the biexponential.

    The effective lifetime is the area under the normalised autocorrelation
    function:

        τ_eff = ∫₀^∞ [A · exp(−t/τ₁) + (1−A) · exp(−t/τ₂)] dt
              = A · τ₁ + (1−A) · τ₂

    Parameters
    ----------
    amp1 : float
        Fast-component amplitude A from :func:`bi_exp_fit`.
    tau1 : float
        Fast relaxation time τ₁ (ps).
    tau2 : float
        Slow relaxation time τ₂ (ps).

    Returns
    -------
    res : float
        Effective H-bond lifetime τ_eff in the same units as τ₁ and τ₂.

    Notes
    -----
    The analytical result A·τ₁ + (1−A)·τ₂ agrees with the numerical
    integration to machine precision.  The numerical route is retained here
    for consistency with non-standard functional forms.
    """
    def function(x):
        return amp1*np.exp(-x/tau1) + (1 - amp1)*np.exp(-x/tau2)

    res, err = quad(function, 0, np.inf)
    print(f"The H-bond lifetime is {res:.4f} ps.")
    return res


def auto_cor(data="hbonds.json", nframes=1000, skip=100, index=":", timestep=None):
    """Compute the H-bond survival autocorrelation function C(t) from a JSON file.

    Reads the per-frame H-bond list produced by :meth:`dimers.pair_filter` (or
    :func:`res_h`) from a JSON file and evaluates the continuous (non-
    intermittent) autocorrelation function

        C(t) = ⟨h(0) · h(t)⟩ / ⟨h(0)²⟩

    where h(t) = 1 if a bond that existed at t = 0 is still intact at time t
    and 0 otherwise.  The trajectory is divided into non-overlapping windows of
    *nframes* frames; C(t) is averaged over all windows to improve statistics.

    Parameters
    ----------
    data : str, optional
        Path to the JSON file containing the H-bond list. Default: "hbonds.json".
    nframes : int, optional
        Number of frames per analysis window. Default: 1000.
    skip : int, optional
        Stride for sampling within each window: only every *skip*-th frame is
        used. Default: 100.
    index : str or int or list of int, optional
        Frame selection from the JSON file:
        ``":"`` — all frames; ``int n`` — first n frames;
        ``[start, stop]`` — slice. Default: ``":"``.
    timestep : float, optional
        Trajectory time step in fs.  Used to build the time axis in ps.
        Default: 5 fs.

    Returns
    -------
    t : ndarray
        Time axis in ps for the sampled points (length = nframes // skip).
    C_ts_mean : ndarray
        Mean C(t) averaged over all trajectory windows.
    C_ts_error : ndarray
        Standard error of C(t) across windows.

    Notes
    -----
    C(t) decays from 1 at t = 0 toward 0 as H-bonds break.  Fitting the
    result with :func:`bi_exp_fit` and integrating with :func:`calc_tau`
    yields the effective H-bond lifetime τ_eff.

    The *continuous* (non-intermittent) definition used here counts a bond as
    broken permanently as soon as it fails the geometry criterion for the first
    time, even if it later reforms.  This gives the *structural relaxation*
    lifetime rather than the longer *reactive* lifetime that allows
    intermittent breaking.
    """
    # Opening JSON data file containing the hbonds information
    f = open(data)

    # returns JSON object as a dictionary
    if index == ":":
        frames = json.load(f)
        
    elif isinstance(index, int):
        frames = json.load(f)[:index]
        
    elif isinstance(index, list):
        frames = json.load(f)[index[0]:index[1]]
    
    frame_chunks = to_sublists(frames, nframes)

    C_ts = np.zeros((len(frame_chunks), nframes//skip))
    
    for i, frame_chunk in enumerate(frame_chunks):
        select = frame_chunk[::skip]
        nbonds_0 = nbonds = select[0]["n_hbonds"] # hij(t0)
        if nbonds_0 == 0: # skip if there is no hydrogen bonds in the first frame of the selected frames
            pass
        else:
            # store the donor and acceptor dicts in to a list
            hbonds_0 = select[0]["hbonds"]
            # for hbond_0 in hbonds_0:
            d_a_list_0 = [[hbonds_0[x]["donor"], hbonds_0[x]["acceptor"]] for x in range(len(hbonds_0))] # start point of h-bonds
            C_t = []
            for frame in select:
                hbonds = frame["hbonds"]
                # Make a list of donors and acceptors for each frame
                d_a_list = [[hbonds[x]["donor"], hbonds[x]["acceptor"]] for x in range(len(hbonds))] # start point of h-bonds

                # Iterate through the elements in hbonds_0, if any hbond breaks (not in the list of hbonds in a frame), remove the hbond from d_a_list_0, and nbonds decreases by one
                for item in list(d_a_list_0):
                    if item not in d_a_list:
                        d_a_list_0.remove(item)
                        nbonds -= 1
                # Calculate the C(t)
                C_t.append(nbonds*nbonds_0/nbonds_0**2)
        C_ts[i] = np.array(C_t)
    # Calculate the mean and error of C_ts
    C_ts_mean = C_ts.mean(axis = 0)
    C_ts_error = C_ts.std(axis = 0)/(len(frames)//nframes)**0.5
    if timestep is None:
        timestep = 5
    t_end = 0 + (nframes - 1)*timestep # 5 fs is the interval of traj
    t = np.linspace(0, t_end, num = nframes//skip)/1000
    return t, C_ts_mean, C_ts_error