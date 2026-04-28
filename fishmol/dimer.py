import os
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
        results_cls = make_dataclass("Results", "pairs dimers")
        self.results = results_cls(pairs=None, dimers=[])
                  
    def make_pairs(self):
        """Enumerate all candidate pairs from the two atom groups.

        Generates the Cartesian product of ``at_g1`` × ``at_g2``, removes
        self-pairs (where the same atom appears in both groups), and removes
        duplicate reverse pairs (i, j) / (j, i) when the two groups overlap.
        The result is stored in ``results.pairs`` as an integer ndarray of
        shape (n_pairs, 2) for distance-only mode, or (n_pairs, 3) for
        H-bond mode (columns: D, H, A).
        """
        # Case 1: Centre of Mass dimers (molecule indices)
        if self.com:
            raw_pairs = [list(x) for x in itertools.product(self.g1, self.g2) if x[0] != x[1]]
            # Remove symmetric duplicates: if (A,B) and (B,A) both exist, keep one.
            seen = set()
            unique_pairs = []
            for p in raw_pairs:
                canon = tuple(sorted(p))
                if canon not in seen:
                    unique_pairs.append(p)
                    seen.add(canon)
            self.results.pairs = np.asarray(unique_pairs)
            
        # Case 2: Simple atom-pair dimers (distance only)
        elif self.criteria.angle is None:
            if all(isinstance(idx, (int, np.integer)) for idx in self.g1):
                raw_pairs = [list(x) for x in itertools.product(self.g1, self.g2) if x[0] != x[1]]
                seen = set()
                unique_pairs = []
                for p in raw_pairs:
                    canon = tuple(sorted(p))
                    if canon not in seen:
                        unique_pairs.append(p)
                        seen.add(canon)
                self.results.pairs = np.asarray(unique_pairs)
            else:
                raise ValueError("at_g1 contains lists but com=False and no angle criterion provided. "
                                 "Use com=True for CoM dimers or provide angle criteria for H-bonds.")
                                 
        # Case 3: Hydrogen-bond dimers (D-H...A)
        else:
            # Normalize at_g1 to list of [D, [H1, H2...]]
            if isinstance(self.g1[0], (int, np.integer)):
                donors = [self.g1]
            else:
                donors = self.g1
                
            unique_triplets = []
            for donor in donors:
                d_idx = donor[0]
                h_list = donor[1]
                for a_idx in self.g2:
                    if d_idx == a_idx:
                        continue
                    for h_idx in h_list:
                        unique_triplets.append([d_idx, h_idx, a_idx])
            self.results.pairs = np.asarray(unique_triplets)
        
    def pair_filter(self, mic=True, filename=None):
        """Apply geometry criteria to identify dimers in each frame."""
        self.results.dimers = []
        n_frames = len(self.traj.frames)
        n_pairs = len(self.results.pairs)
        
        # 1. Resolve distance criteria
        if isinstance(self.criteria.distance, str):
            # Parse string criteria like "1.05vdw_sum"
            match = re.match(r"([\d\.]+)([a-zA-Z_*]+)", self.criteria.distance)
            if not match:
                raise ValueError(f"Invalid distance criterion string: {self.criteria.distance}")
            
            scale = float(match.group(1))
            kind = match.group(2).lower()
            
            # Map atoms to radii
            radii_map = data.vdW_R if "vdw" in kind else data.val_R
            
            # Determine threshold for each candidate pair
            if self.criteria.angle is None or self.com:
                # Pair distance
                thresholds = []
                for pair in self.results.pairs:
                    sym1 = self.traj.frames[0].symbs[pair[0]]
                    sym2 = self.traj.frames[0].symbs[pair[1]]
                    thresholds.append(scale * (radii_map[sym1] + radii_map[sym2]))
            else:
                # D...A distance for H-bonds (triplets: [D, H, A])
                thresholds = []
                for trip in self.results.pairs:
                    symD = self.traj.frames[0].symbs[trip[0]]
                    symA = self.traj.frames[0].symbs[trip[2]]
                    thresholds.append(scale * (radii_map[symD] + radii_map[symA]))
            self.dist_thresholds = np.asarray(thresholds)
        else:
            # Fixed float cutoff
            self.dist_thresholds = np.full(n_pairs, float(self.criteria.distance))

        # 2. Frame-by-frame loop
        for f_idx, frame in enumerate(self.traj.frames):
            if self.com:
                # CoM distances
                coms_a = np.stack([frame[p[0].tolist()].calc_com() for p in self.results.pairs])
                coms_b = np.stack([frame[p[1].tolist()].calc_com() for p in self.results.pairs])
                if mic:
                    # Implement vectorized MIC dist for CoM
                    diff = coms_b - coms_a
                    from fishmol.utils import cart2xys, xys2cart
                    f_diff = cart2xys(diff, self.traj.cell)
                    f_diff -= np.round(f_diff)
                    dists = np.linalg.norm(xys2cart(f_diff, self.traj.cell), axis=1)
                else:
                    dists = np.linalg.norm(coms_b - coms_a, axis=1)
                
                mask = dists <= self.dist_thresholds
                self.results.dimers.append({
                    "frame": f_idx,
                    "pairs": self.results.pairs[mask].tolist(),
                    "dists": dists[mask].tolist()
                })

            elif self.criteria.angle is None:
                # Atom pair distances
                # Using Atoms.dists for performance
                _, dists = frame.dists(combs=self.results.pairs, mic=mic)
                mask = dists <= self.dist_thresholds
                self.results.dimers.append({
                    "frame": f_idx,
                    "pairs": self.results.pairs[mask].tolist(),
                    "dists": dists[mask].tolist()
                })

            else:
                # H-bonds: Distance D...A and Angle D-H...A
                _, da_dists = frame.dists(combs=self.results.pairs[:, [0, 2]], mic=mic)
                _, dha_angles = frame.angles(self.results.pairs, mic=mic)
                
                # Check for flexible angle criterion
                if isinstance(self.criteria.angle, str) and self.criteria.angle.endswith('f'):
                    scale_ang = float(self.criteria.angle[:-1])
                    _, dh_dists = frame.dists(combs=self.results.pairs[:, [0, 1]], mic=mic)
                    # Instantaneous threshold calculation
                    syms = self.traj.frames[0].symbs
                    r_ref = np.array([data.val_R[syms[p[0]]] + data.val_R[syms[p[1]]] for p in self.results.pairs])
                    ang_thresholds = scale_ang * dh_dists / (1.05 * r_ref)
                else:
                    ang_thresholds = np.full(n_pairs, float(self.criteria.angle))
                
                mask = (da_dists <= self.dist_thresholds) & (dha_angles >= ang_thresholds)
                self.results.dimers.append({
                    "frame": f_idx,
                    "d_h_a_pairs": self.results.pairs[mask].tolist(),
                    "d_a_dists": da_dists[mask].tolist(),
                    "d_h_a_angles": dha_angles[mask].tolist()
                })

            update_progress(f_idx / n_frames)

        if filename is not None:
            import json
            import pickle
            import gzip
            
            if filename.endswith(".gz"):
                with gzip.open(filename, "wb") as f:
                    pickle.dump(self.results.dimers, f)
            elif filename.endswith(".pkl"):
                with open(filename, "wb") as f:
                    pickle.dump(self.results.dimers, f)
            else:
                with open(filename, "w") as json_file:
                    json.dump(self.results.dimers, json_file, indent=4)
        update_progress(1)

    def summary(self):
        """Print a scientific interpretation of the applied geometrical criteria."""
        print("="*75)
        print(" GEOMETRICAL DIMER RESOLUTION SUMMARY")
        print("="*75)
        print(f" Frames Processed : {len(self.results.dimers)}")
        
        mode = "Centre-of-Mass (CoM)" if self.com else ("H-Bond (D-H...A)" if self.criteria.angle else "Atom-Pair Distance")
        print(f" Resolution Mode  : {mode}")
        print(f" Distance Cutoff  : {self.criteria.distance}")
        if self.criteria.angle:
            print(f" Angular Cutoff   : {self.criteria.angle}")
            
        print("-" * 75)
        print(" INTERPRETATION GUIDE:")
        if self.com:
            print(" - Using CoM distance to track the clustering/pairing of entire molecules.")
            print("   Robust against internal molecular vibrations, but ignores relative orientation.")
        elif self.criteria.angle:
            print(" - Resolving Hydrogen Bonds using a strict two-parameter (distance & angle) geometric definition.")
            print("   Ensure your cutoffs match the first minimum of the relevant RDF and ADF to capture the true")
            print("   first coordination shell without including non-bonded interstitial water.")
        else:
            print(" - Tracking simple pairwise atomic contacts. Useful for studying specific coordination")
            print("   shells (e.g., Ion-Oxygen coordination in electrolytes).")
        print("="*75)


class DLDDF(object):
    """Dimer Lifetime–Displacement Distribution Function (DLDDF).

    The DLDDF correlates the lifetime of a specific molecular pair (e.g., a 
    hydrogen bond or a cluster) with its net displacement during that lifetime.
    It is computed as a normalized 2D probability density histogram:

        P(τ, Δr) = N_events(τ, Δr) / [ N_total * Δτ * Δ(Δr) ]

    Unlike chunk-based methods, this implementation uses a Continuous Event 
    Tracking algorithm. It tracks every dimer formation from the exact frame it 
    forms to the exact frame it breaks, ensuring completely unbiased lifetime 
    and displacement statistics.

    Key Physical Applications:
    - **Hydrogen Bonding**: Correlating H-bond lifetime with the migration distance of the participating molecules (useful for Grotthuss vs vehicular mechanisms).
    - **Clustering**: Tracking how far a solute-solvent cage moves before it completely dissolves.

    Parameters
    ----------
    traj : Trajectory
        Trajectory object containing the frames to analyse.
    dimer_settings : dict, optional
        Keyword arguments to pass directly to the `dimers` class constructor 
        (e.g., `{'at_g1': ..., 'at_g2': ..., 'criteria': ...}`).
    nbins : int, optional
        Number of histogram bins for both axes. Default: 100.
    range : array-like, optional
        [ (τ_min, τ_max), (Δr_min, Δr_max) ]. Default: [(0.0, 2.0), (0.0, 5.0)].

    Results dataclass fields
    ------------------------
    dlddf    : ndarray, 2D — Normalized probability density histogram.
    r_edges, tau_edges : ndarrays — Bin boundaries.
    r, tau   : ndarrays — Bin centres.
    taus     : list — Raw recorded lifetimes for all events.
    dists    : list — Raw recorded displacements for all events.
    dimers   : list of dict — Serialized dimer list per frame.
    plot     : Figure — The generated 2D contour plot.
    """
    
    def __init__(self, traj, dimer_settings = None, nbins = 100, range = np.array([(0.0, 2.0), (0.0, 5.0)])):
        self.traj = traj.wrap2box()
        self.dimer_settings = dimer_settings
        settings_dc = make_dataclass("settings", "bins range")
        self.settings = settings_dc(nbins, range)
        results_cls = make_dataclass("Results", "dimers r_edges tau_edges r tau dists taus dlddf plot")
        self.results = results_cls(
            dimers=[], r_edges=None, tau_edges=None, r=None, tau=None, 
            dists=[], taus=[], dlddf=None, plot=None
        )
        # Initialize histogram placeholders
        H, re, te = np.histogram2d([-1], [-1], bins=self.settings.bins, range=self.settings.range)
        self.results.dlddf = H
        self.results.r_edges = re
        self.results.tau_edges = te
        self.results.r = (re[:-1] + re[1:]) / 2
        self.results.tau = (te[:-1] + te[1:]) / 2
        
    def dimer_res(self, mic=True, filename=None):
        """Resolve dimers and store results."""
        dimers_data = dimers(self.traj, **self.dimer_settings)
        dimers_data.make_pairs()
        dimers_data.pair_filter(mic=mic, filename=filename)
        self.results.dimers = dimers_data.results.dimers
    
    def calculate(self, mic=True, plot=False, levels=25, xlims=None, ylims=None, filename=None, **kwargs):
        """Compute the DLDDF 2-D histogram using continuous event tracking.

        Unlike the original chunked approach, this implementation tracks every 
        dimer formation event from its birth to its death, ensuring correct 
        lifetimes and displacements without sampling bias or truncation.
        """
        # Determine the data key in the dimer records
        if not self.results.dimers:
            raise ValueError("No dimer data found. Call dimer_res() first.")
        
        key = "pairs" if "pairs" in self.results.dimers[0] else "d_h_a_pairs"
        n_frames = len(self.results.dimers)
        dt_ps = self.traj.timestep / 1000.0
        
        # Track active events: { tuple(pair_indices): start_frame_index }
        active_events = {}
        recorded_taus = []
        recorded_dists = []

        for f_idx in range(n_frames):
            current_dimers = self.results.dimers[f_idx][key]
            # Convert current list of lists to a set of tuples for O(1) lookup
            current_set = {tuple(p) if isinstance(p, list) else (p,) for p in current_dimers}
            
            # 1. Identify dimers that just broke
            broken = [p for p in active_events if p not in current_set]
            for p in broken:
                start_f = active_events.pop(p)
                end_f = f_idx - 1 # Last frame it was active
                
                if end_f >= start_f:
                    lifetime = (end_f - start_f + 1) * dt_ps
                    # Displacement of the pair's centre of mass
                    # Note: p can be (atom1, atom2) or (D, H, A)
                    p_indices = list(p)
                    com_start = self.traj.frames[start_f][p_indices].calc_com()
                    com_end = self.traj.frames[end_f][p_indices].calc_com()
                    
                    if mic:
                        dist = mic_dist(com_start, com_end, cell=self.traj.cell)
                    else:
                        dist = np.linalg.norm(com_end - com_start)
                    
                    recorded_taus.append(lifetime)
                    recorded_dists.append(dist)
            
            # 2. Identify dimers that just formed
            for p in current_set:
                if p not in active_events:
                    active_events[p] = f_idx
            
            update_progress(f_idx / n_frames)

        # Close any events still active at end of trajectory
        for p, start_f in active_events.items():
            end_f = n_frames - 1
            lifetime = (end_f - start_f + 1) * dt_ps
            p_indices = list(p)
            com_start = self.traj.frames[start_f][p_indices].calc_com()
            com_end = self.traj.frames[end_f][p_indices].calc_com()
            dist = mic_dist(com_start, com_end, cell=self.traj.cell) if mic else np.linalg.norm(com_end - com_start)
            recorded_taus.append(lifetime)
            recorded_dists.append(dist)

        self.results.taus = recorded_taus
        self.results.dists = recorded_dists
        
        # 3. Binning
        # density=True normalizes the integral over the range to 1
        H, xedges, yedges = np.histogram2d(recorded_taus, recorded_dists, 
                                           bins=self.settings.bins, range=self.settings.range, density=True)
        self.results.dlddf = H.T # Transpose for plot orientation (rows=r, cols=tau)
        self.results.tau_edges, self.results.r_edges = xedges, yedges
        self.results.tau = (xedges[:-1] + xedges[1:]) / 2
        self.results.r = (yedges[:-1] + yedges[1:]) / 2

        if plot:
            fig, ax = plt.subplots(figsize=(4.8, 4.0))
            X, Y = np.meshgrid(self.results.tau, self.results.r)
            levels_arr = np.linspace(self.results.dlddf.min(), self.results.dlddf.max(), levels)
            CS = ax.contourf(X, Y, self.results.dlddf, cmap=color_ramp, levels=levels_arr, **kwargs)
            ax.contour(X, Y, self.results.dlddf, colors="k", levels=levels_arr, linewidths=0.1, alpha=0.5)

            ax.set_xlabel(r"Lifetime $\tau$ (ps)")
            ax.set_ylabel(r"Displacement $\Delta r$ ($\mathrm{\AA}$)")
            if xlims: ax.set_xlim(xlims)
            if ylims: ax.set_ylim(ylims)

            cbar = plt.colorbar(CS, pad=0.035, ax=ax)
            cbar.set_label("Probability Density")
            
            if filename:
                plt.savefig(filename, dpi=600, bbox_inches='tight')
            plt.show()
            self.results.plot = fig

        update_progress(1.0)
        return self.results

    def summary(self):
        """Print a scientific interpretation of the computed DLDDF."""
        print("="*75)
        print(" DIMER LIFETIME-DISPLACEMENT DISTRIBUTION (DLDDF) SUMMARY")
        print("="*75)
        
        total_events = len(self.results.taus)
        mean_tau = np.mean(self.results.taus) if total_events > 0 else 0
        mean_dist = np.mean(self.results.dists) if total_events > 0 else 0
        
        print(f" Total Dimer Events Tracked : {total_events}")
        print(f" Mean Dimer Lifetime (tau)  : {mean_tau:.4f} ps")
        print(f" Mean Dimer Displacement    : {mean_dist:.4f} Å")
        
        print("-" * 75)
        print(" INTERPRETATION GUIDE:")
        print(" - The 2D contour maps the joint probability of a dimer surviving for time 'tau'")
        print("   and traveling a net distance 'dr' during that exact lifetime.")
        print(" - Events clustered near tau~0, dr~0 represent short-lived, localized collisions.")
        print(" - A population stretching out along the dr axis with long lifetimes indicates")
        print("   stable pairs migrating together (e.g., vehicular transport mechanism).")
        print(" - Fast-decaying taus with large dr suggest rapid exchange or Grotthuss-like hopping.")
        print("="*75)


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

            # r_d_a = frame.dist(*d_a_pair[0].values(), *d_a_pair[1].values(), mic = True)
            # Use the optimized dists call
            da_idx = [list(d_a_pair[0].values())[0], list(d_a_pair[1].values())[0]]
            r_d_a = frame.dist(da_idx[0], da_idx[1], mic=True)
            
            if r_d_a <= 1.02 * vdW_sum:
                # DHA triplet
                dha_idx = [list(d_h_a_pair[0].values())[0], list(d_h_a_pair[0].values())[1], list(d_h_a_pair[1].values())[0]]
                d_h_a_ang = frame.angle(*dha_idx, mic=True)
                d_h = frame.dist(dha_idx[0], dha_idx[1], mic=True)

                if d_h_a_ang >= (np.rad2deg(np.arctan2(r_d_a, d_h)) + 180)*3/8:
                    hbonds.append({
                        "donor": d_h_a_pair[0], "acceptor": d_h_a_pair[1],
                        "R(D-A)": r_d_a, "DHA_ang": d_h_a_ang, "r(D-H)": d_h,
                    })
        results.append({f"frame": i, "n_hbonds": len(hbonds), "hbonds": hbonds})
        update_progress(i/len(traj.frames))
    return results


from scipy.special import gamma

def bi_exp(t, amp1, tau1, tau2):
    """Biexponential decay model C(t) = A·exp(−t/τ₁) + (1−A)·exp(−t/τ₂)."""
    return amp1*np.exp(-t/tau1) + (1-amp1)*np.exp(-t/tau2)


def bi_exp_fit(x, y, amp1=0.5, tau1=100, tau2=150):
    """Fit a biexponential decay to H-bond autocorrelation data.
    
    Returns the dense fitting arrays (x_fit, y_fit) and the optimal 
    parameters [A, τ₁, τ₂].
    """
    params, _ = curve_fit(bi_exp, x, y, p0=[amp1, tau1, tau2], bounds=([0, 0, 0], [1, np.inf, np.inf]))
    x_fit = np.linspace(x.min(), x.max(), num=200)
    y_fit = bi_exp(x_fit, *params)
    
    print("\n" + "="*75)
    print(" BIEXPONENTIAL FIT SUMMARY (Structural Relaxation)")
    print("="*75)
    print(" Model: C(t) = A * exp(-t/tau1) + (1-A) * exp(-t/tau2)")
    print(f" A    (Fast fraction) : {params[0]:.4f}")
    print(f" tau1 (Fast time)     : {params[1]:.4f} ps")
    print(f" tau2 (Slow time)     : {params[2]:.4f} ps")
    print("-" * 75)
    print(" INTERPRETATION GUIDE:")
    print(" - 'tau1' usually represents fast librational breaking or local rattling.")
    print(" - 'tau2' represents the true structural relaxation (cage escape).")
    print(" - If A is very small, the dynamics are mostly single-exponential.")
    print("="*75)
    
    return x_fit, y_fit, params


def kww_exp(t, tau, beta):
    """Kohlrausch-Williams-Watts (KWW) stretched exponential C(t) = exp(−(t/τ)^β)."""
    return np.exp(-np.power(np.maximum(t, 0) / tau, beta))


def kww_fit(x, y, tau=None, beta=0.5):
    """Fit a KWW stretched exponential to intermittent H-bond autocorrelation data.
    
    Returns the dense fitting arrays (x_fit, y_fit) and the optimal 
    parameters [τ, β].
    """
    if tau is None:
        idx_e = np.where(y < 1/np.e)[0]
        tau = x[idx_e[0]] if len(idx_e) > 0 else x[-1]

    params, _ = curve_fit(kww_exp, x, y, p0=[tau, beta], bounds=([0, 0], [np.inf, 1.0]))
    x_fit = np.linspace(x.min(), x.max(), num=200)
    y_fit = kww_exp(x_fit, *params)
    
    print("\n" + "="*75)
    print(" KWW FIT SUMMARY (Reactive / Intermittent Relaxation)")
    print("="*75)
    print(" Model: C(t) = exp(-(t/tau)^beta)")
    print(f" tau  (Relaxation time) : {params[0]:.4f} ps")
    print(f" beta (Stretching exp)  : {params[1]:.4f}")
    print("-" * 75)
    print(" INTERPRETATION GUIDE:")
    print(" - 'tau' represents the characteristic timescale for a bond to permanently break.")
    print(" - 'beta' < 1 indicates a wide distribution of relaxation times (heterogeneity),")
    print("   which is typical for intermittent dynamics where bonds can break and reform.")
    print("   The lower the beta, the longer the 'tail' of surviving bonds.")
    print("="*75)
    
    return x_fit, y_fit, params


def calc_tau(model, *params):
    """Compute the effective H-bond lifetime analytically.
    
    Parameters
    ----------
    model : str
        Either 'biexp' or 'kww'.
    *params : floats
        For 'biexp': amp1, tau1, tau2
        For 'kww': tau, beta
    """
    if model.lower() == 'biexp':
        amp1, tau1, tau2 = params
        res = amp1 * tau1 + (1 - amp1) * tau2
    elif model.lower() == 'kww':
        tau, beta = params
        res = (tau / beta) * gamma(1.0 / beta)
    else:
        raise ValueError("Model must be 'biexp' or 'kww'.")

    print("\n" + "="*75)
    print(" EFFECTIVE LIFETIME SUMMARY")
    print("="*75)
    print(f" Integrated Lifetime (tau_eff) : {res:.4f} ps")
    print(" - Represents the average total survival time of the bond.")
    print("="*75)
    return res


def auto_cor(data="hbonds.json", nframes=1000, sampling=1, skip=10, index=":", timestep=None, intermittent=False):
    """Compute the H-bond survival autocorrelation function C(t) from a JSON file.

    The function evaluates either the continuous or intermittent survival 
    autocorrelation:

        C(t) = ⟨h(t_0) · h(t_0 + t)⟩ / ⟨h(t_0)²⟩

    - **Continuous** (default): h(t_0 + t) is 1 only if the bond has remained 
      intact without breaking from t_0 to t_0 + t. Captures structural relaxation.
    - **Intermittent**: h(t_0 + t) is 1 if the bond exists at t_0 and t_0 + t, 
      regardless of whether it broke in between. Captures reactive lifetimes.

    Parameters
    ----------
    data : str or list of dict
        Path to the JSON file containing the H-bond list, or the list itself.
    nframes : int, optional
        Correlation window length in frames. Default: 1000.
    sampling : int, optional
        Stride for lag times within the window. Default: 1.
    skip : int, optional
        Stride between time origins (t_0) to improve statistics. Default: 10.
    index : str or int or list of int, optional
        Frame selection slice. Default: ":".
    timestep : float, optional
        Trajectory time step in fs. Default: 5.0 fs.
    intermittent : bool, optional
        If True, compute the intermittent autocorrelation. Default: False.

    Returns
    -------
    t : ndarray
        Time axis in ps.
    C_t : ndarray
        Ensemble-averaged autocorrelation values.
    C_t_error : ndarray
        Standard deviation of the autocorrelation across different time origins.
    """
    # 1. Load and slice data
    if isinstance(data, str):
        if not os.path.exists(data):
            raise FileNotFoundError(f"File {data} not found.")
        if os.path.getsize(data) < 40:
            raise EOFError(f"File {data} is too small to contain dimer data.")

        if data.endswith(".gz"):
            import gzip
            import pickle
            with gzip.open(data, 'rb') as f:
                frames_raw = pickle.load(f)
        elif data.endswith(".pkl"):
            import pickle
            with open(data, 'rb') as f:
                frames_raw = pickle.load(f)
        else:
            with open(data, 'r') as f:
                frames_raw = json.load(f)
    else:
        frames_raw = data

    if isinstance(index, int):
        frames_raw = frames_raw[:index]
    elif isinstance(index, (list, tuple)):
        frames_raw = frames_raw[index[0]:index[1]]
    
    dt = timestep if timestep else 5.0
    
    # 2. Pre-process frames into sets of canonical tuples for O(1) lookups
    sample = frames_raw[0]
    if "hbonds" in sample:
        key = "hbonds"
        def canon(b): return tuple(deepflatten(list(b["donor"].values()) + list(b["acceptor"].values())))
    elif "d_h_a_pairs" in sample:
        key = "d_h_a_pairs"
        def canon(b): return tuple(b)
    elif "pairs" in sample:
        key = "pairs"
        def canon(b): return tuple(b)
    else:
        raise ValueError("Unrecognised JSON format.")

    processed_frames = [{canon(b) for b in f[key]} for f in frames_raw]

    # 3. Compute Autocorrelation with Sliding Window
    total_frames = len(processed_frames)
    t0_indices = np.arange(0, total_frames - nframes + 1, skip)
    lags = np.arange(0, nframes, sampling)
    
    if len(t0_indices) == 0:
        raise ValueError(f"Trajectory ({total_frames} frames) too short for window {nframes}.")

    results_list = []

    for t0 in t0_indices:
        active_0 = processed_frames[t0]
        n0 = len(active_0)
        if n0 == 0: continue
        
        c_t = np.zeros(len(lags))
        current_continuous = active_0.copy()
        
        # We must check every frame between lags for continuous mode
        # to ensure it hasn't broken in the gaps created by sampling.
        last_lag = 0
        for i, lag in enumerate(lags):
            if not intermittent:
                # Check all frames between the last sampled lag and current lag
                for step in range(last_lag + 1, lag + 1):
                    current_continuous &= processed_frames[t0 + step]
                c_t[i] = len(current_continuous) / n0
                last_lag = lag
            else:
                # Intermittent only cares about the current frame
                c_t[i] = len(active_0 & processed_frames[t0 + lag]) / n0
        
        results_list.append(c_t)
        update_progress(len(results_list) / len(t0_indices))

    if not results_list:
        raise ValueError("No bonds found at any time origins.")

    # 4. Statistics and Time Axis
    results_arr = np.asarray(results_list)
    C_ts_mean = results_arr.mean(axis=0)
    # Return standard deviation, not standard error, because the number of sliding
    # windows (N) is very large, making standard error artificially small and 
    # hiding the true fluctuation of the system.
    C_ts_error = results_arr.std(axis=0)
    
    t = lags * dt / 1000.0
    
    update_progress(1.0)
    
    print("\n" + "="*75)
    print(" SURVIVAL AUTOCORRELATION FUNCTION SUMMARY")
    print("="*75)
    print(f" Analysis Mode  : {'Intermittent (Reactive)' if intermittent else 'Continuous (Structural)'}")
    print(f" Time Windows   : {len(results_list)} time origins averaged")
    print(f" Max Lag Time   : {t[-1]:.2f} ps")
    print("-" * 75)
    print(" INTERPRETATION GUIDE:")
    if intermittent:
        print(" - Intermittent Mode: C(t) drops only if the bond breaks and NEVER reforms.")
        print(" - Captures the true macroscopic chemical lifetime (reactive relaxation).")
        print(" - Fits best to a multi-exponential or power-law decay.")
    else:
        print(" - Continuous Mode: C(t) drops the moment a bond breaks for the first time.")
        print(" - Captures the local structural relaxation (vibrational + librational breakage).")
        print(" - Fits best to a fast, single or double exponential decay.")
    print("="*75)
    
    return t, C_ts_mean, C_ts_error



