# FishMol

A pure-Python package for post-processing molecular dynamics trajectories.

```
                  Welcome!                    ▄▄█▀                  FishMol
                                          ▄▄███▀                 version 0.0.1
      ○                                ▄▄█████▀                          ○
           ○                        ▄▄████████▄                         /
                                ▄▄▄████████████▄                    ○--○
         ○                ▄▄▄████████████████████▄▄                     \ 
                    ▄▄▄██████▀███████████████████████▄▄                  ○--○           ▄
                ▄▄████████████ ██████████████████████████▄▄             /           ▄▄█▀
         ○   ▄█████████████████ ████████████████████████████▄▄         ○         ▄███▀
          ▄██████████▀▀▀████████ ██████████████████████████████▄▄             ▄█████▀
         ▄████████▀   ○  ▀███████ █████████████████████████████████▄▄      ▄▄█████▀
         ■▄███████▄      ▄███████ █████████████████████████████████████▄▄▄███████▀
          ▀█████████▄▄▄█████████ ███████████████████████████████████████████████
            ▀██████████████████ ████████████████████████████████████████████████▄
              ▀██████████████▀▄███████████████████████████████████▀▀▀▀▀████████████▄
                ▀▀█████████▀▄███████████████████████████████▀▀           ▀▀████████▀
                   ▀▀█████▄████████████████████████▀▀▀▀██▀                 ▀▀████▀
                        ▀▀▀█████████████████▀▀▀▀        ▀■                    ▀█▀
                             ▀▀▀███████▀▀
                                 ▀████
                                    ▀▀▄                Contact: Lei.Lei@durham.ac.uk
```

## Features

- **Trajectory I/O** — fast memory-mapped reading of extended XYZ files; write filtered trajectories back to disk
- **PBC-aware geometry** — distances, angles, dihedrals, and vectors under the minimum image convention for triclinic cells
- **Molecule recognition** — automatic clustering of atoms into molecules via a covalent-radius bonding graph
- **Distribution functions** — radial (RDF), angular (ADF), dihedral (DDF), and combined 2-D (CDF) distribution functions
- **MSD & diffusion** — FFT-based mean square displacement; self-diffusion coefficient via the Einstein relation
- **Diffusion anisotropy** — spherical projection of 1-D MSD to map the directional diffusion surface; Voronoi channel assignment
- **Hydrogen bonds** — frame-by-frame H-bond recognition with distance + angle criteria; biexponential lifetime from autocorrelation
- **Vector Reorientation Dynamics** — Legendre polynomial autocorrelation (*l* = 1, 2, 3) with KWW stretched-exponential fitting
- **Dimer analysis** — dimer lifetime–displacement distribution function (DLDDF)
- **Visualisation** — interactive 3-D viewer embedded in Jupyter via the ASE backend

## Requirements

- Python ≥ 3.8
- [NumPy](https://numpy.org/)
- [SciPy](https://scipy.org/)
- [matplotlib](https://matplotlib.org/)
- [pandas](https://pandas.pydata.org/)
- [recordclass](https://pypi.org/project/recordclass/)
- [iteration_utilities](https://pypi.org/project/iteration-utilities/)
- [colour](https://pypi.org/project/colour/)
- [ASE](https://wiki.fysik.dtu.dk/ase/) (for visualisation)

## Installation

FishMol is in active development. Install directly from source:

```bash
# 1. Clone the repository
git clone https://github.com/Lei-Lei-alpha/fishmol.git
cd fishmol

# 2. Install in editable mode (dependencies resolved automatically)
pip install -e ./

# 3. Verify the installation
fishmol
```

## Quick Start

```python
from fishmol import trj, funcs

# Define the simulation cell (lattice vectors as rows, in Å)
cell = [
    [21.2944,  0.0000,  0.0000],
    [-4.6030, 20.7909,  0.0000],
    [-0.9719, -1.2106, 15.1054],
]

# Load a trajectory (5 fs timestep, all frames)
traj = trj.Trajectory(timestep=5, data="trajectory.xyz", index=":", cell=cell)

# Calibrate centre-of-mass drift, wrap into box
traj.calib().wrap2box()

# Compute radial distribution function between O and H atoms
o_idx = [i for i, s in enumerate(traj.frames[0].symbs) if s == "O"]
h_idx = [i for i, s in enumerate(traj.frames[0].symbs) if s == "H"]
rdf = funcs.RDF(traj, at_g1=o_idx, at_g2=h_idx, nbins=200, range=(0.0, 8.0))
results = rdf.calculate(plot=True)
```

## Documentation

Full worked examples are in the [online documentation](https://lei-lei-alpha.github.io/fishmol):

| Notebook | Description |
|----------|-------------|
| [Trajectory I/O](https://lei-lei-alpha.github.io/fishmol/trajectory_IO.html) | Reading, viewing, calibrating, wrapping, writing, and filtering trajectories; automatic molecule recognition |
| [MSD & Diffusion Coefficient](https://lei-lei-alpha.github.io/fishmol/MSD_diff_coeff.html) | FFT-based MSD, Einstein relation, unit conversion, dual-axis MSD/*D* plot |
| [Diffusion Anisotropy](https://lei-lei-alpha.github.io/fishmol/diff_aniso.html) | Spherical MSD projection, 3-D anisotropy surface, Voronoi channel analysis |
| [Hydrogen Bond Lifetime](https://lei-lei-alpha.github.io/fishmol/H_bond.html) | H-bond recognition, geometry heatmap, biexponential lifetime fitting |
| [Vector Reorientation Dynamics](https://lei-lei-alpha.github.io/fishmol/VRD.html) | Legendre autocorrelation (*l* = 1–3), KWW fitting, O–H and C–C bond examples |
