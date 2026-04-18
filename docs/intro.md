# FishMol Documentation

**FishMol** is a pure-Python package for analysing molecular dynamics (MD) trajectories. It is designed for researchers working with periodic systems and aims to make common post-processing tasks — radial distribution functions, diffusion coefficients, hydrogen-bond lifetimes, vector reorientation dynamics — accessible with a clean, composable API.

## Key Features

| Feature | Module |
|---------|--------|
| Fast trajectory I/O (XYZ format, memory-mapped) | `fishmol.trj` |
| Atom/molecule selection and PBC-aware geometry | `fishmol.atoms` |
| Automatic molecule recognition by bond topology | `fishmol.sel_tools` |
| RDF, ADF, DDF, CDF distribution functions | `fishmol.funcs` |
| Vector Reorientation Dynamics (Legendre *l* = 1, 2, 3) with KWW fitting | `fishmol.funcs` |
| MSD and diffusion coefficient (FFT algorithm) | `fishmol.msd` |
| Diffusion anisotropy via spherical MSD projection | `fishmol.msd` |
| Hydrogen-bond recognition and lifetime autocorrelation | `fishmol.dimer` |
| Dimer lifetime–displacement distribution function | `fishmol.dimer` |
| Voronoi-based diffusion channel assignment | `fishmol.utils` |
| Interactive 3-D visualisation (ASE backend) | `fishmol.vis` |

## Tutorial Notebooks

The following notebooks walk through complete analysis workflows using a model cage-compound system with water molecules, trifluoroacetate anions, phenols, and amines.

```{tableofcontents}
```

## Quick Install

```bash
git clone https://github.com/Lei-Lei-alpha/fishmol.git
cd fishmol
pip install -e ./
```

Then verify:

```bash
fishmol
```

## Contact

Lei Lei — Lei.Lei@durham.ac.uk
