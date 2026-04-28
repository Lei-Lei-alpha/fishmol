# matplotlib plot style file

import matplotlib as mpl
from matplotlib import rcParams
from cycler import cycler
from colour import Color
from matplotlib.colors import LinearSegmentedColormap

#----If you want to use LaTex----#
# mpl.rcParams["text.usetex"]=True
# params= {'text.latex.preamble' : [r'\usepackage{amsmath, amssymb, unicode-math}\usepackage[dvips]{graphicx}\usepackage{xfrac}\usepackage{amsbsy}']}


#----Colour cycle used----#
# plot_colours = ['#008fd5', '#fc4f30', '#e5ae38', '#6d904f', '#8b8b8b', '#810f7c']

#----Colour cycle used----#
ramp_colours = ["#ffffff", "#9ecae1", "#2166ac", "#1a9850", "#ffff33", "#b2182b", "#67000d"]
cdf_cmap = LinearSegmentedColormap.from_list( 'cdf_colour', [ Color( c1 ).rgb for c1 in ramp_colours ] )

# Register the colormap to avoid unhashable type errors in Matplotlib internal lookups
try:
    mpl.colormaps.register(cdf_cmap, name='cdf_colour', force=True)
except AttributeError:
    # Fallback for older matplotlib versions
    import matplotlib.cm as cm
    cm.register_cmap(name='cdf_colour', cmap=cdf_cmap)

#----Plot style----#
rcParams.update({
    "font.size": 13,
    "font.family": "sans-serif",
    "lines.markersize": 5,
    "image.cmap": "cdf_colour", # Use the name of the registered colormap

    # "font.sans-serif": "Arial",
    # "font.weight": "heavy",
    # "axes.prop_cycle": cycler(color = plot_colours),
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "figure.figsize": (4.2,3.6),
    "figure.subplot.left": 0.21,
    "figure.subplot.right": 0.96,
    "figure.subplot.bottom": 0.18,
    "figure.subplot.top": 0.93,
    "legend.frameon": False,
})
