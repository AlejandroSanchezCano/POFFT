"""
===============================================================================
Title:      Style module
Outline:    Style settings for matplotlib and seaborn to ensure consistent
            and publication-quality plots.
Docs:       https://matplotlib.org/stable/users/explain/customizing.html
Author:     Alejandro Sánchez Cano
Date:       28/08/2026
===============================================================================
"""

# Third-party modules
import matplotlib as mpl

# Matplotlib
mpl.rcParams.update({
        # Figure
        "figure.figsize": (4.5, 3.5),  # inches, single-column journal size
        "figure.dpi": 300,             # High resolution
        "savefig.dpi": 600,            # Even higher for saving
        "savefig.bbox": "tight",
        "savefig.transparent": False,
        "savefig.format": "pdf",

        # Fonts
        "font.family": "sans-serif",       
        #"font.serif": ["Times New Roman", "Times", "Computer Modern Roman"],
        "font.size": 9,                # Adjust for readability
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,

        # Axes
        "axes.linewidth": 0.8,
        "axes.labelpad": 4,
        "axes.titlepad": 6,

        # Ticks
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": False,
        "ytick.right": False,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,

        # Grid
        "axes.grid": False,

        # Lines
        "lines.linewidth": 1.0,
        "lines.markersize": 4,

        # Legend
        "legend.frameon": False,
        "legend.fontsize": 16,
        "legend.title_fontsize": 16,

    })

# Seaborn
import seaborn as sns
sns.set_palette("deep") 
sns.color_palette("viridis", as_cmap=True)
sns.set_context("paper")  # ("paper", "notebook", "talk", "poster") scales label size, line thickness...