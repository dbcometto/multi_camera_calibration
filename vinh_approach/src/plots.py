"""Shared matplotlib style for result figures (light theme, for slides)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Categorical slots in fixed order (validated default palette).
SERIES = ["#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac", "#008a17", "#6a4ce2", "#ec4647"]
CRITICAL = "#dc3c3d"   # status color: rejected / failed
INK = "#191919"
INK2 = "#666666"
GRID = "#e6e6e6"


def setup():
    plt.rcParams.update({
        "figure.facecolor": "#ffffff", "axes.facecolor": "#ffffff",
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 13, "axes.titleweight": "bold", "axes.labelsize": 11,
        "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
        "axes.spines.right": False, "legend.frameon": False, "legend.fontsize": 10,
        "lines.linewidth": 2, "font.size": 11, "savefig.dpi": 150,
        "savefig.bbox": "tight",
    })
    return plt
