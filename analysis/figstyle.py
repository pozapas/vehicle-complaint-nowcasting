"""Single matplotlib style module for all four figures (outline "Figure style guide").

Nature-journal conventions: Helvetica/Arial only, 7-8.5 pt text, lowercase bold
panel labels top-left, hairline spines, no chart junk. No figure script may
override any of this locally.
"""

from __future__ import annotations

import os

import matplotlib as mpl
import matplotlib.pyplot as plt

from common import FIG_DIR, GRAY_DIR

# ---------------------------------------------------------------- palette (fixed)
SEV_COLORS = {
    0: "#8C8C8C",   # Y0 none         -  gray
    1: "#4DBBD5",   # Y1 crash/fire   -  blue
    2: "#E18727",   # Y2 injury       -  orange
    3: "#BC3C29",   # Y3 fatality     -  red
}
NOWCAST = "#0072B5"   # deep navy
NAIVE = "#D9D9D9"     # light gray
INK = "#1A1A1A"
MUTED = "#666666"

# Redundant (non-colour) encoding of the severity classes.
# The palette is colourblind-safe but NOT grayscale-separable: converted to
# luminance the classes land at Y0 140, Y1 157, Y2 151, Y3 96 (see
# `grayscale_luminance()`), so Y0/Y1/Y2 are within 17 levels of one another.
# Any panel that distinguishes severity classes must therefore also vary line
# style, marker or hatch. These dicts are the single source of that mapping.
SEV_LS = {0: "-", 1: (0, (1, 1.2)), 2: (0, (4, 1.5)), 3: (0, (6, 1.5, 1, 1.5))}
SEV_MARKER = {0: "o", 1: "s", 2: "^", 3: "D"}
SEV_HATCH = {0: "", 1: "////", 2: "....", 3: "xxxx"}


def grayscale_luminance():
    """Rec.601 luminance (0-255) of each severity colour, as PIL's convert('L')."""
    out = {}
    for s, hexc in SEV_COLORS.items():
        r, g, b = (int(hexc[i:i + 2], 16) for i in (1, 3, 5))
        out[s] = round(0.299 * r + 0.587 * g + 0.114 * b)
    return out

# sequential heatmap shading: light gray -> navy
HEATMAP_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "graynavy", ["#F5F5F5", "#C9DCEA", "#7FAFD0", "#0072B5", "#00436B"]
)

# ---------------------------------------------------------------- widths
W1 = 3.5   # single column, inches
W2 = 7.2   # double column, inches

_FONT_STACK = ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]


def apply():
    """Install the shared rcParams. Called at import time."""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": _FONT_STACK,
        "font.size": 7.0,
        "axes.titlesize": 8.0,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 7.0,
        "figure.titlesize": 8.5,

        "axes.linewidth": 0.5,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "grid.linewidth": 0.25,
        "grid.color": "#D0D0D0",
        "grid.alpha": 1.0,

        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "xtick.major.size": 2.4,
        "ytick.major.size": 2.4,
        "xtick.minor.size": 1.4,
        "ytick.minor.size": 1.4,
        "xtick.direction": "out",
        "ytick.direction": "out",

        "lines.linewidth": 1.0,
        "lines.markersize": 3.0,
        "patch.linewidth": 0.5,

        "legend.frameon": False,
        "legend.handlelength": 1.4,
        "legend.handletextpad": 0.5,
        "legend.labelspacing": 0.3,
        "legend.borderpad": 0.2,

        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,      # embed TrueType, editable text
        "ps.fonttype": 42,
        "text.color": INK,
    })


apply()


def panel_label(ax, letter, dx=-0.085, dy=1.06, fontsize=8.5):
    """Lowercase bold panel label at the top-left of an axes."""
    ax.text(dx, dy, letter, transform=ax.transAxes,
            fontsize=fontsize, fontweight="bold", va="bottom", ha="left",
            color=INK)


def light_grid(ax, axis="y"):
    ax.set_axisbelow(True)
    ax.grid(True, axis=axis, linewidth=0.25, color="#D8D8D8")


def save_fig(fig, name):
    """Write vector PDF, 600-dpi PNG and a grayscale proof. Returns paths."""
    pdf = os.path.join(FIG_DIR, f"{name}.pdf")
    png = os.path.join(FIG_DIR, f"{name}.png")
    fig.savefig(pdf)
    fig.savefig(png, dpi=600)
    gray = _grayscale_proof(png, name)
    plt.close(fig)
    return {"pdf": pdf, "png": png, "gray": gray}


def _grayscale_proof(png, name):
    out = os.path.join(GRAY_DIR, f"{name}_gray.png")
    try:
        from PIL import Image
        im = Image.open(png).convert("L")
        im.save(out)
        return out
    except Exception:                                    # pragma: no cover
        import numpy as np
        import matplotlib.image as mpimg
        a = mpimg.imread(png)
        if a.ndim == 3:
            g = a[..., :3] @ np.array([0.2126, 0.7152, 0.0722])
            mpimg.imsave(out, g, cmap="gray")
            return out
    return None


def sev_handles(classes=(0, 1, 2, 3), labels=None):
    """Legend handles in the fixed severity palette."""
    from matplotlib.lines import Line2D
    from common import SEV_LABELS
    labels = labels or SEV_LABELS
    return [Line2D([0], [0], color=SEV_COLORS[s], lw=1.2, label=labels[s])
            for s in classes]
