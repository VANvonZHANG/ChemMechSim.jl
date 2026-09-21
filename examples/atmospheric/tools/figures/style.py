"""Shared publication-figure style for the atmospheric box-model example.

Journal-final matplotlib conventions: 7 pt sans-serif text, left/bottom spines
only, frameless legends, editable embedded fonts, and a dual PNG (raster
preview, 300 dpi) + PDF (vector) export. Figure 2 / the summary table
(Task 4) reuse this module.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless batch runs: never open a GUI backend
import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use)

# --- Paths ------------------------------------------------------------------
# .../examples/atmospheric/tools/figures/style.py -> .../examples/atmospheric/
FIGURES_DIR = Path(__file__).resolve().parent
EXAMPLE_DIR = FIGURES_DIR.parent.parent
DATA = EXAMPLE_DIR / "output"  # consumed: series.csv, budget.csv, run_meta.txt
OUT = EXAMPLE_DIR / "output"   # produced: fig1_series.{png,pdf}, ...

# --- Palette -----------------------------------------------------------------
# One restrained palette per figure: neutral family for scaffolding, one
# signal family per chemical role. Never remap a species to another hue
# family between panels.
PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "green_1": "#DDF3DE",
    "green_2": "#AADCA9",
    "green_3": "#8BCF8B",
    "red_1": "#F6CFCB",
    "red_2": "#E9A6A1",
    "red_strong": "#B64342",
    "neutral_light": "#CFCECE",
    "neutral_mid": "#767676",
    "neutral_dark": "#4D4D4D",
    "neutral_black": "#272727",
    "gold": "#FFD700",
    "teal": "#42949E",
    "violet": "#9A4D8E",
    "magenta": "#EA84DD",
}

DEFAULT_COLORS = [
    PALETTE["blue_main"],
    PALETTE["green_3"],
    PALETTE["red_strong"],
    PALETTE["teal"],
    PALETTE["violet"],
    PALETTE["neutral_light"],
]


def apply_style():
    """Apply the journal-figure rcParams. Call once before creating figures."""
    plt.rcParams.update(
        {
            # Mandatory editable-text rules (always first, no exceptions)
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",  # text stays as <text> nodes, not paths
            # Journal-final dense figure regime
            "font.size": 7,
            "axes.linewidth": 0.8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
            # Editable TrueType text in PDF
            "pdf.fonttype": 42,
        }
    )


def add_panel_label(ax, label, x=-0.06, y=1.02, fontsize=8,
                    color="black", fontweight="bold"):
    """Place a Nature-style lowercase panel letter near the top-left edge."""
    ax.text(
        x, y, label,
        transform=ax.transAxes,
        fontsize=fontsize,
        fontweight=fontweight,
        color=color,
        ha="left",
        va="bottom",
    )


def save(fig, name, dpi=300):
    """Write OUT/<name>.png (quick-view raster) and OUT/<name>.pdf (vector).

    Returns the list of written paths. Closes the figure to free memory.
    """
    paths = []
    for suffix, kwargs in (("png", {"dpi": dpi}), ("pdf", {})):
        path = OUT / f"{name}.{suffix}"
        fig.savefig(path, bbox_inches="tight", **kwargs)
        paths.append(path)
    plt.close(fig)
    for p in paths:
        print(f"wrote {p}")
    return paths
